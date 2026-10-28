"""Audit stability and recency segments of already-saved spend forecasts.

This is retrospective diagnosis, not a fresh holdout or model-selection run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.experiments import evaluate
from value_prediction.labels import SNAPSHOTS, sha256
from value_prediction.optimization import check_sources

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'outputs/ranking_audit'
DEVELOPMENT = HERE / 'outputs/optimization/development_predictions.csv'
DEVELOPMENT_MANIFEST = DEVELOPMENT.parent / 'manifest.json'
RETROSPECTIVE = HERE / 'outputs/retrospective_check/customer_predictions.csv'
RETROSPECTIVE_MANIFEST = RETROSPECTIVE.parent / 'manifest.json'
RETROSPECTIVE_MONTHLY = RETROSPECTIVE.parent / 'monthly_metrics.csv'
RETROSPECTIVE_SCRIPT = HERE / 'retrospective_check.py'
PROTOCOL = HERE / 'ranking_audit_protocol.md'
METHODS = ('historic_180d', 'extended_tweedie',
           'extended_tweedie_previous_year')
DATES = pd.date_range('2011-01-01', '2011-11-01', freq='MS')
MIN_SHARED = 500
MIN_GROUP_ROWS = 100
MIN_GROUP_BUYERS = 20


def period_for(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp('2011-05-01'):
        return 'rolling_jan_may'
    if date <= pd.Timestamp('2011-08-01'):
        return 'known_jun_aug'
    return 'viewed_sep_nov'


def recency_group(days: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(days, errors='coerce')
    if numeric.isna().any() or not numeric.between(0, 180).all():
        raise ValueError('Recency outside the predeclared 0-180 day window')
    return pd.cut(numeric, bins=[-0.001, 30, 90, 180],
                  labels=['0-30', '31-90', '91-180'])


def _read_prediction(path: Path, dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = pd.read_csv(path, dtype={'customer_id': 'string'},
                       parse_dates=['snapshot_date'])
    rows = rows.loc[rows.method.isin(METHODS)].copy()
    if (rows.empty or rows.duplicated(['snapshot_date', 'customer_id', 'method']).any()
            or set(rows.snapshot_date) != set(dates)
            or rows[['spend_30d', 'predicted_spend_30d']].isna().any().any()
            or not np.isfinite(rows[['spend_30d', 'predicted_spend_30d']].to_numpy()).all()
            or (rows[['spend_30d', 'predicted_spend_30d']] < 0).any().any()):
        raise ValueError(f'Invalid prediction file or dates: {path}')
    counts = rows.groupby(['snapshot_date', 'customer_id'], observed=True).agg(
        method_count=('method', 'nunique'), target_count=('spend_30d', 'nunique'),
        row_count=('method', 'size'))
    if (not counts.method_count.eq(len(METHODS)).all()
            or not counts.target_count.eq(1).all()
            or not counts.row_count.eq(len(METHODS)).all()):
        raise ValueError('Methods do not share a unique cohort and target')
    return rows


def verify_retrospective_metrics(rows: pd.DataFrame) -> None:
    """Reconcile a legacy prediction file whose original manifest lacks its hash."""
    reported = pd.read_csv(RETROSPECTIVE_MONTHLY,
                           parse_dates=['snapshot_date'])
    for (date, method), part in rows.groupby(['snapshot_date', 'method']):
        score = part.predicted_spend_30d.to_numpy(dtype=float)
        metric = evaluate(part, score)
        match = reported.loc[reported.snapshot_date.eq(date)
                             & reported.method.eq(method)]
        if len(match) != 1:
            raise ValueError('Missing original retrospective monthly metric')
        for key in ('mae', 'positive_mae', 'top20_spend_capture',
                    'aggregate_bias'):
            if not np.isclose(metric[key], float(match.iloc[0][key]),
                              rtol=1e-8, atol=1e-8):
                raise ValueError(f'Retrospective metric mismatch: {date}, {method}, {key}')


def load_predictions() -> pd.DataFrame:
    check_sources()
    development_manifest = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding='utf-8'))
    retrospective_manifest = json.loads(RETROSPECTIVE_MANIFEST.read_text(encoding='utf-8'))
    snapshot_hash = sha256(SNAPSHOTS)
    if (development_manifest.get('development_predictions_sha256') != sha256(DEVELOPMENT)
            or development_manifest.get('snapshot_sha256') != snapshot_hash
            or development_manifest.get('test_rows_used') is not False
            or retrospective_manifest.get('snapshot_sha256') != snapshot_hash
            or retrospective_manifest.get('script_sha256') != sha256(RETROSPECTIVE_SCRIPT)
            or any(development_manifest.get(key) != retrospective_manifest.get(key)
                   for key in ('labels_sha256', 'temporal_features_sha256',
                               'seasonal_features_sha256'))
            or not set(METHODS).issubset(
                retrospective_manifest.get('methods_fixed_before_this_run', []))):
        raise ValueError('Prediction source provenance mismatch')
    development = _read_prediction(DEVELOPMENT, DATES[:8])
    retrospective = _read_prediction(RETROSPECTIVE, DATES[8:])
    verify_retrospective_metrics(retrospective)
    rows = pd.concat([development, retrospective], ignore_index=True)
    wide = rows.pivot(index=['snapshot_date', 'customer_id', 'spend_30d'],
                      columns='method', values='predicted_spend_30d').reset_index()
    wide.columns.name = None
    if wide[list(METHODS)].isna().any().any():
        raise ValueError('Missing method score after pivot')
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'},
                            usecols=['snapshot_date', 'customer_id', 'recency_days'],
                            parse_dates=['snapshot_date'])
    snapshots['snapshot_date'] = snapshots.snapshot_date.dt.normalize()
    if snapshots.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate original snapshot key')
    wide = wide.merge(snapshots, on=['snapshot_date', 'customer_id'],
                      how='left', validate='one_to_one', indicator=True)
    if not wide._merge.eq('both').all():
        raise ValueError('Prediction missing its original snapshot')
    wide = wide.drop(columns='_merge')
    wide['recency_group'] = recency_group(wide.recency_days)
    wide['period'] = wide.snapshot_date.map(period_for)
    return wide.sort_values(['snapshot_date', 'customer_id']).reset_index(drop=True)


def _top_ids(shared: pd.DataFrame, score_column: str, quota: int) -> set[str]:
    ranked = shared.sort_values([score_column, 'customer_id'],
                                ascending=[False, True], kind='stable')
    return set(ranked.head(quota).customer_id.astype(str))


def pair_stability(previous: pd.DataFrame, current: pd.DataFrame,
                   min_shared: int = MIN_SHARED) -> pd.DataFrame:
    if len(previous.snapshot_date.unique()) != 1 or len(current.snapshot_date.unique()) != 1:
        raise ValueError('Each stability pair must contain exactly one month')
    previous_date = pd.Timestamp(previous.snapshot_date.iloc[0])
    current_date = pd.Timestamp(current.snapshot_date.iloc[0])
    if current_date != previous_date + pd.DateOffset(months=1):
        raise ValueError('Stability pair months are not adjacent')
    left = previous[['customer_id', *METHODS]]
    right = current[['customer_id', 'spend_30d', *METHODS]]
    shared = left.merge(right, on='customer_id', how='inner',
                        validate='one_to_one', suffixes=('_previous', '_current'))
    n = len(shared)
    base = {'previous_date': previous_date.strftime('%Y-%m-%d'),
            'current_date': current_date.strftime('%Y-%m-%d'),
            'period': period_for(current_date), 'shared_customers': n}
    if n < min_shared:
        return pd.DataFrame([{'method': method, **base,
                              'status': 'too_few_shared_customers'}
                             for method in METHODS])
    quota = int(np.ceil(.2 * n))
    actual = shared.spend_30d.to_numpy(dtype=float)
    total = float(actual.sum())
    if total <= 0:
        raise ValueError('No future customer spend in the common cohort')
    records = []
    for method in METHODS:
        old_ids = _top_ids(shared, f'{method}_previous', quota)
        new_ids = _top_ids(shared, f'{method}_current', quota)
        overlap = len(old_ids & new_ids)
        selected_new = shared.customer_id.isin(new_ids).to_numpy()
        selected_old = shared.customer_id.isin(old_ids).to_numpy()
        records.append({**base, 'method': method, 'status': 'ok',
                        'top20_count': quota, 'overlap_count': overlap,
                        'retention': overlap / quota,
                        'jaccard': overlap / (2 * quota - overlap),
                        'current_list_capture': float(actual[selected_new].sum()) / total,
                        'previous_rank_capture': float(actual[selected_old].sum()) / total})
    return pd.DataFrame(records)


def stability_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for date in DATES[1:]:
        if date == pd.Timestamp('2011-09-01'):
            continue  # Different saved prediction-file versions across this boundary.
        prior = date - pd.DateOffset(months=1)
        previous = wide.loc[wide.snapshot_date.eq(prior)]
        current = wide.loc[wide.snapshot_date.eq(date)]
        if previous.empty or current.empty:
            raise ValueError(f'Missing adjacent-month cohort: {date}')
        pairs.append(pair_stability(previous, current))
    return pd.concat(pairs, ignore_index=True)


def segment_metrics(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    period_rows, month_rows = [], []
    for (period, group), cohort in wide.groupby(['period', 'recency_group'],
                                                observed=True):
        for method in METHODS:
            result = evaluate(cohort, cohort[method].to_numpy(dtype=float))
            period_rows.append({'period': period, 'recency_group': str(group),
                                'method': method,
                                'interpretation_eligible':
                                len(cohort) >= MIN_GROUP_ROWS
                                and int(cohort.spend_30d.gt(0).sum()) >= MIN_GROUP_BUYERS,
                                **result})
    for (date, group), cohort in wide.groupby(['snapshot_date', 'recency_group'],
                                               observed=True):
        for method in METHODS:
            result = evaluate(cohort, cohort[method].to_numpy(dtype=float))
            month_rows.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                               'period': period_for(date),
                               'recency_group': str(group), 'method': method,
                               'interpretation_eligible':
                               len(cohort) >= MIN_GROUP_ROWS
                               and int(cohort.spend_30d.gt(0).sum()) >= MIN_GROUP_BUYERS,
                               **result})
    return pd.DataFrame(period_rows), pd.DataFrame(month_rows)


def make_report(wide: pd.DataFrame, stability: pd.DataFrame,
                pooled: pd.DataFrame) -> str:
    lines = [
        '# 高价值客户名单稳定性与历史活跃度误差审计', '',
        '规则事先记录于 `ranking_audit_protocol.md`。这是已查看月份的诊断，非新的独立验收；不重新训练或选择模型。',
        f'共 {len(wide):,} 条独立客户月快照，期间 2011-01 至 2011-11。', '',
        '## 相邻月共同客户的 Top20 名单', '',
        '| 新月份期间 | 方法 | 月对 | 共同客户数范围 | 名单留存均值 | Jaccard 均值 | 新月名单消费覆盖均值 | 上月排名名单在新月的消费覆盖均值 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for (period, method), rows in stability.groupby(['period', 'method'], sort=False):
        ok = rows.loc[rows.status.eq('ok')]
        if ok.empty:
            lines.append(f'| {period} | {method} | 0 | 样本不足 | — | — | — | — |')
            continue
        lines.append(
            f'| {period} | {method} | {len(ok)} | '
            f'{ok.shared_customers.min():,}–{ok.shared_customers.max():,} | '
            f'{ok.retention.mean():.2%} | {ok.jaccard.mean():.2%} | '
            f'{ok.current_list_capture.mean():.2%} | '
            f'{ok.previous_rank_capture.mean():.2%} |')
    lines += ['', '月对逐项数据见 `list_stability.csv`。共同客户名单是控制客户流入流出的诊断口径；'
              '上月排名也在事后限定的共同客户内重排，不是线上旧名单回放。', '',
              '## 快照时已知的最近购买间隔', '',
              '| 期间 | 最近购买间隔 | 客户月 | 未来购买快照 | 20 特征模型 MAE 相对基线 | 去年同期模型 MAE 相对基线 | 20 特征模型购买者 MAE 相对基线 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for (period, group), rows in pooled.groupby(['period', 'recency_group'],
                                                sort=False):
        by_method = rows.set_index('method')
        base = by_method.loc['historic_180d']
        direct = by_method.loc['extended_tweedie']
        seasonal = by_method.loc['extended_tweedie_previous_year']
        if not bool(base.interpretation_eligible):
            values = ('样本不足', '样本不足', '样本不足')
        else:
            values = (f'{(1 - direct.mae / base.mae):+.2%}',
                      f'{(1 - seasonal.mae / base.mae):+.2%}',
                      f'{(1 - direct.positive_mae / base.positive_mae):+.2%}')
        lines.append(f'| {period} | {group} 天 | {int(base.snapshot_count):,} | '
                     f'{int(base.buyer_count):,} | {values[0]} | {values[1]} | '
                     f'{values[2]} |')
    lines += ['', '正值表示误差降低，负值表示误差增加。完整的 MAE、购买者 MAE、组内按月 Top20 覆盖和预测总额偏差见 '
              '`recency_period_metrics.csv` 与 `recency_monthly_metrics.csv`。', '',
              '稳定性、分组优势和名单消费覆盖均不证明个人金额准确，也不替代原商家未来新月份的多指标验收。', '']
    return '\n'.join(lines)


def main() -> None:
    wide = load_predictions()
    stability = stability_metrics(wide)
    pooled, monthly = segment_metrics(wide)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {'list_stability': OUTPUT / 'list_stability.csv',
             'recency_period_metrics': OUTPUT / 'recency_period_metrics.csv',
             'recency_monthly_metrics': OUTPUT / 'recency_monthly_metrics.csv'}
    for name, frame in [('list_stability', stability),
                        ('recency_period_metrics', pooled),
                        ('recency_monthly_metrics', monthly)]:
        frame.to_csv(paths[name], index=False, encoding='utf-8-sig')
    report = OUTPUT / 'report.md'
    report.write_text(make_report(wide, stability, pooled), encoding='utf-8')
    manifest = {
        'status': 'retrospective_diagnostic_no_independent_acceptance',
        'methods': METHODS, 'snapshot_count': len(wide),
        'customer_count': int(wide.customer_id.nunique()),
        'development_predictions_sha256': sha256(DEVELOPMENT),
        'retrospective_predictions_sha256': sha256(RETROSPECTIVE),
        'retrospective_manifest_prior_development_sha256': json.loads(
            RETROSPECTIVE_MANIFEST.read_text(encoding='utf-8'))[
                'prior_development_predictions_sha256'],
        'excluded_cross_version_pair': '2011-08-01_to_2011-09-01',
        'retrospective_monthly_metrics_sha256': sha256(RETROSPECTIVE_MONTHLY),
        'retrospective_metrics_reconciled': True,
        'snapshot_source_sha256': sha256(SNAPSHOTS),
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'output_sha256': {name: sha256(path) for name, path in paths.items()},
        'report_sha256': sha256(report),
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                    indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {report}')


if __name__ == '__main__':
    main()
