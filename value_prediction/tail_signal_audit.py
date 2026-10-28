"""Diagnose pre-cutoff evidence for large future spend in viewed cohorts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.composition_features import (
    FEATURE_PATH as COMPOSITION_PATH, MANIFEST_PATH as COMPOSITION_MANIFEST,
)
from value_prediction.experiments import top20_mask
from value_prediction.labels import SNAPSHOTS, sha256
from value_prediction.ranking_audit import load_predictions, period_for
from value_prediction.seasonal_features import FEATURE_PATH as SEASONAL_PATH
from value_prediction.temporal_features import FEATURE_PATH as TEMPORAL_PATH

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'outputs/tail_signal_audit'
PROTOCOL = HERE / 'tail_signal_protocol.md'
HURDLE = HERE / 'outputs/hurdle_diagnostics/customer_predictions.csv'
HURDLE_MANIFEST = HURDLE.parent / 'manifest.json'
METHOD = 'extended_tweedie'
CONTINUOUS = (
    'monetary_value_180d', 'max_invoice_value_180d',
    'monetary_value_90d', 'invoice_frequency_180d', 'recency_days',
    'bulk_spend_share_180d', 'median_unit_price_180d',
)
BINARY = ('non_uk_180d', 'prior_year_available')


def mark_actual_top_buyers(frame: pd.DataFrame) -> pd.Series:
    """Use outcomes only to define a retrospective group, never a predictor."""
    if frame.snapshot_date.nunique() != 1:
        raise ValueError('Actual tail selection requires one calendar month')
    if frame.duplicated('customer_id').any():
        raise ValueError('Duplicate customer in actual tail cohort')
    buyers = frame.loc[frame.spend_30d.gt(0)].sort_values(
        ['spend_30d', 'customer_id'], ascending=[False, True], kind='stable')
    if buyers.empty:
        raise ValueError('No future buyers in month')
    quota = int(np.ceil(.01 * len(buyers)))
    chosen = set(buyers.head(quota).customer_id.astype(str))
    return frame.customer_id.astype(str).isin(chosen)


def _merge_feature(wide: pd.DataFrame, path: Path, columns: list[str],
                   allow_missing: tuple[str, ...] = ()) -> pd.DataFrame:
    source = pd.read_csv(path, dtype={'customer_id': 'string'},
                         usecols=['snapshot_date', 'customer_id', *columns],
                         parse_dates=['snapshot_date'])
    if source.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError(f'Duplicate feature key: {path}')
    merged = wide.merge(source, on=['snapshot_date', 'customer_id'], how='left',
                        validate='one_to_one', indicator=True)
    required = [name for name in columns if name not in allow_missing]
    if (len(merged) != len(wide) or not merged._merge.eq('both').all()
            or merged[required].isna().any().any()):
        raise ValueError(f'Incomplete feature join: {path}')
    return merged.drop(columns='_merge')


def load_audit_data() -> pd.DataFrame:
    wide = load_predictions()  # Checks both saved forecast cohorts and source manifests.
    manifest = json.loads(COMPOSITION_MANIFEST.read_text(encoding='utf-8'))
    if (manifest.get('features_sha256') != sha256(COMPOSITION_PATH)
            or manifest.get('source_snapshot_sha256') != sha256(SNAPSHOTS)):
        raise ValueError('Composition feature provenance mismatch')
    wide = _merge_feature(wide, SNAPSHOTS,
                          ['monetary_value_180d', 'invoice_frequency_180d'])
    wide = _merge_feature(wide, TEMPORAL_PATH,
                          ['max_invoice_value_180d', 'monetary_value_90d'])
    wide = _merge_feature(wide, COMPOSITION_PATH,
                          ['bulk_spend_share_180d', 'median_unit_price_180d',
                           'non_uk_180d'])
    wide = _merge_feature(wide, SEASONAL_PATH, ['previous_year_30d_spend'],
                          allow_missing=('previous_year_30d_spend',))
    wide['prior_year_available'] = wide.previous_year_30d_spend.notna().astype('int8')
    if wide[list(CONTINUOUS)].lt(0).any().any() or not wide.non_uk_180d.isin([0, 1]).all():
        raise ValueError('Invalid pre-cutoff signal values')
    for name in ('monetary_value_180d', 'max_invoice_value_180d'):
        wide[f'{name}_percentile'] = wide.groupby('snapshot_date')[name].rank(
            pct=True, method='average')
    wide['historic_monthly_amount'] = wide.monetary_value_180d / 6
    if wide.historic_monthly_amount.le(0).any():
        raise ValueError('Eligible customers need positive pre-cutoff spend')
    wide['future_over_historic_monthly'] = (
        wide.spend_30d / wide.historic_monthly_amount)
    return wide


def summarize_tail(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame]:
    rows = []
    monthly = []
    for date, frame in wide.groupby('snapshot_date', sort=True):
        frame = frame.copy()
        frame['actual_tail'] = mark_actual_top_buyers(frame)
        frame['predicted_top20'] = top20_mask(
            frame, frame[METHOD].to_numpy(dtype=float))
        buyers = frame.loc[frame.spend_30d.gt(0)]
        tail = buyers.loc[buyers.actual_tail]
        others = buyers.loc[~buyers.actual_tail]
        if tail.empty or others.empty:
            raise ValueError(f'Insufficient buyers for tail comparison: {date}')
        period = period_for(date)
        total = float(frame.spend_30d.sum())
        monthly.append({
            'snapshot_date': date.strftime('%Y-%m-%d'), 'period': period,
            'buyer_count': len(buyers), 'actual_tail_count': len(tail),
            'actual_spend': total, 'tail_actual_spend': float(tail.spend_30d.sum()),
            'tail_predicted_spend': float(tail[METHOD].sum()),
            'tail_actual_spend_share': float(tail.spend_30d.sum() / total),
            'tail_predicted_to_actual': float(tail[METHOD].sum() / tail.spend_30d.sum()),
            'tail_in_predicted_top20': float(tail.predicted_top20.mean()),
            'tail_historic_180d_percentile_median':
            float(tail.monetary_value_180d_percentile.median()),
            'tail_max_invoice_percentile_median':
            float(tail.max_invoice_value_180d_percentile.median()),
            'tail_future_over_historic_median':
            float(tail.future_over_historic_monthly.median()),
            'tail_future_over_historic_gt2':
            float(tail.future_over_historic_monthly.gt(2).mean()),
            'other_future_over_historic_median':
            float(others.future_over_historic_monthly.median()),
            'other_future_over_historic_gt2':
            float(others.future_over_historic_monthly.gt(2).mean()),
        })
        for group_name, part in [('actual_top1pct_buyers', tail),
                                 ('other_buyers', others)]:
            rows.append(part.assign(audit_group=group_name))
    labeled = pd.concat(rows, ignore_index=True)
    period_rows, signal_rows = [], []
    for period, period_frame in labeled.groupby('period', sort=False):
        tail = period_frame.loc[period_frame.audit_group.eq('actual_top1pct_buyers')]
        others = period_frame.loc[period_frame.audit_group.eq('other_buyers')]
        all_period = wide.loc[wide.period.eq(period)]
        period_rows.append({
            'period': period, 'buyer_count': len(period_frame),
            'actual_tail_count': len(tail),
            'tail_actual_spend_share':
            float(tail.spend_30d.sum() / all_period.spend_30d.sum()),
            'tail_predicted_to_actual':
            float(tail[METHOD].sum() / tail.spend_30d.sum()),
            'tail_in_predicted_top20': float(tail.predicted_top20.mean()),
            'tail_historic_180d_percentile_median':
            float(tail.monetary_value_180d_percentile.median()),
            'tail_max_invoice_percentile_median':
            float(tail.max_invoice_value_180d_percentile.median()),
            'tail_future_over_historic_median':
            float(tail.future_over_historic_monthly.median()),
            'tail_future_over_historic_gt2':
            float(tail.future_over_historic_monthly.gt(2).mean()),
            'other_future_over_historic_median':
            float(others.future_over_historic_monthly.median()),
            'other_future_over_historic_gt2':
            float(others.future_over_historic_monthly.gt(2).mean()),
        })
        for group_name, part in [('actual_top1pct_buyers', tail),
                                 ('other_buyers', others)]:
            for feature in CONTINUOUS:
                signal_rows.append({'period': period, 'group': group_name,
                                    'feature': feature, 'statistic': 'median',
                                    'value': float(part[feature].median()),
                                    'nonmissing_count': int(part[feature].notna().sum())})
            for feature in BINARY:
                signal_rows.append({'period': period, 'group': group_name,
                                    'feature': feature, 'statistic': 'mean',
                                    'value': float(part[feature].mean()),
                                    'nonmissing_count': int(part[feature].notna().sum())})
            signal_rows.append({
                'period': period, 'group': group_name,
                'feature': 'previous_year_30d_spend',
                'statistic': 'median_among_available',
                'value': float(part.previous_year_30d_spend.median()),
                'nonmissing_count': int(part.previous_year_30d_spend.notna().sum())})
    return pd.DataFrame(monthly), pd.DataFrame(period_rows), pd.DataFrame(signal_rows)


def hurdle_tail_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    manifest = json.loads(HURDLE_MANIFEST.read_text(encoding='utf-8'))
    if (manifest.get('predictions_sha256') != sha256(HURDLE)
            or manifest.get('snapshot_sha256') != sha256(SNAPSHOTS)
            or manifest.get('test_rows_used') is not False):
        raise ValueError('Two-stage forecast source provenance mismatch')
    saved = pd.read_csv(HURDLE, dtype={'customer_id': 'string'},
                        parse_dates=['snapshot_date'],
                        usecols=['snapshot_date', 'customer_id', 'spend_30d',
                                 'purchase_probability', 'mean_if_purchase',
                                 'expected_spend_30d'])
    if saved.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate two-stage customer month')
    development = wide.loc[wide.snapshot_date.le(pd.Timestamp('2011-08-01'))].copy()
    development['actual_tail'] = False
    for date, frame in development.groupby('snapshot_date', sort=True):
        development.loc[frame.index, 'actual_tail'] = mark_actual_top_buyers(frame).to_numpy()
    tail = development.loc[development.actual_tail].copy()
    joined = tail.merge(saved, on=['snapshot_date', 'customer_id', 'spend_30d'],
                        how='left', validate='one_to_one', indicator=True)
    if (len(joined) != len(tail) or not joined._merge.eq('both').all()
            or joined[['purchase_probability', 'mean_if_purchase',
                       'expected_spend_30d']].isna().any().any()):
        raise ValueError('Two-stage tail group does not reconcile')
    rows = []
    for period, frame in joined.groupby('period', sort=False):
        actual = float(frame.spend_30d.sum())
        rows.append({
            'period': period, 'actual_tail_count': len(frame),
            'purchase_probability_mean': float(frame.purchase_probability.mean()),
            'conditional_mean_to_actual': float(frame.mean_if_purchase.sum() / actual),
            'two_stage_expected_to_actual':
            float(frame.expected_spend_30d.sum() / actual),
            'direct_tweedie_to_actual': float(frame[METHOD].sum() / actual),
        })
    return pd.DataFrame(rows)


def predicted_tail_metrics(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = []
    for date, frame in wide.groupby('snapshot_date', sort=True):
        frame = frame.copy()
        quota = int(np.ceil(.01 * len(frame)))
        ranked = frame.sort_values([METHOD, 'customer_id'],
                                   ascending=[False, True], kind='stable')
        selected = ranked.head(quota)
        actual_tail = mark_actual_top_buyers(frame)
        actual_tail_ids = set(frame.loc[actual_tail, 'customer_id'].astype(str))
        overlap = len(actual_tail_ids & set(selected.customer_id.astype(str)))
        actual = float(selected.spend_30d.sum())
        monthly.append({
            'snapshot_date': date.strftime('%Y-%m-%d'),
            'period': period_for(date), 'snapshot_count': len(frame),
            'predicted_top1pct_count': len(selected),
            'actual_top1pct_buyer_count': len(actual_tail_ids),
            'actual_tail_overlap_count': overlap,
            'actual_tail_recall': overlap / len(actual_tail_ids),
            'total_actual_spend': float(frame.spend_30d.sum()),
            'predicted_group_actual_spend': actual,
            'predicted_group_predicted_spend': float(selected[METHOD].sum()),
            'predicted_to_actual':
            float(selected[METHOD].sum() / actual) if actual > 0 else np.nan,
            'actual_spend_share':
            float(actual / frame.spend_30d.sum()),
        })
    monthly = pd.DataFrame(monthly)
    period_rows = []
    for period, part in monthly.groupby('period', sort=False):
        actual = float(part.predicted_group_actual_spend.sum())
        period_rows.append({
            'period': period,
            'months': len(part),
            'predicted_top1pct_count': int(part.predicted_top1pct_count.sum()),
            'actual_top1pct_buyer_count': int(part.actual_top1pct_buyer_count.sum()),
            'actual_tail_overlap_count': int(part.actual_tail_overlap_count.sum()),
            'actual_tail_recall': float(part.actual_tail_overlap_count.sum()
                                        / part.actual_top1pct_buyer_count.sum()),
            'predicted_group_actual_spend': actual,
            'predicted_group_predicted_spend':
            float(part.predicted_group_predicted_spend.sum()),
            'predicted_to_actual':
            float(part.predicted_group_predicted_spend.sum() / actual)
            if actual > 0 else np.nan,
            'actual_spend_share': float(part.predicted_group_actual_spend.sum()
                                        / part.total_actual_spend.sum()),
        })
    return monthly, pd.DataFrame(period_rows)


def make_report(monthly: pd.DataFrame, periods: pd.DataFrame,
                signals: pd.DataFrame, hurdle: pd.DataFrame,
                predicted_tail: pd.DataFrame) -> str:
    lines = [
        '# 高额未来消费的截点前信号：事后审计', '',
        '事先口径见 `tail_signal_protocol.md`。高额组由真实未来消费定义，'
        '只能用于诊断；所有排名分数和特征本身均为截点前可得。'
        '所有月份此前已查看，不是独立验收。', '',
        '| 期间 | 购买快照 | 实际前 1% 人数 | 该组实际消费份额 | 模型预测/该组实际 | 该组进入预测 Top20 | 历史 180 天金额中位百分位 | 未来/旧月均中位倍数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in periods.itertuples(index=False):
        lines.append(f'| {row.period} | {row.buyer_count:,} | '
                     f'{row.actual_tail_count} | {row.tail_actual_spend_share:.2%} | '
                     f'{row.tail_predicted_to_actual:.2%} | '
                     f'{row.tail_in_predicted_top20:.2%} | '
                     f'{row.tail_historic_180d_percentile_median:.1%} | '
                     f'{row.tail_future_over_historic_median:.2f}× |')
    lines += ['', '历史金额百分位在同月全部 eligible 客户中计算。'
              '“未来/旧月均”中的未来结果只用于事后波动审计，不能作为预测时的输入。',
              '', '## 截点前特征：高额组与其他实际购买者', '',
              '| 期间 | 特征 | 高额组 | 其他购买者 |',
              '| --- | --- | ---: | ---: |']
    display = ['monetary_value_180d', 'max_invoice_value_180d',
               'monetary_value_90d', 'invoice_frequency_180d',
               'recency_days', 'bulk_spend_share_180d',
               'median_unit_price_180d', 'non_uk_180d',
               'prior_year_available']
    for period in periods.period:
        for feature in display:
            t = signals.loc[signals.period.eq(period)
                            & signals.feature.eq(feature)].set_index('group')
            high = t.loc['actual_top1pct_buyers', 'value']
            other = t.loc['other_buyers', 'value']
            if feature in BINARY or feature == 'bulk_spend_share_180d':
                shown = (f'{high:.1%}', f'{other:.1%}')
            else:
                shown = (f'{high:.2f}', f'{other:.2f}')
            lines.append(f'| {period} | {feature} | {shown[0]} | {shown[1]} |')
    lines += ['', '## 两阶段金额分解（首轮结果后的解释性补充）', '',
              '仅 1–8 月有已保存的两阶段样本外预测；以下均按事后实际高额组计算。', '',
              '| 期间 | 实际高额组人数 | 平均购买概率 | 假定会买时的金额预测/实际 | 概率×金额预测/实际 | 直接模型预测/实际 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in hurdle.itertuples(index=False):
        lines.append(f'| {row.period} | {row.actual_tail_count} | '
                     f'{row.purchase_probability_mean:.2%} | '
                     f'{row.conditional_mean_to_actual:.2%} | '
                     f'{row.two_stage_expected_to_actual:.2%} | '
                     f'{row.direct_tweedie_to_actual:.2%} |')
    lines += ['', '“假定会买时”的金额预测仍来自截点前模型，'
              '这里仅用真实结果选事后审计组。条件金额也显著偏低时，'
              '低估不能只归因于购买概率折减；但按实际极值选组本身也会造成向下的选择偏差。',
              '', '## 按预测分数事先选出的前 1% 客户', '',
              '| 期间 | 名单客户月 | 命中实际极高消费组 | 名单预测金额/实际金额 | 名单实际消费份额 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in predicted_tail.itertuples(index=False):
        lines.append(f'| {row.period} | {row.predicted_top1pct_count} | '
                     f'{row.actual_tail_recall:.2%} | '
                     f'{row.predicted_to_actual:.2%} | '
                     f'{row.actual_spend_share:.2%} |')
    lines += ['', '这个名单只依据截点前的预测分数选择，'
              '因此它的预测/实际金额比更适合检查高分客户的校准；逐月数据见 `predicted_tail_monthly.csv`。', '',
              '逐月高额组人数、波动和历史最大订单百分位见 `monthly_tail_metrics.csv`。'
              '所有特征的分组汇总见 `feature_signals.csv`。'
              '这些差异不能证明某特征能因果解释未来高额购买，也不能据此在已看过月份调参。', '']
    return '\n'.join(lines)


def main() -> None:
    wide = load_audit_data()
    monthly, periods, signals = summarize_tail(wide)
    hurdle = hurdle_tail_metrics(wide)
    predicted_monthly, predicted_period = predicted_tail_metrics(wide)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {'monthly_tail_metrics': OUTPUT / 'monthly_tail_metrics.csv',
             'period_metrics': OUTPUT / 'period_metrics.csv',
             'feature_signals': OUTPUT / 'feature_signals.csv',
             'hurdle_tail_metrics': OUTPUT / 'hurdle_tail_metrics.csv',
             'predicted_tail_monthly': OUTPUT / 'predicted_tail_monthly.csv',
             'predicted_tail_period': OUTPUT / 'predicted_tail_period.csv'}
    for name, frame in [('monthly_tail_metrics', monthly),
                        ('period_metrics', periods),
                        ('feature_signals', signals),
                        ('hurdle_tail_metrics', hurdle),
                        ('predicted_tail_monthly', predicted_monthly),
                        ('predicted_tail_period', predicted_period)]:
        frame.to_csv(paths[name], index=False, encoding='utf-8-sig')
    report = OUTPUT / 'report.md'
    report.write_text(make_report(monthly, periods, signals, hurdle,
                                  predicted_period), encoding='utf-8')
    manifest = {
        'status': 'viewed_period_tail_diagnostic_only',
        'definition': 'top ceil(1% of actual buyers) in each month',
        'snapshot_rows': len(wide), 'months': len(monthly),
        'prediction_sources_sha256': {
            'development': sha256(HERE / 'outputs/optimization/development_predictions.csv'),
            'retrospective': sha256(HERE / 'outputs/retrospective_check/customer_predictions.csv'),
        },
        'snapshot_source_sha256': sha256(SNAPSHOTS),
        'temporal_features_sha256': sha256(TEMPORAL_PATH),
        'seasonal_features_sha256': sha256(SEASONAL_PATH),
        'composition_features_sha256': sha256(COMPOSITION_PATH),
        'hurdle_predictions_sha256': sha256(HURDLE),
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'outputs_sha256': {name: sha256(path) for name, path in paths.items()},
        'report_sha256': sha256(report),
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                    indent=2) + '\n', encoding='utf-8')
    print(f'wrote {report}', flush=True)


if __name__ == '__main__':
    main()
