"""Build an auditable historical high-value customer list from fixed scores."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.labels import SNAPSHOTS, sha256
from value_prediction.ranking_audit import (
    DEVELOPMENT, DEVELOPMENT_MANIFEST, RETROSPECTIVE,
    RETROSPECTIVE_MANIFEST, load_predictions,
)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'outputs/high_value'
PROTOCOL = HERE / 'high_value_protocol.md'
METHOD = 'extended_tweedie'
BASELINE = 'historic_180d'
DATES = pd.date_range('2011-06-01', '2011-11-01', freq='MS')
EXPORT_COLUMNS = [
    'snapshot_date', 'customer_id', 'priority_rank', 'priority_percentile',
    'priority_band', 'baseline_rank', 'recency_days', 'monetary_value_180d',
]


def rank_month(cohort: pd.DataFrame) -> pd.DataFrame:
    """Rank one entire snapshot with deterministic customer-ID tie breaks."""
    required = {'snapshot_date', 'customer_id', 'spend_30d', METHOD, BASELINE,
                'recency_days', 'monetary_value_180d'}
    if not required.issubset(cohort.columns):
        raise ValueError(f'Missing ranking fields: {sorted(required - set(cohort))}')
    if cohort.empty or cohort.snapshot_date.nunique() != 1:
        raise ValueError('Ranking requires exactly one nonempty month')
    if cohort.customer_id.isna().any() or cohort.customer_id.duplicated().any():
        raise ValueError('Customer IDs must be present and unique per month')
    numeric = cohort[[METHOD, BASELINE, 'spend_30d',
                      'recency_days', 'monetary_value_180d']]
    if (numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all()
            or numeric.lt(0).any().any()):
        raise ValueError('Ranking inputs must be finite and nonnegative')
    quota = int(np.ceil(len(cohort) * .20))
    ranked = cohort.sort_values([METHOD, 'customer_id'],
                                ascending=[False, True], kind='stable').copy()
    ranked['priority_rank'] = np.arange(1, len(ranked) + 1)
    ranked['priority_percentile'] = (
        100 * (len(ranked) - ranked.priority_rank + 1) / len(ranked))
    ranked['priority_band'] = np.where(
        ranked.priority_rank.le(quota), 'Top 20%', 'Other')
    baseline = cohort.sort_values([BASELINE, 'customer_id'],
                                  ascending=[False, True], kind='stable')
    baseline_ranks = pd.Series(np.arange(1, len(baseline) + 1),
                               index=baseline.customer_id.to_numpy())
    ranked['baseline_rank'] = ranked.customer_id.map(baseline_ranks).astype(int)
    return ranked


def monthly_metric(ranked: pd.DataFrame) -> dict:
    quota = int(np.ceil(len(ranked) * .20))
    if (ranked.priority_rank.tolist() != list(range(1, len(ranked) + 1))
            or ranked.baseline_rank.nunique() != len(ranked)
            or int(ranked.priority_band.eq('Top 20%').sum()) != quota):
        raise ValueError('Invalid monthly ranking or quota')
    actual = float(ranked.spend_30d.sum())
    if actual <= 0:
        raise ValueError('Future spend total must be positive for evaluation')
    model_spend = float(ranked.loc[ranked.priority_rank.le(quota),
                                    'spend_30d'].sum())
    baseline_spend = float(ranked.loc[ranked.baseline_rank.le(quota),
                                       'spend_30d'].sum())
    return {
        'snapshot_date': pd.Timestamp(ranked.snapshot_date.iloc[0]).strftime('%Y-%m-%d'),
        'customers': len(ranked), 'top20_count': quota,
        'actual_spend': actual, 'model_list_actual_spend': model_spend,
        'baseline_list_actual_spend': baseline_spend,
        'model_capture': model_spend / actual,
        'baseline_capture': baseline_spend / actual,
        'gain_pp': 100 * (model_spend - baseline_spend) / actual,
    }


def period_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    if set(monthly.snapshot_date) != set(DATES.strftime('%Y-%m-%d')):
        raise ValueError('Missing or unexpected evaluation month')
    rows = []
    for name, months in [('known_jun_aug', DATES[:3]),
                         ('viewed_sep_nov', DATES[3:])]:
        part = monthly.loc[monthly.snapshot_date.isin(months.strftime('%Y-%m-%d'))]
        if len(part) != 3:
            raise ValueError(f'Incomplete evaluation period: {name}')
        actual = float(part.actual_spend.sum())
        model = float(part.model_list_actual_spend.sum())
        base = float(part.baseline_list_actual_spend.sum())
        rows.append({
            'period': name, 'months': len(part), 'customers': int(part.customers.sum()),
            'actual_spend': actual, 'model_capture': model / actual,
            'baseline_capture': base / actual,
            'gain_pp': 100 * (model - base) / actual,
            'nonworse_months': int(part.gain_pp.ge(-1e-10).sum()),
        })
    return pd.DataFrame(rows)


def demonstration_decision(monthly: pd.DataFrame,
                           periods: pd.DataFrame) -> dict:
    conditions = {
        'both_periods_gain_at_least_1pp': bool(len(periods) == 2
            and periods.gain_pp.ge(1 - 1e-10).all()),
        'all_six_months_nonworse': bool(len(monthly) == 6
            and monthly.gain_pp.ge(-1e-10).all()),
        'complete_months_and_quota': bool(len(monthly) == 6
            and set(monthly.snapshot_date) == set(DATES.strftime('%Y-%m-%d'))
            and monthly.top20_count.eq(np.ceil(monthly.customers * .2)).all()),
    }
    return {'historical_demo_allowed': all(conditions.values()),
            'conditions': conditions,
            'evidence_scope': 'viewed_historical_replay_not_independent_test'}


def validate_training_maturity(development: dict, retrospective: dict) -> None:
    """Reject a saved fold that could have trained on an unfinished label."""
    records = [*development.get('folds', []),
               *retrospective.get('fit_cutoffs', [])]
    relevant = [record for record in records if pd.Timestamp(
        record.get('target_date', record.get('snapshot_date')))
                in DATES]
    if len(relevant) != len(DATES):
        raise ValueError('Missing saved training cutoff for a ranking month')
    dates = set()
    for record in relevant:
        target = pd.Timestamp(record.get('target_date', record.get('snapshot_date')))
        if target in dates or int(record['fit_count']) <= 0:
            raise ValueError('Duplicate or empty saved training fold')
        dates.add(target)
        if pd.Timestamp(record['latest_fit_label_end']) > target:
            raise ValueError('Saved fold contains immature future labels')
        if ('latest_fit_cutoff' in record
                and pd.Timestamp(record['latest_fit_cutoff']) >= target):
            raise ValueError('Saved fold includes target-month snapshots')


def load_cohorts() -> pd.DataFrame:
    validate_training_maturity(
        json.loads(DEVELOPMENT_MANIFEST.read_text(encoding='utf-8')),
        json.loads(RETROSPECTIVE_MANIFEST.read_text(encoding='utf-8')))
    wide = load_predictions()
    wide = wide.loc[wide.snapshot_date.isin(DATES)].copy()
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'},
                            usecols=['snapshot_date', 'customer_id',
                                     'monetary_value_180d'],
                            parse_dates=['snapshot_date'])
    snapshots['snapshot_date'] = snapshots.snapshot_date.dt.normalize()
    if snapshots.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate source snapshot customer key')
    wide = wide.merge(snapshots, on=['snapshot_date', 'customer_id'],
                      how='left', validate='one_to_one', indicator=True)
    if not wide._merge.eq('both').all():
        raise ValueError('Ranking cohort does not match original snapshots')
    return wide.drop(columns='_merge')


def report_text(monthly: pd.DataFrame, periods: pd.DataFrame,
                decision: dict) -> str:
    lines = [
        '# 未来 30 天高消费潜力名单：历史回放', '',
        '固定使用既有 20 特征模型的排序，与过去 180 天消费额排序比较。'
        '这是已查看月份的课程演示，**不是新独立测试，也不是个人消费额预测验收**。', '',
        '## 同一批客户的 Top 20% 名单', '',
        '| 月份 | 客户数 | 名额 | 模型覆盖 | 历史金额基线覆盖 | 提升 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in monthly.itertuples():
        lines.append(f'| {row.snapshot_date[:7]} | {row.customers:,} | '
                     f'{row.top20_count:,} | {row.model_capture:.2%} | '
                     f'{row.baseline_capture:.2%} | {row.gain_pp:+.2f} 个百分点 |')
    lines += ['', '| 期间 | 客户月 | 模型覆盖 | 基线覆盖 | 提升 | 不退步月份 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in periods.itertuples():
        lines.append(f'| {row.period} | {row.customers:,} | '
                     f'{row.model_capture:.2%} | {row.baseline_capture:.2%} | '
                     f'{row.gain_pp:+.2f} 个百分点 | {row.nonworse_months}/3 |')
    lines += ['', f'**历史演示门槛：{"达到" if decision["historical_demo_allowed"] else "未达到"}。** '
              '该门槛是在既有结果已被查看后制定，仅决定课程演示是否展示名单，'
              '不表示统计显著、未来泛化或营销增量。', '',
              '名单 CSV 只含当时可用的客户 ID、历史特征与排名；'
              '未来实际消费额只保留在汇总评价中。金额数值预测仍未通过原强化验收。',
              '比较月份并非线上触达试验，没有活动成本、对照组或真实 ROI。', '']
    return '\n'.join(lines)


def main() -> None:
    cohorts = load_cohorts()
    rankings = [rank_month(part) for _, part in cohorts.groupby('snapshot_date', sort=True)]
    monthly = pd.DataFrame([monthly_metric(r) for r in rankings])
    periods = period_metrics(monthly)
    decision = demonstration_decision(monthly, periods)
    export = pd.concat(rankings, ignore_index=True)[EXPORT_COLUMNS].copy()
    if ('spend_30d' in export.columns or METHOD in export.columns
            or BASELINE in export.columns or export.duplicated(
                ['snapshot_date', 'customer_id']).any()
            or len(export) != int(monthly.customers.sum())):
        raise ValueError('Unsafe or inconsistent customer export')
    export['snapshot_date'] = pd.to_datetime(export.snapshot_date).dt.strftime('%Y-%m-%d')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {
        'customer_priority': OUTPUT / 'customer_priority.csv',
        'monthly_metrics': OUTPUT / 'monthly_metrics.csv',
        'period_metrics': OUTPUT / 'period_metrics.csv',
    }
    for name, frame in [('customer_priority', export),
                        ('monthly_metrics', monthly), ('period_metrics', periods)]:
        frame.to_csv(paths[name], index=False, encoding='utf-8-sig')
    decision_path = OUTPUT / 'decision.json'
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
    report = OUTPUT / 'report.md'
    report.write_text(report_text(monthly, periods, decision), encoding='utf-8')
    manifest = {
        'status': decision['evidence_scope'], 'model_method': METHOD,
        'baseline_method': BASELINE, 'historical_demo_allowed':
        decision['historical_demo_allowed'],
        'development_predictions_sha256': sha256(DEVELOPMENT),
        'retrospective_predictions_sha256': sha256(RETROSPECTIVE),
        'snapshot_source_sha256': sha256(SNAPSHOTS),
        'protocol_sha256': sha256(PROTOCOL), 'script_sha256': sha256(Path(__file__)),
        'outputs_sha256': {name: sha256(path) for name, path in paths.items()},
        'decision_sha256': sha256(decision_path), 'report_sha256': sha256(report),
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                   indent=2) + '\n', encoding='utf-8')
    print(f'Historical demo allowed: {decision["historical_demo_allowed"]}; '
          f'wrote {report}')


if __name__ == '__main__':
    main()
