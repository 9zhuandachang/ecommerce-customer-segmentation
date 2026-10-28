"""Customer-cluster bootstrap for previously viewed development confirmation.

This quantifies sampling uncertainty conditional on June-August 2011. It does
not estimate variation across future calendar months or create a new holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.experiments import TARGET, evaluate
from value_prediction.labels import sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/uncertainty'
HURDLE = ROOT / 'value_prediction/outputs/hurdle_diagnostics/customer_predictions.csv'
HURDLE_MANIFEST = HURDLE.parent / 'manifest.json'
OPTIMIZATION = ROOT / 'value_prediction/outputs/optimization/development_predictions.csv'
OPTIMIZATION_MANIFEST = OPTIMIZATION.parent / 'manifest.json'
DATES = pd.date_range('2011-06-01', '2011-08-01', freq='MS')
SEED = 42
RESAMPLES = 500
SCORES = {
    'historic_180d': 'historic_180d',
    'prior_year_tweedie': 'extended_tweedie_previous_year',
    'buyer_lag_two_stage': 'expected_spend_lag_calibrated',
}


def load_comparison() -> pd.DataFrame:
    hurdle_manifest = json.loads(HURDLE_MANIFEST.read_text(encoding='utf-8'))
    optimization_manifest = json.loads(
        OPTIMIZATION_MANIFEST.read_text(encoding='utf-8'))
    if (hurdle_manifest.get('test_rows_used') is not False
            or optimization_manifest.get('test_rows_used') is not False
            or sha256(HURDLE) != hurdle_manifest['predictions_sha256']
            or sha256(OPTIMIZATION) != optimization_manifest['development_predictions_sha256']):
        raise ValueError('Development prediction provenance mismatch')
    left = pd.read_csv(HURDLE, dtype={'customer_id': 'string'},
                       parse_dates=['snapshot_date'])
    left = left.loc[left.snapshot_date.isin(DATES)].copy()
    right = pd.read_csv(OPTIMIZATION, dtype={'customer_id': 'string'},
                        parse_dates=['snapshot_date'])
    right = right.loc[right.snapshot_date.isin(DATES)
                      & right.method.isin(SCORES.values())].copy()
    if right.duplicated(['snapshot_date', 'customer_id', 'method']).any():
        raise ValueError('Duplicate model score key')
    right = right.pivot(index=['snapshot_date', 'customer_id', TARGET],
                        columns='method', values='predicted_spend_30d').reset_index()
    if left.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate hurdle score key')
    merged = left.merge(right, on=['snapshot_date', 'customer_id', TARGET],
                        how='left', validate='one_to_one', indicator=True)
    if (len(merged) != len(left) or not merged._merge.eq('both').all()
            or merged[list(SCORES.values())].isna().any().any()
            or set(merged.snapshot_date) != set(DATES)):
        raise ValueError('Mismatched development cohorts or missing predictions')
    merged = merged.drop(columns='_merge').reset_index(drop=True)
    if (merged[list(SCORES.values()) + ['expected_spend_lag_calibrated',
                                       'median_if_purchase_prior_year',
                                       'average_order_value_180d']] < 0).any().any():
        raise ValueError('Negative development scores')
    return merged


def paired_metrics(data: pd.DataFrame) -> dict:
    data = data.reset_index(drop=True)
    baseline = evaluate(data, data.historic_180d.to_numpy(dtype=float))
    direct = evaluate(data, data.extended_tweedie_previous_year.to_numpy(dtype=float))
    two_stage = evaluate(data, data.expected_spend_lag_calibrated.to_numpy(dtype=float))
    buyer = data[TARGET].gt(0)
    if not buyer.any():
        raise ValueError('No buyers in comparison sample')
    actual = data.loc[buyer, TARGET].to_numpy(dtype=float)
    conditional = np.abs(
        data.loc[buyer, 'median_if_purchase_prior_year'].to_numpy(dtype=float)
        - actual).mean()
    prior_order = np.abs(
        data.loc[buyer, 'average_order_value_180d'].to_numpy(dtype=float)
        - actual).mean()
    return {
        'prior_year_tweedie_mae_reduction': 1 - direct['mae'] / baseline['mae'],
        'prior_year_tweedie_top20_gain_pp': 100 * (
            direct['top20_spend_capture'] - baseline['top20_spend_capture']),
        'buyer_lag_two_stage_mae_reduction': 1 - two_stage['mae'] / baseline['mae'],
        'buyer_lag_two_stage_top20_gain_pp': 100 * (
            two_stage['top20_spend_capture'] - baseline['top20_spend_capture']),
        'prior_year_conditional_median_buyer_mae_reduction': (
            1 - conditional / prior_order),
    }


def bootstrap(data: pd.DataFrame, seed: int = SEED,
              resamples: int = RESAMPLES) -> tuple[pd.DataFrame, dict]:
    if seed < 0 or resamples < 2:
        raise ValueError('Invalid bootstrap configuration')
    point = paired_metrics(data)
    groups = data.groupby('customer_id', sort=True).indices
    arrays = list(groups.values())
    n_customers = len(arrays)
    if n_customers < 100:
        raise ValueError('Too few customer clusters')
    rng = np.random.default_rng(seed)
    records = []
    for i in range(resamples):
        sampled = rng.integers(n_customers, size=n_customers)
        positions = np.concatenate([arrays[j] for j in sampled])
        records.append({'resample': i + 1,
                        **paired_metrics(data.iloc[positions])})
        if (i + 1) % 100 == 0:
            print(f'completed customer bootstrap {i + 1}/{resamples}', flush=True)
    draws = pd.DataFrame(records)
    rows = []
    for metric, value in point.items():
        low, high = np.quantile(draws[metric], [.025, .975])
        rows.append({'metric': metric, 'point': value,
                     'lower_95': float(low), 'upper_95': float(high),
                     'resamples': resamples})
    return pd.DataFrame(rows), {'draws': draws, 'customer_clusters': n_customers,
                                'snapshot_count': len(data)}


def main() -> None:
    data = load_comparison()
    intervals, details = bootstrap(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(OUTPUT / 'paired_intervals.csv', index=False,
                     encoding='utf-8-sig')
    details['draws'].to_csv(OUTPUT / 'bootstrap_draws.csv', index=False,
                            encoding='utf-8-sig')
    manifest = {
        'status': 'development_only_customer_sampling_uncertainty',
        'protocol_sha256': sha256(ROOT / 'value_prediction/optimization_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'hurdle_predictions_sha256': sha256(HURDLE),
        'optimization_predictions_sha256': sha256(OPTIMIZATION),
        'intervals_sha256': sha256(OUTPUT / 'paired_intervals.csv'),
        'draws_sha256': sha256(OUTPUT / 'bootstrap_draws.csv'),
        'seed': SEED, 'resamples': RESAMPLES,
        'customer_clusters': details['customer_clusters'],
        'snapshot_count': details['snapshot_count'],
        'dates': [date.strftime('%Y-%m-%d') for date in DATES],
        'test_rows_used': False,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    names = {
        'prior_year_tweedie_mae_reduction': '去年同期直接模型整体 MAE 降幅',
        'prior_year_tweedie_top20_gain_pp': '去年同期直接模型 Top20 覆盖提升（百分点）',
        'buyer_lag_two_stage_mae_reduction': '购买者滞后校准两阶段模型整体 MAE 降幅',
        'buyer_lag_two_stage_top20_gain_pp': '购买者滞后校准两阶段模型 Top20 覆盖提升（百分点）',
        'prior_year_conditional_median_buyer_mae_reduction':
            '去年同期条件中位数对历史订单均价的购买者 MAE 降幅',
    }
    lines = [
        '# 开发确认期客户成组不确定性', '',
        f'2011 年 6–8 月共有 {len(data):,} 条客户月快照、'
        f'{details["customer_clusters"]:,} 名客户。按客户重抽样 {RESAMPLES} 次，'
        f'随机种子 {SEED}，所有方法共享每次抽到的客户。',
        '这是固定三个月内的客户抽样区间，不包含未来月份变化，也不是新的独立测试。',
        '', '| 比较 | 点估计 | 95% 客户抽样区间 |',
        '| --- | ---: | ---: |',
    ]
    for row in intervals.itertuples(index=False):
        suffix = ' 个百分点' if row.metric.endswith('_pp') else '%'
        scale = 1 if suffix.startswith(' 个') else 100
        lines.append(f'| {names[row.metric]} | {row.point * scale:+.2f}{suffix} | '
                     f'[{row.lower_95 * scale:+.2f}{suffix}, '
                     f'{row.upper_95 * scale:+.2f}{suffix}] |')
    lines += ['', '区间只检查这些已查看月份里，客户构成变化会让结论波动多少。'
              '即使整个区间高于零，也不能证明新月份或另一家商家会达到原挑战标准。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote development uncertainty report to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
