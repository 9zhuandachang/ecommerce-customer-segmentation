"""Paired customer-cluster uncertainty for the fixed order-dynamics experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.experiments import TARGET
from value_prediction.labels import sha256


HERE = Path(__file__).resolve().parent
INPUT = HERE / 'outputs/order_dynamics_experiment/development_predictions.csv'
INPUT_MANIFEST = INPUT.parent / 'manifest.json'
PROTOCOL = HERE / 'order_dynamics_uncertainty_protocol.md'
OUTPUT = HERE / 'outputs/order_dynamics_uncertainty'
ORIGINAL = 'conditional_l1_median'
CANDIDATE = 'conditional_l1_order_dynamics'
KEYS = ['snapshot_date', 'customer_id']
SEED = 42
RESAMPLES = 500
PERIODS = {
    'rolling_jan_may': pd.date_range('2011-01-01', '2011-05-01', freq='MS'),
    'known_jun_aug': pd.date_range('2011-06-01', '2011-08-01', freq='MS'),
}


def load_comparison() -> pd.DataFrame:
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding='utf-8'))
    if (manifest['test_rows_used'] is not False
            or manifest['development_predictions_sha256'] != sha256(INPUT)
            or manifest['script_sha256'] != sha256(
                HERE / 'order_dynamics_experiment.py')
            or manifest['protocol_sha256'] != sha256(
                HERE / 'order_dynamics_protocol.md')):
        raise ValueError('Order dynamics prediction provenance changed')
    data = pd.read_csv(INPUT, dtype={'customer_id': 'string'},
                       parse_dates=['snapshot_date'])
    data = data.loc[data.method.isin([ORIGINAL, CANDIDATE])].copy()
    if (data.duplicated(KEYS + ['method']).any()
            or set(data.snapshot_date) != set(pd.concat(
                [pd.Series(dates) for dates in PERIODS.values()]))):
        raise ValueError('Comparison has duplicate or unexpected cutoffs')
    old = data.loc[data.method.eq(ORIGINAL),
                   KEYS + [TARGET, 'predicted_amount_if_purchase_30d']].copy()
    new = data.loc[data.method.eq(CANDIDATE),
                   KEYS + [TARGET, 'predicted_amount_if_purchase_30d']].copy()
    merged = old.merge(new, on=KEYS, how='outer', validate='one_to_one',
                       indicator=True, suffixes=('_old', '_new'))
    if (len(merged) != len(old) or len(old) != len(new)
            or not merged._merge.eq('both').all()
            or not np.allclose(merged[f'{TARGET}_old'],
                               merged[f'{TARGET}_new'], rtol=0, atol=1e-9)):
        raise ValueError('Old/new customer-month labels do not reconcile')
    merged = merged.rename(columns={
        f'{TARGET}_old': TARGET,
        'predicted_amount_if_purchase_30d_old': 'old',
        'predicted_amount_if_purchase_30d_new': 'new',
    }).drop(columns=[f'{TARGET}_new', '_merge'])
    if (not np.isfinite(merged[[TARGET, 'old', 'new']].to_numpy(dtype=float)).all()
            or (merged[[TARGET, 'old', 'new']] < 0).any().any()):
        raise ValueError('Invalid actual or predicted amounts')
    return merged.loc[merged[TARGET].gt(0)].reset_index(drop=True)


def paired_metrics(buyers: pd.DataFrame) -> dict:
    if buyers.empty or (buyers[TARGET] <= 0).any():
        raise ValueError('Expected actual buyer rows only')
    actual = buyers[TARGET].to_numpy(dtype=float)
    old = buyers.old.to_numpy(dtype=float)
    new = buyers.new.to_numpy(dtype=float)
    if (not np.isfinite(actual).all() or not np.isfinite(old).all()
            or not np.isfinite(new).all()
            or (old < 0).any() or (new < 0).any()):
        raise ValueError('Invalid paired amount inputs')
    old_mae = np.abs(old - actual).mean()
    new_mae = np.abs(new - actual).mean()
    old_bias = old.sum() / actual.sum() - 1
    new_bias = new.sum() / actual.sum() - 1
    return {
        'buyer_mae_reduction': float(1 - new_mae / old_mae),
        'absolute_bias_difference_pp': float(100 * (abs(new_bias) - abs(old_bias))),
    }


def bootstrap(buyers: pd.DataFrame, seed: int = SEED,
              resamples: int = RESAMPLES) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if seed < 0 or resamples < 2:
        raise ValueError('Invalid bootstrap configuration')
    groups = buyers.groupby('customer_id', sort=True).indices
    arrays = list(groups.values())
    n_customers = len(arrays)
    if n_customers < 100:
        raise ValueError('Too few customer clusters')
    rng = np.random.default_rng(seed)
    point = paired_metrics(buyers)
    draws = []
    for i in range(resamples):
        sampled = rng.integers(n_customers, size=n_customers)
        positions = np.concatenate([arrays[j] for j in sampled])
        draws.append({'resample': i + 1, **paired_metrics(buyers.iloc[positions])})
    draws = pd.DataFrame(draws)
    intervals = pd.DataFrame([
        {'metric': name, 'point': value,
         'lower_95': float(np.quantile(draws[name], .025)),
         'upper_95': float(np.quantile(draws[name], .975))}
        for name, value in point.items()
    ])
    return intervals, draws, n_customers


def main() -> None:
    buyers = load_comparison()
    summaries, draws = [], []
    for period, dates in PERIODS.items():
        part = buyers.loc[buyers.snapshot_date.isin(dates)].reset_index(drop=True)
        intervals, samples, n_customers = bootstrap(part)
        intervals.insert(0, 'period', period)
        intervals['buyer_snapshot_count'] = len(part)
        intervals['buyer_customer_count'] = n_customers
        samples.insert(0, 'period', period)
        summaries.append(intervals)
        draws.append(samples)
        print(f'completed uncertainty period {period}', flush=True)
    result = pd.concat(summaries, ignore_index=True)
    samples = pd.concat(draws, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / 'paired_intervals.csv', index=False, encoding='utf-8-sig')
    samples.to_csv(OUTPUT / 'bootstrap_draws.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'status': 'development_only_customer_sampling_not_future_month_validation',
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'input_predictions_sha256': sha256(INPUT),
        'intervals_sha256': sha256(OUTPUT / 'paired_intervals.csv'),
        'draws_sha256': sha256(OUTPUT / 'bootstrap_draws.csv'),
        'seed': SEED, 'resamples': RESAMPLES,
        'test_rows_used': False,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['#Characteristics of order amount changes: customer sampling uncertainty', '',
             'The purchase records of the same customer across months are resampled together, and the two models use the same sample.'
             'The range only describes fluctuations in customer composition for a fixed viewed month and does not include differences in future months.', '',
             '| Period | Metric | Point Estimate | 95% Customer Sampling Range | Number of Customers |',
             '| --- | --- | ---: | ---: | ---: |']
    for row in result.itertuples(index=False):
        if row.metric == 'buyer_mae_reduction':
            name, scale, suffix = 'Buyer MAE Decrease', 100, '%'
        else:
            name, scale, suffix = 'Absolute amount deviation change', 1, 'percentage points'
        lines.append(f'| {row.period} | {name} | {row.point * scale:+.2f}{suffix} | '
                     f'[{row.lower_95 * scale:+.2f}{suffix}, '
                     f'{row.upper_95 * scale:+.2f}{suffix}] | '
                     f'{row.buyer_customer_count:,} |')
    lines += ['', 'A positive "Change in Absolute Amount Bias" indicates that the new feature makes the amount more biased.'
              'This range does not replace prior 5% improvement thresholds, cross-month stability, or independent acceptance.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote order dynamics uncertainty to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
