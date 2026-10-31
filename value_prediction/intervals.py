"""Leakage-safe rolling spend intervals from already saved OOF predictions.

Run ``python -m value_prediction.intervals``. This is development evidence,
not a guarantee of nominal coverage on unseen future months.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.experiments import TARGET
from value_prediction.hurdle_diagnostics import high_amount_mask
from value_prediction.labels import sha256

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/intervals'
PROTOCOL = ROOT / 'value_prediction/interval_protocol.md'
HURDLE = ROOT / 'value_prediction/outputs/hurdle_diagnostics/customer_predictions.csv'
HURDLE_MANIFEST = HURDLE.parent / 'manifest.json'
OPTIMIZATION = ROOT / 'value_prediction/outputs/optimization/development_predictions.csv'
OPTIMIZATION_MANIFEST = OPTIMIZATION.parent / 'manifest.json'
DATES = pd.date_range('2011-02-01', '2011-08-01', freq='MS')
NOMINAL = .90
POINTS = {
    'unconditional': 'predicted_spend_30d',
    'if_purchase': 'median_if_purchase_prior_year',
}
METHODS = ('absolute', 'log1p', 'absolute_grouped', 'log1p_grouped')


def finite_quantile(residuals: np.ndarray,
                    nominal: float = NOMINAL) -> float:
    """The split-conformal upper order statistic without interpolation."""
    values = np.asarray(residuals, dtype=float)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or (values < 0).any() or not 0 < nominal < 1):
        raise ValueError('Invalid finite-sample residuals')
    rank = int(np.ceil((len(values) + 1) * nominal))
    if rank > len(values):
        raise ValueError('Too few residuals for a finite interval')
    return float(np.partition(values, rank - 1)[rank - 1])


def amount_interval(point: np.ndarray, residual_quantile: float,
                    method: str) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(point, dtype=float)
    if (scores.ndim != 1 or not np.isfinite(scores).all() or (scores < 0).any()
            or not np.isfinite(residual_quantile) or residual_quantile < 0):
        raise ValueError('Invalid amount interval inputs')
    if method == 'absolute':
        low = np.maximum(scores - residual_quantile, 0)
        high = scores + residual_quantile
    elif method == 'log1p':
        center = np.log1p(scores)
        low = np.expm1(np.maximum(center - residual_quantile, 0))
        high = np.expm1(center + residual_quantile)
    else:
        raise ValueError(f'Unknown interval method: {method}')
    if (not np.isfinite(low).all() or not np.isfinite(high).all()
            or (low < 0).any() or (high < low).any()):
        raise ValueError('Invalid interval bounds')
    return low, high


def historical_residuals(calibration: pd.DataFrame,
                         score_column: str, method: str) -> np.ndarray:
    actual = calibration[TARGET].to_numpy(dtype=float)
    predicted = calibration[score_column].to_numpy(dtype=float)
    if method == 'absolute':
        return np.abs(actual - predicted)
    if method == 'log1p':
        return np.abs(np.log1p(actual) - np.log1p(predicted))
    raise ValueError(f'Unknown historical residual method: {method}')


def calibrated_interval(calibration: pd.DataFrame, current: pd.DataFrame,
                        target: str, method: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply all-customer or pre-score-group finite-sample quantiles."""
    if target not in POINTS or method not in METHODS:
        raise ValueError('Unknown interval target or method')
    point_column = POINTS[target]
    base_method = method.removesuffix('_grouped')
    point = current[point_column].to_numpy(dtype=float)
    if not method.endswith('_grouped'):
        quantile = finite_quantile(historical_residuals(
            calibration, point_column, base_method))
        low, high = amount_interval(point, quantile, base_method)
        return low, high, {'quantile': quantile,
                           'calibration_rows': len(calibration)}
    high_flag = f'high_{target}'
    if (high_flag not in calibration or high_flag not in current
            or not calibration[high_flag].isin([True, False]).all()
            or not current[high_flag].isin([True, False]).all()):
        raise ValueError('Missing pre-outcome amount group')
    low = np.empty(len(current), dtype=float)
    high = np.empty(len(current), dtype=float)
    audit = {}
    current_group = current[high_flag].to_numpy(dtype=bool)
    for segment, name in ((False, 'ordinary'), (True, 'high')):
        group = calibration.loc[calibration[high_flag].eq(segment)]
        fallback = len(group) < 100
        source = calibration if fallback else group
        quantile = finite_quantile(historical_residuals(
            source, point_column, base_method))
        selected = current_group == segment
        low[selected], high[selected] = amount_interval(
            point[selected], quantile, base_method)
        audit[f'{name}_quantile'] = quantile
        audit[f'{name}_calibration_rows'] = len(source)
        audit[f'{name}_fell_back_to_all'] = fallback
    return low, high, audit


def load_predictions() -> pd.DataFrame:
    h_manifest = json.loads(HURDLE_MANIFEST.read_text(encoding='utf-8'))
    o_manifest = json.loads(OPTIMIZATION_MANIFEST.read_text(encoding='utf-8'))
    if (h_manifest.get('test_rows_used') is not False
            or o_manifest.get('test_rows_used') is not False
            or h_manifest['predictions_sha256'] != sha256(HURDLE)
            or o_manifest['development_predictions_sha256'] != sha256(OPTIMIZATION)):
        raise ValueError('OOF prediction source hash or scope mismatch')
    left = pd.read_csv(HURDLE, dtype={'customer_id': 'string'},
                       parse_dates=['snapshot_date'])
    right = pd.read_csv(OPTIMIZATION, dtype={'customer_id': 'string'},
                        parse_dates=['snapshot_date'])
    right = right.loc[right.method.eq('extended_tweedie_previous_year'),
                      ['snapshot_date', 'customer_id', TARGET,
                       'predicted_spend_30d']]
    keys = ['snapshot_date', 'customer_id', TARGET]
    if left.duplicated(keys[:2]).any() or right.duplicated(keys[:2]).any():
        raise ValueError('Duplicate customer-month prediction key')
    data = left[keys + ['median_if_purchase_prior_year']].merge(
        right, on=keys, how='left', validate='one_to_one', indicator=True)
    if (len(data) != len(left) or not data._merge.eq('both').all()
            or data[list(POINTS.values())].isna().any().any()
            or set(data.snapshot_date) != set(pd.date_range(
                '2011-01-01', '2011-08-01', freq='MS'))):
        raise ValueError('Incomplete or mismatched OOF prediction cohorts')
    if ((data[list(POINTS.values())] < 0).any().any()
            or (data[TARGET] < 0).any()):
        raise ValueError('Negative prediction or target')
    data = data.drop(columns='_merge').reset_index(drop=True)
    for target, score_column in POINTS.items():
        data[f'high_{target}'] = False
        for _, part in data.groupby('snapshot_date', sort=True):
            mask = high_amount_mask(part.reset_index(drop=True),
                                    part[score_column].to_numpy(dtype=float))
            data.loc[part.index, f'high_{target}'] = mask
    return data


def recent_mature_history(data: pd.DataFrame,
                          target_date: pd.Timestamp) -> tuple[pd.DataFrame, list[str]]:
    mature = data.loc[data.snapshot_date.lt(target_date)
                      & data.snapshot_date.add(pd.Timedelta(days=30)).le(target_date)]
    dates = sorted(mature.snapshot_date.unique())[-3:]
    recent = mature.loc[mature.snapshot_date.isin(dates)].copy()
    if recent.empty or (recent.snapshot_date + pd.Timedelta(days=30)
                        > target_date).any():
        raise ValueError('No fully mature interval calibration cohorts')
    return recent, [pd.Timestamp(date).strftime('%Y-%m-%d') for date in dates]


def interval_metrics(frame: pd.DataFrame, target: str,
                     method: str, group: str) -> dict:
    eligible = frame.loc[frame[TARGET].gt(0)] if target == 'if_purchase' else frame
    if group == 'predicted_top10':
        eligible = eligible.loc[eligible[f'high_{target}']]
    elif group != 'all':
        raise ValueError(f'Unknown interval group: {group}')
    if eligible.empty:
        raise ValueError('No rows for interval metric')
    low = eligible[f'{target}_{method}_low'].to_numpy(dtype=float)
    high = eligible[f'{target}_{method}_high'].to_numpy(dtype=float)
    actual = eligible[TARGET].to_numpy(dtype=float)
    covered = (actual >= low) & (actual <= high)
    width = high - low
    return {
        'row_count': len(eligible), 'actual_spend': float(actual.sum()),
        'empirical_coverage': float(covered.mean()),
        'above_upper_share': float((actual > high).mean()),
        'mean_width': float(width.mean()),
        'median_width': float(np.median(width)),
        'mean_upper': float(high.mean()),
    }


def main() -> None:
    data = load_predictions()
    parts, folds, monthly = [], [], []
    for date in DATES:
        history, history_dates = recent_mature_history(data, date)
        current = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if current.empty:
            raise ValueError(f'No interval cohort at {date.date()}')
        fold = {'target_date': date.strftime('%Y-%m-%d'),
                'mature_history_cutoffs': '|'.join(history_dates),
                'latest_history_label_end': (
                    history.snapshot_date.max() + pd.Timedelta(days=30)
                ).strftime('%Y-%m-%d')}
        for target, score_column in POINTS.items():
            calibration = (history.loc[history[TARGET].gt(0)].copy()
                           if target == 'if_purchase' else history)
            minimum = 100 if target == 'if_purchase' else 500
            if len(calibration) < minimum:
                raise ValueError(f'Too few {target} calibration rows at {date.date()}')
            fold[f'{target}_calibration_rows'] = len(calibration)
            for method in METHODS:
                low, high, audit = calibrated_interval(
                    calibration, current, target, method)
                current[f'{target}_{method}_low'] = low
                current[f'{target}_{method}_high'] = high
                fold.update({f'{target}_{method}_{key}': value
                             for key, value in audit.items()})
                for group in ('all', 'predicted_top10'):
                    monthly.append({
                        'snapshot_date': date.strftime('%Y-%m-%d'),
                        'target': target, 'method': method, 'group': group,
                        **interval_metrics(current, target, method, group),
                    })
        parts.append(current)
        folds.append(fold)
        print(f'completed interval cutoff {date.date()}', flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    monthly = pd.DataFrame(monthly)
    pooled = pd.DataFrame([
        {'period': period, 'target': target, 'method': method, 'group': group,
         **interval_metrics(predictions.loc[
             predictions.snapshot_date.isin(dates)].copy(),
             target, method, group)}
        for period, dates in (
            ('rolling_feb_may', DATES[:4]),
            ('known_jun_aug', DATES[4:]),
        )
        for target in POINTS
        for method in METHODS
        for group in ('all', 'predicted_top10')
    ])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / 'customer_intervals.csv', index=False,
                       encoding='utf-8-sig', date_format='%Y-%m-%d')
    monthly.to_csv(OUTPUT / 'monthly_metrics.csv', index=False,
                   encoding='utf-8-sig')
    pooled.to_csv(OUTPUT / 'pooled_metrics.csv', index=False,
                  encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(OUTPUT / 'folds.csv', index=False,
                               encoding='utf-8-sig')
    manifest = {
        'status': 'development_only_empirical_interval_coverage',
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'hurdle_predictions_sha256': sha256(HURDLE),
        'optimization_predictions_sha256': sha256(OPTIMIZATION),
        'intervals_sha256': sha256(OUTPUT / 'customer_intervals.csv'),
        'monthly_sha256': sha256(OUTPUT / 'monthly_metrics.csv'),
        'folds_sha256': sha256(OUTPUT / 'folds.csv'),
        'nominal_coverage': NOMINAL,
        'methods': METHODS,
        'test_rows_used': False,
        'snapshots_scored': len(predictions),
        'folds': folds,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    lines = [
        '# Future 30 days amount prediction interval: development period check', '',
        'Fixed scheme: `interval_protocol.md`. Unconditional amounts use the year-ago direct model;'
        'The "if purchased" amount is the conditional median. The range utilizes only forecast residuals from past months that have ended at that time.',
        'Nominal coverage 90%, while comparing the original amount absolute residual with the `log1p` amount residual.'
        '6 – 8 Viewed months ago, the following are development confirmation results rather than independent acceptance.', '',
        '| Period | Goal | Form | Customer Group | Experience Coverage | Above Upper Bound | Average Width | Median Width |',
        '| --- | --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f'| {row.period} | {row.target} | {row.method} | '
                     f'{row.group} | {row.empirical_coverage:.2%} | '
                     f'{row.above_upper_share:.2%} | {row.mean_width:.2f} | '
                     f'{row.median_width:.2f} |')
    lines += ['', '## Month-by-month coverage', '',
              '| Cutoff | Target | Form | Universal coverage | Pre-prediction 10% coverage |',
              '| --- | --- | --- | ---: | ---: |']
    for date in DATES:
        for target in POINTS:
            for method in METHODS:
                row_all = monthly.loc[
                    monthly.snapshot_date.eq(date.strftime('%Y-%m-%d'))
                    & monthly.target.eq(target) & monthly.method.eq(method)
                    & monthly.group.eq('all')].iloc[0]
                row_top = monthly.loc[
                    monthly.snapshot_date.eq(date.strftime('%Y-%m-%d'))
                    & monthly.target.eq(target) & monthly.method.eq(method)
                    & monthly.group.eq('predicted_top10')].iloc[0]
                lines.append(f'| {date.date()} | {target} | {method} | '
                             f'{row_all.empirical_coverage:.2%} | '
                             f'{row_top.empirical_coverage:.2%} |')
    lines += ['', 'Interval width and coverage are empirical results for the same set of months viewed.'
              'Customer duplication across months, long tail amounts and seasonal variations mean that the 90% notional does not guarantee future coverage.'
              'The interval does not replace the enhanced acceptance of the original single point amount model, nor is it connected to the dashboard.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote interval report to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
