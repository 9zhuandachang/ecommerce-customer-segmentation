"""Forward validation of spend conditional on a future purchase.

This is a separate scenario forecast: it cannot identify in advance who will
buy. The unconditional expected-spend score remains in optimization.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.validation import FEATURES as BASE_FEATURES, load_snapshots
from archive.spending.experiments import LABELS, TARGET, attach_labels
from src.spending.labels import SNAPSHOTS, sha256
from archive.spending.optimization import (
    DEVELOPMENT_DATES, check_sources, fit_rows_for_date,
)
from src.spending.features import FEATURE_PATH, FEATURES as EXTRA_FEATURES

from src.paths import ROOT
OUTPUT = ROOT / 'archive/artifacts/spending/conditional_amount'
METHODS = ('historic_monthly_rate', 'average_prior_invoice',
           'conditional_gamma_mean', 'conditional_l1_median')


def conditional_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Score a scenario forecast only where the future purchase occurred."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if (actual.shape != predicted.shape or actual.ndim != 1
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()
            or (actual < 0).any() or (predicted < 0).any()):
        raise ValueError('Invalid conditional forecast inputs')
    buyer = actual > 0
    if not buyer.any():
        raise ValueError('No buyers to evaluate')
    total = float(actual[buyer].sum())
    return {
        'buyer_count': int(buyer.sum()),
        'buyer_mae': float(np.abs(actual[buyer] - predicted[buyer]).mean()),
        'buyer_amount_bias': float(predicted[buyer].sum() / total - 1),
    }


def fit_predict(method: str, fit: pd.DataFrame,
                holdout: pd.DataFrame) -> np.ndarray:
    if method == 'historic_monthly_rate':
        return holdout.monetary_value_180d.to_numpy(dtype=float) / 6
    if method == 'average_prior_invoice':
        return holdout.average_order_value_180d.to_numpy(dtype=float)
    if method not in ('conditional_gamma_mean', 'conditional_l1_median'):
        raise ValueError(f'Unknown conditional method: {method}')
    buyer = fit[TARGET].gt(0)
    if buyer.sum() < 100:
        raise ValueError('Too few mature buyer labels')
    objective = ('gamma' if method == 'conditional_gamma_mean'
                 else 'regression_l1')
    model = lgb.LGBMRegressor(
        objective=objective, n_estimators=120, learning_rate=.05,
        num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
    model.fit(fit.loc[buyer, BASE_FEATURES + EXTRA_FEATURES],
              fit.loc[buyer, TARGET])
    return np.maximum(model.predict(holdout[BASE_FEATURES + EXTRA_FEATURES]), 0)


def main() -> None:
    check_sources()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    snapshots = snapshots.loc[snapshots.split.isin(['train', 'validation'])].copy()
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    labels = labels.loc[labels.split.isin(['train', 'validation'])].copy()
    temporal = pd.read_csv(FEATURE_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    development = attach_labels(snapshots, labels).merge(
        temporal, on=['snapshot_date', 'customer_id'], how='left',
        validate='one_to_one', indicator=True)
    if (not development._merge.eq('both').all()
            or development[EXTRA_FEATURES].isna().any().any()):
        raise ValueError('Missing temporal features')
    development = development.drop(columns='_merge')

    rows, monthly = [], []
    for date in DEVELOPMENT_DATES:
        fit = fit_rows_for_date(development, date)
        holdout = development.loc[development.snapshot_date.eq(date)].copy()
        if holdout.empty:
            raise ValueError(f'No holdout rows: {date.date()}')
        actual = holdout[TARGET].to_numpy(dtype=float)
        for method in METHODS:
            predicted = fit_predict(method, fit, holdout)
            metric = conditional_metrics(actual, predicted)
            monthly.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                            'method': method, **metric})
            part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
            part['method'] = method
            part['predicted_amount_if_purchase_30d'] = predicted
            rows.append(part)
        print(f'completed conditional cutoff {date.date()}', flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    summary = []
    for period, dates in [('rolling_jan_may', DEVELOPMENT_DATES[:5]),
                          ('known_jun_aug', DEVELOPMENT_DATES[5:]),
                          ('rolling_jan_aug', DEVELOPMENT_DATES)]:
        for method in METHODS:
            part = predictions.loc[predictions.snapshot_date.isin(dates)
                                   & predictions.method.eq(method)]
            summary.append({'period': period, 'method': method,
                            **conditional_metrics(
                                part[TARGET].to_numpy(dtype=float),
                                part.predicted_amount_if_purchase_30d.to_numpy(dtype=float))})
    summary = pd.DataFrame(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / 'development_predictions.csv', index=False,
                       encoding='utf-8-sig', date_format='%Y-%m-%d')
    pd.DataFrame(monthly).to_csv(OUTPUT / 'monthly_development.csv', index=False,
                                 encoding='utf-8-sig')
    summary.to_csv(OUTPUT / 'development_comparison.csv', index=False,
                   encoding='utf-8-sig')
    manifest = {
        'status': 'development_only_scenario_forecast_not_unconditional_spend',
        'protocol_sha256': sha256(ROOT / 'archive/docs/spending/optimization_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'snapshot_sha256': sha256(SNAPSHOTS),
        'labels_sha256': sha256(LABELS),
        'temporal_features_sha256': sha256(FEATURE_PATH),
        'test_rows_used': False,
        'candidate_methods': METHODS,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    lines = [
        '#Consumption forecast under purchase conditions: development period results', '',
        'The question is "How much will this customer spend if they make a purchase 30 days in the future?"',
        'During evaluation, the error is only calculated for customers who actually made purchases; the actual purchase status does not enter the prediction features.',
        'This value cannot be used as a proxy for future spending by all customers or as a substitute for unconditional forecasts of buyer error.',
        'The 1 – 5 months and the already viewed 6 – 8 months are only used for development analysis; the 9 – 11 months are not used for model comparison.', '',
        '| Period | Method | Buyer MAE | Buyer Amount Deviation |',
        '| --- | --- | ---: | ---: |',
    ]
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.buyer_mae:.2f} | '
                     f'{row.buyer_amount_bias:+.2%} |')
    lines += ['', 'The historical monthly consumption rate and the average order amount of the last 180 day are simple comparisons using only the data before the snapshot.',
              'L 1 The median of model prediction conditions, suitable for minimizing the absolute error of buyers; its total amount is usually low.',
              'The Gamma model predicts the conditional mean, and the amount deviation is more suitable as a component of the expected amount.',
              'Month-by-month and per-customer output and input hashes are found in the same directory.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote conditional comparison to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
