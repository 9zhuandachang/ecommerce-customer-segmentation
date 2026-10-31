"""Forward development check of expected orders times expected order value."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.validation import FEATURES as BASE_FEATURES, load_snapshots
from archive.spending.experiments import LABELS, TARGET, attach_labels
from src.spending.labels import CLEAN, SCHEDULE, SNAPSHOTS, sha256
from archive.spending.optimization import (
    DEVELOPMENT_DATES, check_sources, fit_predict, fit_rows_for_date,
    score_with_group_bias,
)
from src.spending.features import FEATURE_PATH, FEATURES as EXTRA_FEATURES

from src.paths import ROOT
OUTPUT = ROOT / 'archive/artifacts/spending/count_amount'
PROTOCOL = ROOT / 'archive/docs/spending/count_amount_protocol.md'
FEATURES = BASE_FEATURES + EXTRA_FEATURES
METHODS = ('historic_180d', 'extended_tweedie', 'count_times_order_amount')
KEYS = ['snapshot_date', 'customer_id']


def build_future_invoice_counts(snapshots: pd.DataFrame, clean: pd.DataFrame,
                                schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count distinct invoices only within each complete future 30-day window."""
    snapshots, clean, schedule = snapshots.copy(), clean.copy(), schedule.copy()
    snapshots['snapshot_date'] = pd.to_datetime(snapshots.snapshot_date, errors='raise')
    clean['invoice_date'] = pd.to_datetime(clean.invoice_date, errors='raise')
    schedule['snapshot_date'] = pd.to_datetime(schedule.snapshot_date, errors='raise')
    schedule['label_end_exclusive'] = pd.to_datetime(
        schedule.label_end_exclusive, errors='raise')
    snapshots['customer_id'] = snapshots.customer_id.astype('string')
    clean['customer_id'] = clean.customer_id.astype('string')
    if (snapshots.duplicated(KEYS).any() or schedule.snapshot_date.duplicated().any()
            or not set(snapshots.snapshot_date).issubset(set(schedule.snapshot_date))
            or (schedule.label_end_exclusive - schedule.snapshot_date
                != pd.Timedelta(days=30)).any()):
        raise ValueError('Invalid invoice-count snapshot schedule')
    if clean[['customer_id', 'invoice', 'invoice_date']].isna().any().any():
        raise ValueError('Missing eligible transaction key for invoice counts')
    parts, audits = [], []
    for row in (schedule.loc[schedule.snapshot_date.isin(snapshots.snapshot_date)]
                .sort_values('snapshot_date').itertuples(index=False)):
        cohort = snapshots.loc[snapshots.snapshot_date.eq(row.snapshot_date), KEYS]
        future = clean.loc[clean.invoice_date.ge(row.snapshot_date)
                           & clean.invoice_date.lt(row.label_end_exclusive),
                           ['customer_id', 'invoice']]
        counts = future.groupby('customer_id', observed=True).invoice.nunique()
        part = cohort.copy()
        part['invoice_count_30d_future'] = (
            part.customer_id.map(counts).fillna(0).astype('int64'))
        cohort_invoices = (future.loc[future.customer_id.isin(cohort.customer_id)]
                           .drop_duplicates(['customer_id', 'invoice']))
        if int(part.invoice_count_30d_future.sum()) != len(cohort_invoices):
            raise AssertionError('Future invoice count does not reconcile')
        audits.append({
            'snapshot_date': row.snapshot_date,
            'label_end_exclusive': row.label_end_exclusive,
            'snapshot_count': len(cohort),
            'buyer_count': int(part.invoice_count_30d_future.gt(0).sum()),
            'cohort_invoice_count': len(cohort_invoices),
            'reconciled': True,
        })
        parts.append(part)
    counts = pd.concat(parts, ignore_index=True)
    if len(counts) != len(snapshots) or counts.duplicated(KEYS).any():
        raise AssertionError('Invoice count does not cover snapshots')
    return counts, pd.DataFrame(audits)


def fit_predict_count_amount(fit: pd.DataFrame,
                             target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train separate mature-label models for order count and mean order value."""
    if (fit.empty or target.empty
            or (fit.snapshot_date + pd.Timedelta(days=30)
                > target.snapshot_date.min()).any()):
        raise ValueError('Count-amount training contains immature labels')
    count = fit.invoice_count_30d_future.to_numpy(dtype=float)
    spend = fit[TARGET].to_numpy(dtype=float)
    if ((count < 0).any() or (spend < 0).any()
            or not np.equal(count, np.rint(count)).all()
            or not np.array_equal(count > 0, spend > 0)):
        raise ValueError('Count and spend labels disagree')
    buyers = count > 0
    if buyers.sum() < 100 or (~buyers).sum() < 100:
        raise ValueError('Too few buyer or zero-order training rows')
    parameters = dict(n_estimators=120, learning_rate=.05, num_leaves=15,
                      random_state=42, n_jobs=1, verbose=-1)
    count_model = lgb.LGBMRegressor(objective='poisson', **parameters)
    count_model.fit(fit[FEATURES], count)
    amount_model = lgb.LGBMRegressor(objective='gamma', **parameters)
    amount_model.fit(fit.loc[buyers, FEATURES], spend[buyers] / count[buyers])
    expected_count = np.maximum(count_model.predict(target[FEATURES]), 0)
    expected_order_amount = np.maximum(amount_model.predict(target[FEATURES]), 0)
    expected_spend = expected_count * expected_order_amount
    if any(not np.isfinite(value).all() for value in (
            expected_count, expected_order_amount, expected_spend)):
        raise ValueError('Count-amount forecast is nonfinite')
    return expected_spend, expected_count, expected_order_amount


def advancement_checks(summary: pd.DataFrame) -> dict:
    rows = summary.loc[summary.period.eq('rolling_jan_may')].set_index('method')
    original = rows.loc['extended_tweedie']
    candidate = rows.loc['count_times_order_amount']
    return {
        'buyer_mae_reduction_at_least_5pct':
        bool(candidate.positive_mae <= .95 * original.positive_mae),
        'overall_mae_increase_at_most_2pct':
        bool(candidate.mae <= 1.02 * original.mae),
        'top20_capture_not_lower':
        bool(candidate.top20_spend_capture >= original.top20_spend_capture),
        'aggregate_bias_abs_not_larger':
        bool(abs(candidate.aggregate_bias) <= abs(original.aggregate_bias)),
    }


def load_development_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    check_sources()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    snapshots = snapshots.loc[snapshots.split.isin(['train', 'validation'])].copy()
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    labels = labels.loc[labels.split.isin(['train', 'validation'])].copy()
    temporal = pd.read_csv(FEATURE_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    data = attach_labels(snapshots, labels).merge(
        temporal, on=KEYS, how='left', validate='one_to_one', indicator=True)
    if (not data._merge.eq('both').all() or data[EXTRA_FEATURES].isna().any().any()):
        raise ValueError('Missing pre-cutoff monetary features')
    data = data.drop(columns='_merge')
    schedule = pd.read_csv(SCHEDULE, parse_dates=['snapshot_date',
                                                   'label_end_exclusive'])
    clean = pd.read_csv(CLEAN, dtype={'customer_id': 'string', 'invoice': 'string'},
                        usecols=['customer_id', 'invoice', 'invoice_date'],
                        parse_dates=['invoice_date'])
    counts, audit = build_future_invoice_counts(data, clean, schedule)
    data = data.merge(counts, on=KEYS, validate='one_to_one')
    if (len(data) != len(snapshots) or data[FEATURES].isna().any().any()
            or not np.array_equal(data.invoice_count_30d_future.gt(0).to_numpy(),
                                  data[TARGET].gt(0).to_numpy())):
        raise ValueError('Invoice count, spend label or feature reconciliation failed')
    return data, audit


def main() -> None:
    data, audit = load_development_data()
    predictions, monthly, folds = [], [], []
    for date in DEVELOPMENT_DATES:
        fit = fit_rows_for_date(data, date)
        target = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if target.empty:
            raise ValueError(f'No development customers for {date.date()}')
        candidate, predicted_count, predicted_order_amount = (
            fit_predict_count_amount(fit, target))
        scores = {
            'historic_180d': fit_predict('historic_180d', fit, target),
            'extended_tweedie': fit_predict('extended_tweedie', fit, target),
            'count_times_order_amount': candidate,
        }
        folds.append({
            'snapshot_date': date.strftime('%Y-%m-%d'),
            'fit_rows': len(fit), 'target_rows': len(target),
            'latest_fit_label_end': (
                fit.snapshot_date.max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
        })
        for method, values in scores.items():
            if len(values) != len(target) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Invalid {method} score for {date.date()}')
            monthly.append({
                'snapshot_date': date.strftime('%Y-%m-%d'), 'method': method,
                'zero_spend_share': float(target[TARGET].eq(0).mean()),
                **score_with_group_bias(target, values),
            })
            part = target[KEYS + [TARGET]].copy()
            part['method'] = method
            part['predicted_spend_30d'] = values
            if method == 'count_times_order_amount':
                part['predicted_invoice_count_30d'] = predicted_count
                part['predicted_amount_per_invoice'] = predicted_order_amount
            predictions.append(part)
        print(f'completed count-amount cutoff {date.date()}', flush=True)

    forecast = pd.concat(predictions, ignore_index=True)
    monthly_frame = pd.DataFrame(monthly)
    summary_rows = []
    for period, dates in [('rolling_jan_may', DEVELOPMENT_DATES[:5]),
                          ('known_jun_aug', DEVELOPMENT_DATES[5:])]:
        for method in METHODS:
            part = forecast.loc[forecast.method.eq(method)
                                & forecast.snapshot_date.isin(dates)].reset_index(drop=True)
            values = part.predicted_spend_30d.to_numpy(dtype=float)
            summary_rows.append({
                'period': period, 'method': method,
                'zero_spend_share': float(part[TARGET].eq(0).mean()),
                **score_with_group_bias(part, values),
            })
    summary = pd.DataFrame(summary_rows)
    count_diagnostics = []
    for period, dates in [('rolling_jan_may', DEVELOPMENT_DATES[:5]),
                          ('known_jun_aug', DEVELOPMENT_DATES[5:])]:
        predicted = forecast.loc[forecast.method.eq('count_times_order_amount')
                                 & forecast.snapshot_date.isin(dates)].copy()
        actual = data.loc[data.snapshot_date.isin(dates),
                          KEYS + ['invoice_count_30d_future']]
        joined = predicted.merge(actual, on=KEYS, validate='one_to_one')
        if len(joined) != len(actual):
            raise ValueError('Count diagnostics do not cover cohort')
        actual_count = float(joined.invoice_count_30d_future.sum())
        total_spend = float(joined[TARGET].sum())
        count_diagnostics.append({
            'period': period, 'actual_invoice_count': int(actual_count),
            'predicted_invoice_count': float(joined.predicted_invoice_count_30d.sum()),
            'predicted_count_to_actual': float(
                joined.predicted_invoice_count_30d.sum() / actual_count),
            'predicted_order_amount_on_actual_orders_to_actual_spend': float(
                (joined.predicted_amount_per_invoice
                 * joined.invoice_count_30d_future).sum() / total_spend),
        })
    diagnostics = pd.DataFrame(count_diagnostics)
    checks = advancement_checks(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT / 'development_predictions.csv', index=False,
                    encoding='utf-8-sig', date_format='%Y-%m-%d')
    monthly_frame.to_csv(OUTPUT / 'monthly_metrics.csv', index=False,
                         encoding='utf-8-sig')
    summary.to_csv(OUTPUT / 'period_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(OUTPUT / 'folds.csv', index=False,
                               encoding='utf-8-sig')
    diagnostics.to_csv(OUTPUT / 'count_diagnostics.csv', index=False,
                       encoding='utf-8-sig')
    audit.to_csv(OUTPUT / 'invoice_count_audit.csv', index=False,
                 encoding='utf-8-sig', date_format='%Y-%m-%d')
    manifest = {
        'status': 'development_only_not_independent_acceptance',
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'snapshot_sha256': sha256(SNAPSHOTS),
        'labels_sha256': sha256(LABELS),
        'clean_sha256': sha256(CLEAN),
        'schedule_sha256': sha256(SCHEDULE),
        'temporal_features_sha256': sha256(FEATURE_PATH),
        'invoice_count_reconciled': bool(audit.reconciled.all()),
        'development_snapshot_count': len(data),
        'test_rows_used': False,
        'advancement_checks': checks,
        'advance_to_new_test': all(checks.values()),
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Number of orders × Average order amount: Development period experiment', '',
        'The only candidate for ex ante fixation is to use mature historical labels to separately predict the number of orders and the amount of each transaction.'
        'Then multiply them to get the unconditional estimated consumption amount in the next 30 days. The number of orders is based on unique invoices.'
        'Multiple items in the same order will not be counted repeatedly. 2011 years 9 – 11 months were not used in this experiment.', '',
        '| Period | Method | All MAE | Buyers MAE | Top 20 Consumption Coverage | Total Deviation | Top 20 Group Deviation |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | {row.top20_group_bias:+.2%} |')
    lines += ['', '## Post-disassembly: number of orders and average order amount', '',
              '| Period | Actual number of orders | Forecast number of orders/actual | Forecast average order amount/actual amount weighted by actual order number |',
              '| --- | ---: | ---: | ---: |']
    for row in diagnostics.itertuples(index=False):
        lines.append(f'| {row.period} | {row.actual_invoice_count:,} | '
                     f'{row.predicted_count_to_actual:.2%} | '
                     f'{row.predicted_order_amount_on_actual_orders_to_actual_spend:.2%} |')
    lines += ['', 'This teardown is only used for error diagnosis. Actual future order numbers are not used to generate forecast scores.']
    lines += ['', '## Fixed in advance 1 – 5 monthly advancement conditions', '']
    for name, passed in checks.items():
        lines.append(f'- `{name}`: {"satisfy" if passed else "not satisfied"}')
    lines += ['', f'Is it worth going into a new external fixation test: **{"Yes" if all(checks.values()) else "No"}**.'
              'Even if it is satisfied, it can only mean that it is worthy of re-inspection and cannot be used as a basis for independent acceptance or system access.',
              'Customer-level development forecasts, month-by-month metrics, tag checks, training cutoffs and file hashes are in the same directory.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote count-amount development comparison to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
