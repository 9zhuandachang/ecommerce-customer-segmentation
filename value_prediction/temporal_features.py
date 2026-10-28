"""Build auditable pre-cutoff spend features for every existing snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.labels import CLEAN, OUTPUT, SCHEDULE, SNAPSHOTS, sha256

KEYS = ['snapshot_date', 'customer_id']
FEATURES = [
    'monetary_value_30d',
    'monetary_value_60d',
    'monetary_value_90d',
    'monetary_value_previous_90d',
    'last_invoice_value_180d',
    'max_invoice_value_180d',
    'median_invoice_value_180d',
    'active_purchase_days_180d',
    'invoice_count_previous_90d',
    'lifetime_spend_before_cutoff',
    'lifetime_invoice_count_before_cutoff',
    'spend_before_180d_window',
]
FEATURE_PATH = OUTPUT / 'temporal_features.csv'
MANIFEST_PATH = OUTPUT / 'temporal_feature_manifest.json'


def _window_amount(observed: pd.DataFrame, cutoff: pd.Timestamp,
                   days: int) -> pd.Series:
    return (observed.loc[observed.invoice_date.ge(cutoff - pd.Timedelta(days=days))]
            .groupby('customer_id', observed=True).line_cents.sum())


def build_temporal_features(snapshots: pd.DataFrame, clean: pd.DataFrame,
                            schedule: pd.DataFrame) -> pd.DataFrame:
    """Use only transactions before cutoff; retain 180-day and lifetime sums."""
    snapshots = snapshots.copy()
    clean = clean.copy()
    schedule = schedule.copy()
    snapshots['snapshot_date'] = pd.to_datetime(snapshots.snapshot_date, errors='raise')
    clean['invoice_date'] = pd.to_datetime(clean.invoice_date, errors='raise')
    schedule['snapshot_date'] = pd.to_datetime(schedule.snapshot_date, errors='raise')
    schedule['feature_start'] = pd.to_datetime(schedule.feature_start, errors='raise')
    snapshots['customer_id'] = snapshots.customer_id.astype('string')
    clean['customer_id'] = clean.customer_id.astype('string')
    if snapshots.duplicated(KEYS).any() or schedule.snapshot_date.duplicated().any():
        raise ValueError('Duplicate snapshot or cutoff key')
    if set(snapshots.snapshot_date) != set(schedule.snapshot_date):
        raise ValueError('Snapshot and schedule cutoff dates differ')
    if (schedule.snapshot_date - schedule.feature_start != pd.Timedelta(days=180)).any():
        raise ValueError('Expected an exact 180-day feature window')
    amounts = pd.to_numeric(clean.line_amount, errors='raise').to_numpy(dtype=float)
    cents = np.rint(amounts * 100)
    if (not np.isfinite(amounts).all() or (amounts <= 0).any()
            or not np.allclose(amounts * 100, cents, rtol=0, atol=1e-6)):
        raise ValueError('Eligible line amounts must be positive exact cents')
    clean['line_cents'] = cents.astype('int64')

    parts: list[pd.DataFrame] = []
    for row in schedule.sort_values('snapshot_date').itertuples(index=False):
        cutoff, start = row.snapshot_date, row.feature_start
        cohort = snapshots.loc[snapshots.snapshot_date.eq(cutoff),
                               KEYS + ['monetary_value_180d']].copy()
        past = clean.loc[clean.invoice_date.lt(cutoff)]
        observed = past.loc[past.invoice_date.ge(start)]
        if cohort.empty or observed.empty:
            raise ValueError(f'Empty cohort or past transactions: {cutoff.date()}')
        full_amount = observed.groupby('customer_id', observed=True).line_cents.sum()
        lifetime_amount = past.groupby('customer_id', observed=True).line_cents.sum()
        earlier_amount = (past.loc[past.invoice_date.lt(start)]
                          .groupby('customer_id', observed=True).line_cents.sum())
        expected = np.rint(cohort.monetary_value_180d.to_numpy(dtype=float) * 100)
        actual = cohort.customer_id.map(full_amount).fillna(0).to_numpy(dtype='int64')
        if not np.array_equal(actual, expected.astype('int64')):
            raise ValueError(f'180-day spend does not reconcile: {cutoff.date()}')
        earlier = cohort.customer_id.map(earlier_amount).fillna(0).to_numpy(dtype='int64')
        lifetime = cohort.customer_id.map(lifetime_amount).fillna(0).to_numpy(dtype='int64')
        if not np.array_equal(lifetime - earlier, actual):
            raise ValueError(f'Lifetime spend does not reconcile: {cutoff.date()}')

        amount_30 = _window_amount(observed, cutoff, 30)
        amount_60 = _window_amount(observed, cutoff, 60)
        amount_90 = _window_amount(observed, cutoff, 90)
        previous_90 = full_amount.sub(amount_90, fill_value=0)
        invoice_level = (observed.groupby(['customer_id', 'invoice'], observed=True,
                                          as_index=False)
                         .agg(invoice_date=('invoice_date', 'max'),
                              invoice_cents=('line_cents', 'sum')))
        invoice_group = invoice_level.groupby('customer_id', observed=True)
        last_invoice = (invoice_level.sort_values(['invoice_date', 'invoice'], kind='stable')
                        .drop_duplicates('customer_id', keep='last')
                        .set_index('customer_id').invoice_cents)
        max_invoice = invoice_group.invoice_cents.max()
        median_invoice = invoice_group.invoice_cents.median()
        active_days = (invoice_level.assign(purchase_day=invoice_level.invoice_date.dt.floor('D'))
                       .groupby('customer_id', observed=True).purchase_day.nunique())
        previous_invoice_count = (invoice_level.loc[
            invoice_level.invoice_date.lt(cutoff - pd.Timedelta(days=90))]
            .groupby('customer_id', observed=True).invoice.size())

        customer = cohort.customer_id
        cohort['lifetime_spend_before_cutoff'] = lifetime / 100
        cohort['spend_before_180d_window'] = earlier / 100
        cohort['lifetime_invoice_count_before_cutoff'] = (
            customer.map(past.groupby('customer_id', observed=True).invoice.nunique())
            .fillna(0).to_numpy(dtype='int64'))
        for name, values in [
            ('monetary_value_30d', amount_30),
            ('monetary_value_60d', amount_60),
            ('monetary_value_90d', amount_90),
            ('monetary_value_previous_90d', previous_90),
            ('last_invoice_value_180d', last_invoice),
            ('max_invoice_value_180d', max_invoice),
            ('median_invoice_value_180d', median_invoice),
        ]:
            cohort[name] = customer.map(values).fillna(0).to_numpy(dtype=float) / 100
        cohort['active_purchase_days_180d'] = (
            customer.map(active_days).fillna(0).to_numpy(dtype='int64'))
        cohort['invoice_count_previous_90d'] = (
            customer.map(previous_invoice_count).fillna(0).to_numpy(dtype='int64'))
        if not (cohort.monetary_value_30d.le(cohort.monetary_value_60d + 1e-9)
                & cohort.monetary_value_60d.le(cohort.monetary_value_90d + 1e-9)).all():
            raise AssertionError('Nested monetary windows are inconsistent')
        parts.append(cohort[KEYS + FEATURES])
    features = pd.concat(parts, ignore_index=True).sort_values(KEYS, kind='stable')
    if len(features) != len(snapshots) or features[KEYS + FEATURES].isna().any().any():
        raise AssertionError('Temporal feature coverage failed')
    return features.reset_index(drop=True)


def main() -> None:
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'}, low_memory=False)
    clean = pd.read_csv(CLEAN, dtype={'customer_id': 'string'},
                        usecols=['customer_id', 'invoice', 'invoice_date', 'line_amount'],
                        low_memory=False)
    schedule = pd.read_csv(SCHEDULE)
    features = build_temporal_features(snapshots, clean, schedule)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_PATH, index=False, encoding='utf-8-sig',
                    date_format='%Y-%m-%d')
    manifest = {
        'definition': 'Eligible transactions before snapshot_date; recent windows are within 180 days and lifetime amounts start at source start',
        'snapshot_sha256': sha256(SNAPSHOTS),
        'clean_sha256': sha256(CLEAN),
        'schedule_sha256': sha256(SCHEDULE),
        'features_sha256': sha256(FEATURE_PATH),
        'row_count': len(features),
        'features': FEATURES,
        'all_180d_and_lifetime_amounts_reconciled': True,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
    print(f'temporal features: {len(features)} rows, {len(FEATURES)} columns')


if __name__ == '__main__':
    main()
