"""Audited pre-cutoff transaction-composition features for spend prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.labels import CLEAN, SCHEDULE, SNAPSHOTS, sha256

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'outputs/composition'
FEATURE_PATH = OUTPUT / 'composition_features.csv'
MANIFEST_PATH = OUTPUT / 'manifest.json'
PROTOCOL = HERE / 'composition_protocol.md'
KEYS = ['snapshot_date', 'customer_id']
FEATURES = ['bulk_spend_share_180d', 'median_unit_price_180d', 'non_uk_180d']


def build_composition_features(snapshots: pd.DataFrame, clean: pd.DataFrame,
                               schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = snapshots.copy()
    clean = clean.copy()
    schedule = schedule.copy()
    snapshots['snapshot_date'] = pd.to_datetime(snapshots.snapshot_date, errors='raise')
    clean['invoice_date'] = pd.to_datetime(clean.invoice_date, errors='raise')
    schedule['snapshot_date'] = pd.to_datetime(schedule.snapshot_date, errors='raise')
    schedule['feature_start'] = pd.to_datetime(schedule.feature_start, errors='raise')
    snapshots['customer_id'] = snapshots.customer_id.astype('string')
    clean['customer_id'] = clean.customer_id.astype('string')
    clean['country'] = clean.country.astype('string')
    if (snapshots.duplicated(KEYS).any() or schedule.snapshot_date.duplicated().any()
            or set(snapshots.snapshot_date) != set(schedule.snapshot_date)
            or (schedule.snapshot_date - schedule.feature_start
                != pd.Timedelta(days=180)).any()):
        raise ValueError('Invalid snapshot or 180-day cutoff schedule')
    if (clean[['invoice_date', 'customer_id', 'quantity', 'unit_price',
               'country', 'line_amount']].isna().any().any()
            or (clean[['quantity', 'unit_price', 'line_amount']] <= 0).any().any()):
        raise ValueError('Composition source has missing or nonpositive transaction fields')
    amounts = clean.line_amount.to_numpy(dtype=float)
    cents = np.rint(amounts * 100)
    if (not np.isfinite(amounts).all()
            or not np.allclose(amounts * 100, cents, atol=1e-6, rtol=0)):
        raise ValueError('Transaction amounts are not exact cents')
    clean['line_cents'] = cents.astype('int64')
    clean['non_uk'] = clean.country.ne('United Kingdom')
    rows, audits = [], []
    for cutoff in schedule.sort_values('snapshot_date').itertuples(index=False):
        date, start = cutoff.snapshot_date, cutoff.feature_start
        cohort = snapshots.loc[snapshots.snapshot_date.eq(date),
                               KEYS + ['monetary_value_180d']].copy()
        observed = clean.loc[clean.invoice_date.ge(start)
                             & clean.invoice_date.lt(date)].copy()
        if cohort.empty or observed.empty:
            raise ValueError(f'Empty cutoff cohort or history: {date.date()}')
        total = observed.groupby('customer_id', observed=True).line_cents.sum()
        actual_cents = cohort.customer_id.map(total).fillna(0).to_numpy(dtype='int64')
        expected_cents = np.rint(cohort.monetary_value_180d.to_numpy(dtype=float)
                                 * 100).astype('int64')
        if not np.array_equal(actual_cents, expected_cents) or (actual_cents <= 0).any():
            raise ValueError(f'Composition transaction amounts do not reconcile: {date.date()}')
        bulk = (observed.loc[observed.quantity.ge(12)]
                .groupby('customer_id', observed=True).line_cents.sum())
        median_price = observed.groupby('customer_id', observed=True).unit_price.median()
        non_uk = observed.groupby('customer_id', observed=True).non_uk.any()
        cohort['bulk_spend_share_180d'] = (
            cohort.customer_id.map(bulk).fillna(0).to_numpy(dtype=float)
            / actual_cents)
        cohort['median_unit_price_180d'] = cohort.customer_id.map(median_price)
        cohort['non_uk_180d'] = cohort.customer_id.map(non_uk)
        if (cohort[FEATURES].isna().any().any()
                or not cohort.bulk_spend_share_180d.between(0, 1).all()
                or (cohort.median_unit_price_180d <= 0).any()):
            raise ValueError(f'Invalid composition feature values: {date.date()}')
        cohort['non_uk_180d'] = cohort.non_uk_180d.astype('int8')
        rows.append(cohort[KEYS + FEATURES])
        audits.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                       'cohort_rows': len(cohort),
                       'observed_transaction_rows': len(observed),
                       'amounts_reconciled': True,
                       'latest_input_datetime': observed.invoice_date.max().isoformat(),
                       'feature_start': start.strftime('%Y-%m-%d')})
    result = pd.concat(rows, ignore_index=True).sort_values(KEYS,
                                                             kind='stable').reset_index(drop=True)
    if (len(result) != len(snapshots) or result.duplicated(KEYS).any()
            or not (result.non_uk_180d.isin([0, 1])).all()):
        raise ValueError('Composition feature coverage failed')
    return result, pd.DataFrame(audits)


def main() -> None:
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'},
                            usecols=KEYS + ['monetary_value_180d'])
    clean = pd.read_csv(CLEAN, dtype={'customer_id': 'string', 'country': 'string'},
                        usecols=['customer_id', 'invoice_date', 'quantity',
                                 'unit_price', 'country', 'line_amount'])
    schedule = pd.read_csv(SCHEDULE, usecols=['snapshot_date', 'feature_start'])
    result, audit = build_composition_features(snapshots, clean, schedule)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(FEATURE_PATH, index=False, encoding='utf-8-sig',
                  date_format='%Y-%m-%d')
    audit_path = OUTPUT / 'cutoff_audit.csv'
    audit.to_csv(audit_path, index=False, encoding='utf-8-sig')
    manifest = {
        'status': 'pre_cutoff_features_all_amounts_reconciled',
        'definition': FEATURES, 'snapshot_rows': len(result),
        'source_snapshot_sha256': sha256(SNAPSHOTS),
        'source_clean_sha256': sha256(CLEAN),
        'source_schedule_sha256': sha256(SCHEDULE),
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'features_sha256': sha256(FEATURE_PATH),
        'cutoff_audit_sha256': sha256(audit_path),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
    print(f'wrote {FEATURE_PATH}', flush=True)


if __name__ == '__main__':
    main()
