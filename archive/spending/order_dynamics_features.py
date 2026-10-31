"""Build pre-cutoff invoice-value dynamics for existing customer snapshots."""

from __future__ import annotations

from src.paths import ROOT

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.spending.labels import CLEAN, SCHEDULE, SNAPSHOTS, sha256


HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'archive/artifacts/spending/order_dynamics'
FEATURE_PATH = OUTPUT / 'features.csv'
KEYS = ['snapshot_date', 'customer_id']
FEATURES = ['last_three_invoice_mean_180d', 'invoice_value_cv_180d',
            'recent_three_vs_all_order_mean_180d']


def build_features(snapshots: pd.DataFrame, clean: pd.DataFrame,
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
    if (snapshots.duplicated(KEYS).any() or schedule.snapshot_date.duplicated().any()
            or set(snapshots.snapshot_date) != set(schedule.snapshot_date)
            or (schedule.snapshot_date - schedule.feature_start
                != pd.Timedelta(days=180)).any()):
        raise ValueError('Invalid snapshot or 180-day schedule')
    amounts = pd.to_numeric(clean.line_amount, errors='raise').to_numpy(dtype=float)
    cents = np.rint(amounts * 100)
    if (clean[['customer_id', 'invoice', 'invoice_date']].isna().any().any()
            or not np.isfinite(amounts).all() or (amounts <= 0).any()
            or not np.allclose(amounts * 100, cents, rtol=0, atol=1e-6)):
        raise ValueError('Invalid eligible transaction amounts or keys')
    clean['line_cents'] = cents.astype('int64')

    parts, audits = [], []
    for row in schedule.sort_values('snapshot_date').itertuples(index=False):
        date, start = row.snapshot_date, row.feature_start
        cohort = snapshots.loc[snapshots.snapshot_date.eq(date),
                               KEYS + ['monetary_value_180d',
                                       'invoice_frequency_180d']].copy()
        observed = clean.loc[clean.invoice_date.ge(start)
                             & clean.invoice_date.lt(date)]
        if cohort.empty or observed.empty:
            raise ValueError(f'Empty cutoff cohort or observed history: {date.date()}')
        invoices = (observed.groupby(['customer_id', 'invoice'], observed=True,
                                     as_index=False)
                    .agg(invoice_date=('invoice_date', 'max'),
                         invoice_cents=('line_cents', 'sum')))
        invoices = invoices.sort_values(['customer_id', 'invoice_date', 'invoice'],
                                        kind='stable')
        grouped = invoices.groupby('customer_id', observed=True)
        summed = grouped.invoice_cents.sum()
        counts = grouped.invoice.size()
        expected_cents = np.rint(cohort.monetary_value_180d.to_numpy(dtype=float)
                                 * 100).astype('int64')
        actual_cents = cohort.customer_id.map(summed).fillna(0).to_numpy(dtype='int64')
        actual_count = cohort.customer_id.map(counts).fillna(0).to_numpy(dtype='int64')
        if (not np.array_equal(expected_cents, actual_cents)
                or not np.array_equal(
                    actual_count,
                    cohort.invoice_frequency_180d.to_numpy(dtype='int64'))
                or (actual_count < 1).any()):
            raise ValueError(f'Invoice amounts or counts do not reconcile: {date.date()}')
        average = grouped.invoice_cents.mean()
        std = grouped.invoice_cents.std(ddof=0).fillna(0)
        recent_three = (invoices.groupby('customer_id', observed=True).tail(3)
                        .groupby('customer_id', observed=True).invoice_cents.mean())
        customer = cohort.customer_id
        cohort['last_three_invoice_mean_180d'] = (
            customer.map(recent_three).to_numpy(dtype=float) / 100)
        cohort['invoice_value_cv_180d'] = (
            customer.map(std / average).to_numpy(dtype=float))
        cohort['recent_three_vs_all_order_mean_180d'] = (
            customer.map(recent_three / average).to_numpy(dtype=float))
        if (not np.isfinite(cohort[FEATURES].to_numpy(dtype=float)).all()
                or (cohort.last_three_invoice_mean_180d <= 0).any()
                or (cohort.invoice_value_cv_180d < 0).any()
                or (cohort.recent_three_vs_all_order_mean_180d <= 0).any()):
            raise ValueError(f'Invalid order dynamics features: {date.date()}')
        parts.append(cohort[KEYS + FEATURES])
        audits.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                       'feature_start': start.strftime('%Y-%m-%d'),
                       'latest_input_datetime': invoices.invoice_date.max().isoformat(),
                       'cohort_rows': len(cohort),
                       'invoice_rows': len(invoices),
                       'amounts_and_counts_reconciled': True})
    features = pd.concat(parts, ignore_index=True).sort_values(KEYS, kind='stable')
    if len(features) != len(snapshots) or features.duplicated(KEYS).any():
        raise ValueError('Order dynamics feature coverage failed')
    return features.reset_index(drop=True), pd.DataFrame(audits)


def main() -> None:
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'},
                            usecols=KEYS + ['monetary_value_180d',
                                            'invoice_frequency_180d'])
    clean = pd.read_csv(CLEAN, dtype={'customer_id': 'string'},
                        usecols=['customer_id', 'invoice', 'invoice_date', 'line_amount'])
    schedule = pd.read_csv(SCHEDULE, usecols=['snapshot_date', 'feature_start'])
    features, audit = build_features(snapshots, clean, schedule)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_PATH, index=False, encoding='utf-8-sig',
                    date_format='%Y-%m-%d')
    audit_path = OUTPUT / 'cutoff_audit.csv'
    audit.to_csv(audit_path, index=False, encoding='utf-8-sig')
    manifest = {
        'status': 'pre_cutoff_invoice_features_reconciled',
        'snapshot_rows': len(features),
        'features': FEATURES,
        'snapshot_sha256': sha256(SNAPSHOTS),
        'clean_sha256': sha256(CLEAN),
        'schedule_sha256': sha256(SCHEDULE),
        'protocol_sha256': sha256(ROOT / 'archive/docs/spending/order_dynamics_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'features_sha256': sha256(FEATURE_PATH),
        'audit_sha256': sha256(audit_path),
        'all_amounts_and_counts_reconciled': True,
    }
    (OUTPUT / 'feature_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'order dynamics features: {len(features)} rows', flush=True)


if __name__ == '__main__':
    main()
