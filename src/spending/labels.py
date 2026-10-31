"""Audited spend_30d labels on the existing customer-month snapshot cohort.

Run ``python -m src.spending.labels`` from the project root.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from src.paths import ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

SOURCE = ROOT / 'data/processed/customer_snapshot_pipeline'
SNAPSHOTS = SOURCE / 'customer_snapshots.csv'
CLEAN = SOURCE / 'clean_transactions.csv'
SCHEDULE = SOURCE / 'cutoff_schedule.csv'
OUTPUT = ROOT / 'data/processed/spending'
KEYS = ['snapshot_date', 'customer_id']


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_labels(snapshots: pd.DataFrame, clean: pd.DataFrame,
                 schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum positive eligible line amounts in each [cutoff, cutoff+30d) window."""
    snapshots = snapshots.copy()
    clean = clean.copy()
    schedule = schedule.copy()
    snapshots['snapshot_date'] = pd.to_datetime(snapshots['snapshot_date'], errors='raise')
    clean['invoice_date'] = pd.to_datetime(clean['invoice_date'], errors='raise')
    schedule['snapshot_date'] = pd.to_datetime(schedule['snapshot_date'], errors='raise')
    schedule['label_end_exclusive'] = pd.to_datetime(
        schedule['label_end_exclusive'], errors='raise')
    snapshots['customer_id'] = snapshots['customer_id'].astype('string')
    clean['customer_id'] = clean['customer_id'].astype('string')
    if snapshots.duplicated(KEYS).any() or schedule.snapshot_date.duplicated().any():
        raise ValueError('Duplicate snapshot or cutoff key')
    if set(snapshots.snapshot_date) != set(schedule.snapshot_date):
        raise ValueError('Snapshot dates and cutoff schedule differ')
    if (schedule.label_end_exclusive - schedule.snapshot_date
            != pd.Timedelta(days=30)).any():
        raise ValueError('Spend label requires the existing 30-day window')
    amounts = pd.to_numeric(clean['line_amount'], errors='raise').to_numpy(dtype=float)
    cents = np.rint(amounts * 100)
    if (not np.isfinite(amounts).all() or (amounts <= 0).any()
            or not np.allclose(amounts * 100, cents, rtol=0, atol=1e-6)):
        raise ValueError('Eligible line amounts must be positive exact cents')
    clean['line_cents'] = cents.astype('int64')
    label_parts = []
    audits = []
    for row in schedule.sort_values('snapshot_date').itertuples(index=False):
        cutoff = row.snapshot_date
        end = row.label_end_exclusive
        cohort = snapshots.loc[snapshots.snapshot_date.eq(cutoff),
                               KEYS + ['split', 'use_for_modeling', 'repurchase_30d']].copy()
        future = clean.loc[clean.invoice_date.ge(cutoff) & clean.invoice_date.lt(end),
                           ['customer_id', 'line_cents']]
        by_customer = future.groupby('customer_id', observed=True).line_cents.sum()
        cohort['spend_30d_cents'] = cohort.customer_id.map(by_customer).fillna(0).astype('int64')
        if not (cohort.repurchase_30d.to_numpy(dtype=int)
                == cohort.spend_30d_cents.gt(0).to_numpy(dtype=int)).all():
            raise ValueError(f'Spend/purchase label mismatch at {cutoff.date()}')
        cohort_total = int(future.loc[future.customer_id.isin(cohort.customer_id),
                                      'line_cents'].sum())
        if int(cohort.spend_30d_cents.sum()) != cohort_total:
            raise AssertionError(f'Spend reconciliation failed at {cutoff.date()}')
        audits.append({
            'snapshot_date': cutoff, 'label_end_exclusive': end,
            'cohort_snapshots': len(cohort),
            'positive_snapshots': int(cohort.spend_30d_cents.gt(0).sum()),
            'cohort_future_lines': int(future.customer_id.isin(cohort.customer_id).sum()),
            'cohort_future_spend': cohort_total / 100,
            'all_future_spend': int(future.line_cents.sum()) / 100,
            'reconciled': True,
        })
        label_parts.append(cohort)
    labels = pd.concat(label_parts, ignore_index=True).sort_values(KEYS, kind='stable')
    labels['spend_30d'] = labels.spend_30d_cents / 100
    labels = labels.drop(columns='spend_30d_cents').reset_index(drop=True)
    audit = pd.DataFrame(audits)
    if len(labels) != len(snapshots) or labels[KEYS].isna().any().any():
        raise AssertionError('Snapshot label coverage failed')
    return labels, audit


def main() -> None:
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'},
                            low_memory=False)
    clean = pd.read_csv(CLEAN, dtype={'customer_id': 'string'},
                        usecols=['customer_id', 'invoice_date', 'line_amount'],
                        low_memory=False)
    schedule = pd.read_csv(SCHEDULE)
    labels, audit = build_labels(snapshots, clean, schedule)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUTPUT / 'spend_labels.csv', index=False, encoding='utf-8-sig',
                  date_format='%Y-%m-%d')
    audit.to_csv(OUTPUT / 'spend_label_audit.csv', index=False, encoding='utf-8-sig',
                 date_format='%Y-%m-%d')
    manifest = {
        'label': 'spend_30d',
        'definition': 'Sum of eligible line_amount for existing customer in [snapshot_date, snapshot_date+30 days); zero if none',
        'amount_basis': 'gross eligible product transactions; cancellations, refunds and non-product rows were excluded upstream',
        'snapshot_sha256': sha256(SNAPSHOTS),
        'clean_sha256': sha256(CLEAN),
        'schedule_sha256': sha256(SCHEDULE),
        'labels_sha256': sha256(OUTPUT / 'spend_labels.csv'),
        'audit_sha256': sha256(OUTPUT / 'spend_label_audit.csv'),
        'snapshot_count': len(labels), 'cutoff_count': len(audit),
        'positive_count': int(labels.spend_30d.gt(0).sum()),
        'total_cohort_future_spend': round(float(labels.spend_30d.sum()), 2),
        'all_reconciled': bool(audit.reconciled.all()),
    }
    (OUTPUT / 'label_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'labels={len(labels)} positives={manifest["positive_count"]} '
          f'cutoffs={len(audit)} spend={manifest["total_cohort_future_spend"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
