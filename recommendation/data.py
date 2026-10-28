"""Audited recommendation inputs, fixed catalogs, and next-purchase labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/customer_snapshot_pipeline'
CLEAN = SOURCE / 'clean_transactions.csv'
SNAPSHOTS = SOURCE / 'customer_snapshots.csv'
SCHEDULE = SOURCE / 'cutoff_schedule.csv'
KEYS = ['snapshot_date', 'customer_id']
EVENT_KEYS = ['customer_id', 'invoice', 'invoice_date', 'stock_code']


def sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def source_hashes() -> dict:
    return {key: sha256(path) for key, path in {'transactions': CLEAN, 'snapshots': SNAPSHOTS,
                                              'schedule': SCHEDULE}.items()}


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    expected = json.loads((ROOT / 'value_prediction/outputs/temporal_feature_manifest.json').read_text(encoding='utf-8'))
    hashes = source_hashes()
    for key, name in [('transactions', 'clean_sha256'), ('snapshots', 'snapshot_sha256'), ('schedule', 'schedule_sha256')]:
        if hashes[key] != expected[name]:
            raise ValueError(f'Recommendation input changed: {key}')
    tx = pd.read_csv(CLEAN, usecols=['invoice', 'stock_code', 'description', 'invoice_date', 'customer_id'],
                     dtype={'invoice': 'string', 'stock_code': 'string', 'customer_id': 'string'},
                     parse_dates=['invoice_date'])
    if tx[EVENT_KEYS].isna().any().any():
        raise ValueError('Missing recommendation transaction key')
    raw_count = len(tx)
    tx = tx.drop_duplicates(EVENT_KEYS, keep='last').sort_values('invoice_date', kind='stable').reset_index(drop=True)
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'}, parse_dates=['snapshot_date'])
    snapshots.snapshot_date = snapshots.snapshot_date.dt.normalize()
    flags = snapshots.use_for_modeling.astype(str).str.lower()
    if not flags.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Invalid snapshot eligibility')
    snapshots = snapshots.loc[flags.isin(['true', '1'])].copy()
    if snapshots.duplicated(KEYS).any():
        raise ValueError('Duplicate customer-month')
    audit = dict(source_hashes=hashes, source_lines=raw_count, distinct_invoice_item_events=len(tx),
                 collapsed_duplicate_invoice_item_lines=raw_count - len(tx), eligible_snapshots=len(snapshots),
                 source_start=str(tx.invoice_date.min()), source_end=str(tx.invoice_date.max()))
    return tx, snapshots, audit


def past_window(tx: pd.DataFrame, cutoff: pd.Timestamp, days: int = 180) -> pd.DataFrame:
    return tx.loc[tx.invoice_date.ge(cutoff - pd.Timedelta(days=days)) & tx.invoice_date.lt(cutoff)]


def make_training(tx: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    past = past_window(tx, cutoff)
    popularity = past.groupby('stock_code', sort=True).size().rename('purchase_invoices_180d')
    names = (past.sort_values('invoice_date', kind='stable').drop_duplicates('stock_code', keep='last')
             .set_index('stock_code').description)
    catalog = popularity.reset_index().sort_values(['purchase_invoices_180d', 'stock_code'],
                                                  ascending=[False, True], kind='stable').reset_index(drop=True)
    catalog['description'] = catalog.stock_code.map(names).fillna(catalog.stock_code)
    catalog['catalog_cutoff'] = str(cutoff.date())
    if len(catalog) < 10:
        raise ValueError('At least ten known products are required')
    users = sorted(past.customer_id.unique())
    counts = history_counts(tx, cutoff, catalog, users)
    x = counts.copy()
    x.data = np.log1p(x.data)
    pop = {}
    for days in (30, 90, 180):
        values = past_window(tx, cutoff, days).groupby('stock_code').size()
        pop[days] = catalog.stock_code.map(values).fillna(0).to_numpy(dtype=float)
    return dict(cutoff=cutoff, catalog=catalog, counts=counts, x=x, popularity=pop,
                fit_customers=len(users), fit_events=len(past),
                latest_transaction=str(past.invoice_date.max()), _gram=None, _support=None)


def history_counts(tx: pd.DataFrame, cutoff: pd.Timestamp, catalog: pd.DataFrame,
                   customers: list | np.ndarray, half_life: int | None = None) -> sparse.csr_matrix:
    past = past_window(tx, cutoff)
    rows = pd.Index(customers).get_indexer(past.customer_id)
    cols = pd.Index(catalog.stock_code).get_indexer(past.stock_code)
    valid = (rows >= 0) & (cols >= 0)
    weights = np.ones(valid.sum(), dtype=float)
    if half_life is not None:
        if half_life <= 0:
            raise ValueError('Half-life must be positive')
        age = (cutoff - past.loc[valid, 'invoice_date']).dt.total_seconds().to_numpy() / 86400
        weights = np.exp2(-age / half_life)
    return sparse.coo_matrix((weights, (rows[valid], cols[valid])),
                             shape=(len(customers), len(catalog))).tocsr()


def make_context(tx: pd.DataFrame, snapshots: pd.DataFrame, cutoff: pd.Timestamp, catalog: pd.DataFrame) -> dict:
    customers = snapshots.loc[snapshots.snapshot_date.eq(cutoff), 'customer_id'].sort_values().to_numpy()
    if not len(customers) or len(set(customers)) != len(customers):
        raise ValueError('Expected one unique customer cohort')
    counts = history_counts(tx, cutoff, catalog, customers)
    x = counts.copy()
    x.data = np.log1p(x.data)
    return dict(cutoff=cutoff, customers=customers, catalog=catalog, counts=counts, x=x,
                personal={None: counts, 30: history_counts(tx, cutoff, catalog, customers, 30),
                          90: history_counts(tx, cutoff, catalog, customers, 90)},
                no_known_history=np.diff(counts.indptr) == 0)


def next_basket_labels(tx: pd.DataFrame, snapshots: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    cohort = snapshots.loc[snapshots.snapshot_date.eq(cutoff)].sort_values('customer_id')
    future = tx.loc[tx.invoice_date.ge(cutoff) & tx.invoice_date.lt(cutoff + pd.Timedelta(days=30))]
    first = future.groupby('customer_id').invoice_date.min()
    first_lines = future.loc[future.invoice_date.eq(future.customer_id.map(first))]
    targets = first_lines.groupby('customer_id').stock_code.agg(lambda x: sorted(set(x)))
    history = past_window(tx, cutoff).groupby('customer_id').stock_code.agg(set)
    records = []
    for row in cohort.itertuples():
        answer = targets.get(row.customer_id, [])
        previously = history.get(row.customer_id, set())
        if bool(answer) != bool(row.repurchase_30d):
            raise ValueError('Next-purchase availability disagrees with audited repurchase label')
        records.append(dict(snapshot_date=cutoff, customer_id=row.customer_id, split=row.split,
                            next_purchase_datetime=first.get(row.customer_id, pd.NaT),
                            target_items=answer, repeat_target_items=[item for item in answer if item in previously],
                            new_target_items=[item for item in answer if item not in previously]))
    return pd.DataFrame(records)


def labels_for_export(labels: pd.DataFrame) -> pd.DataFrame:
    out = labels.copy()
    for column in ('target_items', 'repeat_target_items', 'new_target_items'):
        out[column] = out[column].map(lambda x: json.dumps(x, ensure_ascii=False))
    out['target_count'] = labels.target_items.map(len)
    return out
