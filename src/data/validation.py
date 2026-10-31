"""Shared customer-snapshot schema and chronological validation."""

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    'recency_days', 'invoice_frequency_180d', 'monetary_value_180d',
    'average_order_value_180d', 'product_diversity_180d',
    'invoice_count_30d', 'invoice_count_90d', 'customer_history_days',
]
RFM = ['recency_days', 'invoice_frequency_180d', 'monetary_value_180d']
KEYS = ['snapshot_date', 'customer_id', 'split']
SPLITS = ['train', 'validation', 'test']
LABEL = 'repurchase_30d'


def load_snapshots(path: Path) -> tuple[pd.DataFrame, dict]:
    """Follow upstream partitioning to verify sample uniqueness, chronological order, and cross-partition label windows."""
    data = pd.read_csv(path, dtype={'customer_id': 'string'})
    required = set(KEYS + FEATURES + ['use_for_modeling', LABEL])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    original_count = len(data)
    flags = data['use_for_modeling'].astype(str).str.lower()
    if not flags.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Invalid use_for_modeling flag')
    data['snapshot_date'] = pd.to_datetime(data['snapshot_date'], errors='raise')
    if data[KEYS].isna().any().any():
        raise ValueError('Missing snapshot key / split')
    if data.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate snapshot_date + customer_id')
    data = data.loc[flags.isin(['true', '1'])].copy()
    if set(data['split']) != set(SPLITS):
        raise ValueError('Expected nonempty train, validation and test partitions')
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors='raise')
    values = data[FEATURES].to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError('Features must be nonnegative finite values or missing')
    if data[RFM].isna().any().any():
        raise ValueError('R/F/M is required for eligibility, profiles and baselines')
    if not data['recency_days'].between(0, 180).all():
        raise ValueError('Recency outside the 180-day observation window')
    if (data['invoice_frequency_180d'] < 1).any():
        raise ValueError('Eligible customers must have at least one invoice')
    # Labels are only used as a result of training/validation ranking evaluation and do not enter clustering or preprocessing.
    if not data[LABEL].isin([0, 1]).all():
        raise ValueError('Expected binary repurchase_30d')
    for earlier, later in zip(SPLITS, SPLITS[1:]):
        previous = data.loc[data['split'].eq(earlier), 'snapshot_date']
        boundary = data.loc[data['split'].eq(later), 'snapshot_date'].min()
        if previous.max() >= boundary:
            raise ValueError('Temporal partitions overlap')
        if (previous + pd.Timedelta(days=30) > boundary).any():
            raise ValueError('Label window crosses the next partition; fix upstream eligibility')
    for column in ['last_purchase_datetime', 'first_purchase_datetime']:
        if column in data:
            purchase = pd.to_datetime(data[column], errors='raise')
            if purchase.isna().any() or (purchase >= data['snapshot_date']).any():
                raise ValueError(f'{column} must be strictly before snapshot_date')
    data = data.sort_values(KEYS[:2], kind='stable').reset_index(drop=True)
    audit = {
        'source_rows': original_count, 'eligible_rows': len(data),
        'excluded_by_use_for_modeling': original_count - len(data),
        'missing_features': data[FEATURES].isna().sum().to_dict(),
        'unique_snapshot_keys': True, 'temporal_order_and_label_boundary_check': True,
        'upstream_transaction_and_complete_future_window_reaudit':
            'Unavailable from snapshots alone; requires stage 2/3 transaction artifacts.',
    }
    return data, audit
