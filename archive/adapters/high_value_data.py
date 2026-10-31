"""Load the selected ranking comparison and enforce its export contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import ROOT
OUTPUT = ROOT / 'artifacts/spending/current'
PRIORITY = OUTPUT / 'customer_priority.csv'
MONTHLY = OUTPUT / 'priority_monthly.csv'
LOCK = OUTPUT / 'locked_selection.json'
COMPARISON = OUTPUT / 'comparison.csv'
MANIFEST = OUTPUT / 'manifest.json'
EXPORT_COLUMNS = {'snapshot_date', 'customer_id', 'priority_rank', 'priority_percentile',
                  'priority_band', 'baseline_rank', 'recency_days', 'monetary_value_180d', 'ranking_method'}


def file_hash(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def validate_priority(customers: pd.DataFrame, monthly: pd.DataFrame, lock: dict) -> None:
    if (set(customers.columns) != EXPORT_COLUMNS or customers.duplicated(['snapshot_date', 'customer_id']).any()
            or customers.isna().any().any() or monthly.snapshot_date.duplicated().any()
            or set(customers.snapshot_date) != set(monthly.snapshot_date)
            or not customers.ranking_method.eq(lock['selected_method']).all()
            or not monthly.method.eq(lock['selected_method']).all()
            or not monthly.baseline_method.eq(lock['best_rule']).all()):
        raise ValueError('Ranking cohort, method, or export columns do not match the locked selection')
    numeric = customers[['priority_rank', 'priority_percentile', 'baseline_rank',
                         'recency_days', 'monetary_value_180d']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError('Invalid ranking values')
    for date, part in customers.groupby('snapshot_date', sort=True):
        n, quota = len(part), int(np.ceil(.2 * len(part)))
        metric = monthly.loc[monthly.snapshot_date.eq(date)].iloc[0]
        expected = np.arange(1, n + 1)
        ranked = part.sort_values('priority_rank')
        if (not np.array_equal(ranked.priority_rank, expected)
                or not np.array_equal(np.sort(part.baseline_rank), expected)
                or not np.allclose(ranked.priority_percentile, 100 * (n - expected + 1) / n)
                or not np.array_equal(ranked.priority_band, np.where(expected <= quota, 'Top 20%', 'Other'))
                or metric.customers != n or metric.top20_count != quota):
            raise ValueError('Invalid ranking positions, percentiles, or monthly quota')


def load_high_value_comparison() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    if manifest.get('status') != 'completed_internal_ranking_comparison':
        raise ValueError('Ranking comparison is incomplete')
    for name, expected in manifest['outputs_sha256'].items():
        path = (OUTPUT / name).resolve()
        if path.parent != OUTPUT.resolve() or file_hash(path) != expected:
            raise ValueError(f'Ranking comparison output changed: {name}')
    if (lock['model_sha256'] != manifest['outputs_sha256']['ranking_bundle.joblib']
            or lock['source_hashes'] != manifest['source_hashes']
            or lock['selected_method'] != manifest['selected_method']
            or lock['best_rule'] != manifest['baseline_method']
            or lock['review_labels_used_for_training_or_selection']
            or lock['review_model_updates']):
        raise ValueError('Ranking selection or training boundary changed')
    sources = {'snapshots': ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv',
               'temporal_features': ROOT / 'data/processed/spending/temporal_features.csv',
               'labels': ROOT / 'data/processed/spending/spend_labels.csv'}
    for name, path in sources.items():
        if file_hash(path) != lock['source_hashes'][name]:
            raise ValueError(f'Ranking source changed: {name}')
    customers = pd.read_csv(PRIORITY, dtype={'customer_id': 'string'})
    monthly = pd.read_csv(MONTHLY)
    comparison = pd.read_csv(COMPARISON)
    validate_priority(customers, monthly, lock)
    return customers, monthly, lock, comparison
