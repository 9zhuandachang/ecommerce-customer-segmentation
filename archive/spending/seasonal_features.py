"""Audited same-period-prior-year spend, with explicit source-coverage missingness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.spending.labels import CLEAN, OUTPUT, SNAPSHOTS, sha256

FEATURES = ['previous_year_30d_spend']
FEATURE_PATH = OUTPUT / 'seasonal_features.csv'
MANIFEST_PATH = OUTPUT / 'seasonal_feature_manifest.json'


def build_seasonal_features(snapshots: pd.DataFrame,
                            clean: pd.DataFrame) -> pd.DataFrame:
    snapshots = snapshots.copy()
    clean = clean.copy()
    snapshots['snapshot_date'] = pd.to_datetime(snapshots.snapshot_date, errors='raise')
    clean['invoice_date'] = pd.to_datetime(clean.invoice_date, errors='raise')
    snapshots['customer_id'] = snapshots.customer_id.astype('string')
    clean['customer_id'] = clean.customer_id.astype('string')
    if snapshots.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate seasonal snapshot key')
    amount = pd.to_numeric(clean.line_amount, errors='raise').to_numpy(dtype=float)
    cents = np.rint(amount * 100)
    if (not np.isfinite(amount).all() or (amount <= 0).any()
            or not np.allclose(amount * 100, cents, rtol=0, atol=1e-6)):
        raise ValueError('Invalid eligible transaction amounts')
    clean['line_cents'] = cents.astype('int64')
    source_start = clean.invoice_date.min()
    parts = []
    for date, cohort in snapshots.groupby('snapshot_date', sort=True):
        window_start = date - pd.Timedelta(days=365)
        window_end = window_start + pd.Timedelta(days=30)
        if window_end >= date:
            raise AssertionError('Prior-year feature overlaps prediction date')
        part = cohort[['snapshot_date', 'customer_id']].copy()
        if window_start < source_start:
            part['previous_year_30d_spend'] = np.nan
        else:
            observed = clean.loc[clean.invoice_date.ge(window_start)
                                 & clean.invoice_date.lt(window_end)]
            by_customer = observed.groupby('customer_id', observed=True).line_cents.sum()
            part['previous_year_30d_spend'] = (
                part.customer_id.map(by_customer).fillna(0).to_numpy(dtype=float) / 100)
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if (len(result) != len(snapshots)
            or result.duplicated(['snapshot_date', 'customer_id']).any()):
        raise AssertionError('Seasonal feature coverage failed')
    return result.sort_values(['snapshot_date', 'customer_id'],
                              kind='stable').reset_index(drop=True)


def main() -> None:
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'},
                            usecols=['snapshot_date', 'customer_id'])
    clean = pd.read_csv(CLEAN, dtype={'customer_id': 'string'},
                        usecols=['customer_id', 'invoice_date', 'line_amount'])
    result = build_seasonal_features(snapshots, clean)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(FEATURE_PATH, index=False, encoding='utf-8-sig',
                  date_format='%Y-%m-%d')
    manifest = {
        'definition': 'Spend in [cutoff-365d, cutoff-335d); null where source coverage is incomplete',
        'snapshot_sha256': sha256(SNAPSHOTS),
        'clean_sha256': sha256(CLEAN),
        'features_sha256': sha256(FEATURE_PATH),
        'row_count': len(result),
        'source_incomplete_rows': int(result.previous_year_30d_spend.isna().sum()),
        'features': FEATURES,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
    print(f'seasonal features: {len(result)} rows; '
          f'{manifest["source_incomplete_rows"]} unavailable histories', flush=True)


if __name__ == '__main__':
    main()
