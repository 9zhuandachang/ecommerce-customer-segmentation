"""Shared spend-label joins, ranking quotas, and mature training windows."""

import numpy as np
import pandas as pd
from src.data.validation import LABEL

TARGET = "spend_30d"
TOP_FRACTION = .2


def attach_labels(snapshots: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Require a one-to-one match with the existing snapshot cohort."""
    keys = ['snapshot_date', 'customer_id']
    labels = labels.copy()
    labels['snapshot_date'] = pd.to_datetime(labels['snapshot_date'], errors='raise')
    labels['customer_id'] = labels['customer_id'].astype('string')
    if labels.duplicated(keys).any() or snapshots.duplicated(keys).any():
        raise ValueError('Duplicate customer-month key')
    merged = snapshots.merge(labels[keys + ['split', LABEL, TARGET]],
                             on=keys, how='left', validate='one_to_one',
                             suffixes=('', '_label'), indicator=True)
    if not merged._merge.eq('both').all() or len(merged) != len(snapshots):
        raise ValueError('Spend labels do not cover the modeling snapshots')
    if not merged.split.eq(merged.split_label).all() or not merged[LABEL].eq(
            merged[f'{LABEL}_label']).all():
        raise ValueError('Spend label split or repurchase label differs')
    if merged[TARGET].isna().any() or (merged[TARGET] < 0).any():
        raise ValueError('Invalid 30-day spend label')
    return merged.drop(columns=['split_label', f'{LABEL}_label', '_merge'])


def top20_mask(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    ranked = frame[['snapshot_date', 'customer_id']].copy()
    ranked['score'] = scores
    ranked['row_number'] = np.arange(len(frame))
    ranked = ranked.sort_values(['snapshot_date', 'score', 'customer_id'],
                                ascending=[True, False, True], kind='stable')
    position = ranked.groupby('snapshot_date', sort=False).cumcount() + 1
    quota = np.ceil(ranked.groupby('snapshot_date').customer_id.transform('size')
                    * TOP_FRACTION)
    mask = np.zeros(len(frame), dtype=bool)
    mask[ranked.loc[position <= quota, 'row_number'].to_numpy(dtype=int)] = True
    return mask


def fit_rows_for_date(data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """A training label is available only when its full 30-day window ended."""
    fit = data.loc[data.snapshot_date.add(pd.Timedelta(days=30)).le(date)].copy()
    if fit.empty or (fit.snapshot_date + pd.Timedelta(days=30) > date).any():
        raise ValueError(f'No leakage-safe training rows for {date.date()}')
    return fit
