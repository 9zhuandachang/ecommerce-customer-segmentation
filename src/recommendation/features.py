"""Pre-cutoff customer-product features and reproducible shared training samples."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.recommendation.data import make_context, past_window, history_counts
from src.recommendation.algorithms import scores
from src.recommendation.profiles import FEATURES as PROFILE_FEATURES, purchase_profiles

BASE = ['bought_before', 'log_count_180d', 'log_count_90d', 'log_count_30d',
        'item_recency_180d', 'item_first_seen_180d', 'mean_item_gap_180d', 'has_item_gap',
        'item_due_ratio', 'personal_order_share', 'log_popularity_30d', 'log_popularity_180d',
        'popularity_trend', 'log_product_buyers_180d', 'product_seen_180d']
PROFILE = ['profile_' + name for name in PROFILE_FEATURES]
INTERACTIONS = ['repeat_x_bought', 'recent_repeat_x_bought', 'change_x_bought',
                'concentration_x_count', 'repeat_x_count', 'recent_repeat_x_popularity']
SIGNAL = ['ease_score_scaled', 'no_ease_signal']
ALL_FEATURES = BASE + PROFILE + INTERACTIONS + SIGNAL + ['repeat_x_ease']
FEATURE_SETS = {'A': BASE, 'B': BASE + PROFILE + INTERACTIONS,
                'C': BASE + SIGNAL, 'D': ALL_FEATURES}
FEATURE_INDEX = {key: np.array([ALL_FEATURES.index(name) for name in values]) for key, values in FEATURE_SETS.items()}
GROUP_NAMES = {'A': 'A basic', 'B': 'BBasics+Image', 'C': 'C Basics + EASE', 'D': 'Dcomplete'}


def item_context(tx: pd.DataFrame, snapshots: pd.DataFrame, cutoff: pd.Timestamp, ease: dict) -> dict:
    context = make_context(tx, snapshots, cutoff, ease['catalog'])
    past = past_window(tx, cutoff)
    customers, catalog = pd.Index(context['customers']), pd.Index(context['catalog'].stock_code)
    shape = context['counts'].shape
    counts = context['counts'].toarray().astype(np.float32)
    last, first, gap = [np.full(shape, 180., dtype=np.float32) for _ in range(3)]
    has_gap = np.zeros(shape, dtype=np.float32)
    timing = past.groupby(['customer_id', 'stock_code']).invoice_date.agg(['min', 'max', 'nunique']).reset_index()
    rows, columns = customers.get_indexer(timing.customer_id), catalog.get_indexer(timing.stock_code)
    valid = (rows >= 0) & (columns >= 0)
    timing, rows, columns = timing.loc[valid], rows[valid], columns[valid]
    last[rows, columns] = (cutoff - timing['max']).dt.total_seconds().to_numpy() / 86400
    first[rows, columns] = (cutoff - timing['min']).dt.total_seconds().to_numpy() / 86400
    enough = timing['nunique'].to_numpy() > 1
    gap[rows[enough], columns[enough]] = ((timing.loc[enough, 'max'] - timing.loc[enough, 'min']).dt.total_seconds().to_numpy()
                                        / 86400 / (timing.loc[enough, 'nunique'].to_numpy() - 1))
    has_gap[rows[enough], columns[enough]] = 1.
    recent_counts = {days: history_counts(past_window(tx, cutoff, days), cutoff, context['catalog'], customers).toarray().astype(np.float32)
                     for days in (30, 90)}
    invoice_count = past[['customer_id', 'invoice']].drop_duplicates().groupby('customer_id').size()
    invoices = pd.Series(context['customers']).map(invoice_count).fillna(0).to_numpy(dtype=np.float32)
    pop180 = past.groupby('stock_code').size()
    pop30 = past_window(tx, cutoff, 30).groupby('stock_code').size()
    buyers = past.groupby('stock_code').customer_id.nunique()
    first_product = past.groupby('stock_code').invoice_date.min()
    map_vector = lambda values: pd.Series(context['catalog'].stock_code).map(values).fillna(0).to_numpy(dtype=np.float32)
    pop180, pop30, buyers = map_vector(pop180), map_vector(pop30), map_vector(buyers)
    product_dates = pd.Series(context['catalog'].stock_code).map(first_product)
    age = ((cutoff - product_dates).dt.total_seconds() / 86400).fillna(0).to_numpy(dtype=np.float32)
    raw = np.asarray(scores(ease, context), dtype=np.float32)
    maximum = np.max(np.abs(raw), axis=1)
    normalized = raw / np.maximum(maximum[:, None], 1e-12)
    profiles = purchase_profiles(tx, context['customers'], cutoff)
    context.update(counts_dense=counts, recent_counts=recent_counts, item_last=last, item_first=first,
                   item_gap=gap, has_gap=has_gap, invoice_count=invoices,
                   pop30=pop30, pop180=pop180, product_buyers=buyers, product_age=age,
                   ease_raw=raw, ease_normalized=normalized, no_ease_signal=maximum < 1e-12,
                   profiles=profiles)
    return context


def pair_features(context: dict, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    rows, columns = np.asarray(rows), np.asarray(columns)
    counts = context['counts_dense'][rows, columns]
    bought = (counts > 0).astype(np.float32)
    count_log = np.log1p(counts)
    recent30 = context['recent_counts'][30][rows, columns]
    recent90 = context['recent_counts'][90][rows, columns]
    last, gap = context['item_last'][rows, columns], context['item_gap'][rows, columns]
    pop30, pop180 = context['pop30'][columns], context['pop180'][columns]
    log_pop30 = np.log1p(pop30)
    profile = context['profiles'][PROFILE_FEATURES].to_numpy(dtype=np.float32)[rows]
    historical, recent, change, concentration = [profile[:, i] for i in range(4)]
    signal = context['ease_normalized'][rows, columns]
    return np.column_stack([
        bought, count_log, np.log1p(recent90), np.log1p(recent30), last / 180,
        context['item_first'][rows, columns] / 180, gap / 180, context['has_gap'][rows, columns],
        np.minimum(last / (gap + 1), 5), counts / np.maximum(context['invoice_count'][rows], 1),
        log_pop30, np.log1p(pop180), (pop30 + 1) / (pop180 / 6 + 1),
        np.log1p(context['product_buyers'][columns]), context['product_age'][columns] / 180,
        profile, historical * bought, recent * bought, change * bought,
        concentration * count_log, historical * count_log, recent * log_pop30,
        signal, context['no_ease_signal'][rows].astype(np.float32), historical * signal,
    ]).astype(np.float32)


def negative_sample(positive: np.ndarray, ease: np.ndarray, personal: np.ndarray,
                    seed: int, budget: int) -> np.ndarray:
    """Shared hard + uniform negatives; held-out candidates are never sampled."""
    candidates = np.arange(len(ease))
    available = np.ones(len(ease), dtype=bool)
    available[positive] = False
    budget = min(budget, int(available.sum()))
    if budget == 0:
        return np.empty(0, dtype=np.int32)
    ranked = [np.argsort(-ease, kind='stable'), np.argsort(-personal, kind='stable'), candidates]
    hard = []
    # Interleave three past-only rankings. The final uniform draw uses the remaining negatives.
    for position in range(len(candidates)) if budget // 2 else []:
        for order in ranked:
            item = order[position]
            if available[item]:
                hard.append(item)
                available[item] = False
                if len(hard) >= budget // 2:
                    break
        if len(hard) >= budget // 2:
            break
    rng = np.random.default_rng(seed)
    rest = rng.choice(candidates[available], size=budget - len(hard), replace=False)
    return np.r_[hard, rest].astype(np.int32)


def sample_training_pairs(context: dict, labels: pd.DataFrame) -> dict:
    if list(labels.customer_id) != list(context['customers']):
        raise ValueError('Training feature and label customers differ')
    catalog = pd.Index(context['catalog'].stock_code)
    pair_rows, pair_columns, targets, groups, queries = [], [], [], [], []
    unknown_only = 0
    for row, label in enumerate(labels.itertuples()):
        if not label.target_items:
            continue
        pos = catalog.get_indexer(label.target_items)
        pos = np.sort(pos[pos >= 0]).astype(np.int32)
        if not len(pos):
            unknown_only += 1
            continue
        digest = hashlib.sha256(f'42:{context["cutoff"].date()}:{label.customer_id}'.encode()).digest()
        seed = int.from_bytes(digest[:8], 'little')
        budget = min(120, max(50, 5 * len(pos)))
        neg = negative_sample(pos, context['ease_raw'][row], context['counts_dense'][row], seed, budget)
        items = np.r_[pos, neg]
        pair_rows.append(np.full(len(items), row, dtype=np.int32))
        pair_columns.append(items)
        targets.append(np.r_[np.ones(len(pos), dtype=np.uint8), np.zeros(len(neg), dtype=np.uint8)])
        groups.append(len(items))
        queries.append(dict(snapshot_date=str(context['cutoff'].date()), customer_id=str(label.customer_id),
                            positives=len(pos), negatives=len(neg), target_count=len(label.target_items)))
    rows, columns = np.concatenate(pair_rows), np.concatenate(pair_columns)
    result = dict(x=pair_features(context, rows, columns), y=np.concatenate(targets),
                groups=np.array(groups, dtype=np.int32), product_columns=columns,
                catalog_codes=context['catalog'].stock_code.tolist(),
                queries=pd.DataFrame(queries), date=context['cutoff'],
                label_end=context['cutoff'] + pd.Timedelta(days=30), unknown_only_buyers=unknown_only)
    result['sampling_weights'] = sampling_weights(result['queries'], len(result['catalog_codes']))
    return result


def sampling_weights(queries: pd.DataFrame, catalog_size: int) -> np.ndarray:
    """Exact inclusion weights: all positives/hard negatives, uniform remainder."""
    parts = []
    for row in queries.itertuples():
        hard = row.negatives // 2
        uniform = row.negatives - hard
        remainder = catalog_size - row.positives - hard
        parts.append(np.r_[np.ones(row.positives + hard),
                           np.full(uniform, remainder / uniform if uniform else 1.)])
    return np.concatenate(parts).astype(np.float32)


def mature_training(parts: list[dict], cutoff: pd.Timestamp) -> dict:
    eligible = [part for part in parts if part['label_end'] <= cutoff]
    if not eligible:
        raise ValueError('No mature item training samples')
    result = {key: np.concatenate([part[key] for part in eligible]) for key in ('x', 'y', 'groups', 'sampling_weights')}
    result.update(queries=pd.concat([part['queries'] for part in eligible], ignore_index=True),
                  latest_label_end=str(max(part['label_end'] for part in eligible)), fit_cutoff=str(cutoff.date()))
    if len(result['x']) != result['groups'].sum() or len(result['groups']) != len(result['queries']):
        raise ValueError('Item training groups do not match samples')
    return result
