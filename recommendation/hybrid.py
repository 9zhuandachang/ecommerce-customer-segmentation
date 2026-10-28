"""Exact two-route reciprocal-rank fusion and fast full-denominator metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

LEARNERS = ('itemknn', 'implicit_als', 'ease')
BASE_NAMES = {'popular': '热门商品', 'personal_repeat': '纯个人补货', 'itemknn': '纯ItemKNN',
              'implicit_als': '纯ALS', 'ease': '纯EASE'}
NAMES = dict(BASE_NAMES)
for _algorithm in LEARNERS:
    NAMES[f'fixed_{_algorithm}'] = f'固定混合＋{BASE_NAMES[_algorithm][1:]}'
    NAMES[f'adaptive_{_algorithm}'] = f'画像混合＋{BASE_NAMES[_algorithm][1:]}'


def hybrid_specs(algorithm: str, adaptive: bool) -> list[dict]:
    return [dict(id=f'{"adaptive" if adaptive else "fixed"}_{algorithm}_{a}_{b}',
                 method=f'{"adaptive" if adaptive else "fixed"}_{algorithm}', algorithm=algorithm,
                 intercept=a / 10., strength=b)
            for a in range(11) for b in ((.5, 1., 2.) if adaptive else (0.,))]


def mixing_weights(spec: dict, predictions: np.ndarray, prior: float) -> np.ndarray:
    return np.clip(spec['intercept'] + spec['strength'] * (predictions - prior), 0., 1.)


def prepare_routes(personal: np.ndarray, algorithm: np.ndarray) -> dict:
    if personal.shape != algorithm.shape or personal.shape[1] < 10:
        raise ValueError('Routes must share at least ten catalog products')
    if not np.isfinite(personal).all() or not np.isfinite(algorithm).all():
        raise ValueError('Nonfinite recommendation scores')
    rows, columns = personal.shape
    old = personal > 0
    repeat_ids = np.argsort(-personal, axis=1, kind='stable')[:, :10]
    repeat_valid = np.take_along_axis(old, repeat_ids, axis=1)
    explore_ids = np.argsort(-np.where(~old, algorithm, -1e300), axis=1, kind='stable')[:, :10]
    explore_valid = ~np.take_along_axis(old, explore_ids, axis=1)
    ids = np.concatenate([repeat_ids, explore_ids, np.broadcast_to(np.arange(10), (rows, 10))], axis=1)
    valid = np.concatenate([repeat_valid, explore_valid, np.ones((rows, 10), dtype=bool)], axis=1)
    # Invalid route entries must not suppress a later valid popularity fallback.
    for position in range(1, ids.shape[1]):
        valid[:, position] &= ~np.any((ids[:, :position] == ids[:, position, None]) & valid[:, :position], axis=1)
    reciprocal = 1 / np.arange(1, 11, dtype=float)
    repeat_score = np.zeros_like(ids, dtype=float)
    explore_score = np.zeros_like(ids, dtype=float)
    repeat_score[:, :10] = reciprocal
    explore_score[:, 10:20] = reciprocal
    return dict(ids=ids, valid=valid, repeat=repeat_score, explore=explore_score,
                catalog_size=columns)


def fuse_routes(routes: dict, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(routes['ids']),) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError('Expected one bounded mixing weight for every customer')
    values = weights[:, None] * routes['repeat'] + (1 - weights[:, None]) * routes['explore']
    values[~routes['valid']] = -np.inf
    order = np.lexsort((routes['ids'], -values), axis=1)[:, :10]
    result = np.take_along_axis(routes['ids'], order, axis=1)
    if np.any(np.diff(np.sort(result, axis=1), axis=1) == 0):
        raise AssertionError('Duplicate route product')
    return result


def metric_context(context: dict, labels: pd.DataFrame) -> dict:
    if list(context['customers']) != list(labels.customer_id):
        raise ValueError('Fast metric cohorts differ')
    catalog = pd.Index(context['catalog'].stock_code)
    truth = np.zeros((len(labels), len(catalog)), dtype=bool)
    for index, items in enumerate(labels.target_items):
        columns = catalog.get_indexer(items)
        truth[index, columns[columns >= 0]] = True
    return dict(truth=truth, count=labels.target_items.map(len).to_numpy(),
                repeat_count=labels.repeat_target_items.map(len).to_numpy(),
                new_count=labels.new_target_items.map(len).to_numpy(),
                old=context['counts'].toarray() > 0)


def fast_metrics(ranks: np.ndarray, truth: dict) -> dict:
    hits = truth['truth'][np.arange(len(ranks))[:, None], ranks]
    old = truth['old'][np.arange(len(ranks))[:, None], ranks]
    count = truth['count']
    buyers = count > 0
    discounts = 1 / np.log2(np.arange(10) + 2)
    ideal = np.r_[0., np.cumsum(discounts)][np.minimum(count, 10)]
    ndcg = np.divide(hits @ discounts, ideal, out=np.zeros(len(ranks)), where=ideal > 0)
    recall = np.divide(hits.sum(axis=1), count, out=np.zeros(len(ranks)), where=count > 0)
    repeat_count, new_count = truth['repeat_count'], truth['new_count']
    repeat_recall = np.divide((hits & old).sum(axis=1), repeat_count,
                              out=np.full(len(ranks), np.nan), where=repeat_count > 0)
    new_recall = np.divide((hits & ~old).sum(axis=1), new_count,
                           out=np.full(len(ranks), np.nan), where=new_count > 0)
    return dict(ndcg_10=float(ndcg[buyers].mean()), recall_10=float(recall[buyers].mean()),
                new_recall_10=float(np.nanmean(new_recall)), repeat_recall_10=float(np.nanmean(repeat_recall)))
