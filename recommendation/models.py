"""ItemKNN, implicit-feedback ALS, and EASE with the same purchase inputs."""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from threadpoolctl import threadpool_limits

FAMILIES = ('popular', 'personal_repeat', 'itemknn', 'implicit_als', 'ease')
RULES = FAMILIES[:2]
LEARNERS = FAMILIES[2:]
NAMES = {'popular': 'Popular products', 'personal_repeat': 'Personal repeat purchase', 'itemknn': 'ItemKNN',
         'implicit_als': 'Implicit ALS', 'ease': 'EASE'}


def configurations() -> list[dict]:
    rows = []
    for days in (30, 90, 180):
        rows.append(dict(id=f'popular_{days}', method='popular', days=days))
    for half in (None, 30, 90):
        rows.append(dict(id=f'personal_repeat_{half}', method='personal_repeat', half_life=half))
    for k in (20, 50, 100, 200):
        for shrink in (0, 20):
            rows.append(dict(id=f'itemknn_{k}_{shrink}', method='itemknn', neighbors=k, shrink=shrink))
    for factors in (16, 32):
        for regularization in (.1, 1.):
            for alpha in (1., 10.):
                rows.append(dict(id=f'als_{factors}_{regularization}_{alpha}', method='implicit_als',
                                 factors=factors, regularization=regularization, alpha=alpha, iterations=15))
    for strength in (10, 30, 100, 300, 1000, 3000, 10000, 30000):
        rows.append(dict(id=f'ease_{strength}', method='ease', regularization=strength))
    return rows


def implicit_step(x: sparse.csr_matrix, fixed: np.ndarray, regularization: float, alpha: float) -> np.ndarray:
    """Exact Hu et al. normal equations; missing interactions have confidence one."""
    x = x.tocsr()
    base = fixed.T @ fixed + regularization * np.eye(fixed.shape[1])
    result = np.zeros((x.shape[0], fixed.shape[1]), dtype=float)
    for row in range(x.shape[0]):
        start, end = x.indptr[row:row + 2]
        if start == end:
            continue
        items = fixed[x.indices[start:end]]
        extra_confidence = alpha * x.data[start:end]
        a = base + (items.T * extra_confidence) @ items
        b = items.T @ (1 + extra_confidence)
        result[row] = linalg.solve(a, b, assume_a='pos', check_finite=False)
    return result


def implicit_loss(x: sparse.csr_matrix, users: np.ndarray, items: np.ndarray,
                  regularization: float, alpha: float) -> float:
    observed = x.tocoo()
    predictions = np.einsum('ij,ij->i', users[observed.row], items[observed.col])
    confidence = 1 + alpha * observed.data
    all_zero_loss = np.sum((users.T @ users) * (items.T @ items))
    return float(all_zero_loss + np.sum(confidence * (1 - predictions) ** 2 - predictions ** 2)
                 + regularization * (np.sum(users ** 2) + np.sum(items ** 2)))


def ease_weights(gram: np.ndarray, regularization: float) -> np.ndarray:
    if regularization <= 0:
        raise ValueError('EASE requires positive regularization')
    g = np.array(gram, dtype=float, copy=True)
    g.flat[::len(g) + 1] += regularization
    precision = linalg.solve(g, np.eye(len(g)), assume_a='pos', check_finite=False)
    weights = -precision / np.diag(precision)[None, :]
    np.fill_diagonal(weights, 0)
    return weights.astype(np.float32)


def fit_model(spec: dict, training: dict, seed: int = 42) -> dict:
    method, x = spec['method'], training['x']
    model = dict(spec=spec, method=method, catalog=training['catalog'],
                 fit_cutoff=training['cutoff'], popularity=training['popularity'][180], seed=seed)
    if method == 'popular':
        model['scores'] = training['popularity'][spec['days']]
    elif method == 'personal_repeat':
        pass
    elif method in ('itemknn', 'ease'):
        if training['_gram'] is None:
            training['_gram'] = (x.T @ x).toarray()
        gram = training['_gram']
        if method == 'ease':
            with threadpool_limits(limits=4, user_api='blas'):
                model['weights'] = ease_weights(gram, spec['regularization'])
        else:
            norm = np.sqrt(np.diag(gram))
            weights = gram / np.maximum(norm[:, None] * norm[None, :], 1e-12)
            if spec['shrink']:
                if training['_support'] is None:
                    binary = x.copy()
                    binary.data = np.ones_like(binary.data)
                    training['_support'] = (binary.T @ binary).toarray()
                support = training['_support']
                weights *= support / (support + spec['shrink'])
            np.fill_diagonal(weights, 0)
            neighbors = min(spec['neighbors'], len(weights) - 1)
            ids = np.argsort(-weights, axis=1, kind='stable')[:, :neighbors]
            values = np.take_along_axis(weights, ids, axis=1)
            rows = np.repeat(np.arange(len(weights)), neighbors)
            matrix = sparse.csr_matrix((values.ravel(), (rows, ids.ravel())), shape=weights.shape)
            matrix.eliminate_zeros()
            model['weights'] = matrix.astype(np.float32)
    elif method == 'implicit_als':
        rng = np.random.default_rng(seed)
        factors = rng.normal(0, .01, (x.shape[1], spec['factors']))
        transposed = x.T.tocsr()
        with threadpool_limits(limits=1, user_api='blas'):
            for _ in range(spec['iterations']):
                users = implicit_step(x, factors, spec['regularization'], spec['alpha'])
                factors = implicit_step(transposed, users, spec['regularization'], spec['alpha'])
        model['item_factors'] = factors
    else:
        raise ValueError('Unknown recommendation algorithm')
    return model


def scores(model: dict, context: dict) -> np.ndarray:
    if not model['catalog'].stock_code.equals(context['catalog'].stock_code):
        raise ValueError('Inconsistent candidate catalog')
    method, spec = model['method'], model['spec']
    if method == 'popular':
        result = np.broadcast_to(model['scores'], (len(context['customers']), len(model['catalog'])))
    elif method == 'personal_repeat':
        result = context['personal'][spec['half_life']].toarray()
    elif method == 'implicit_als':
        with threadpool_limits(limits=1, user_api='blas'):
            users = implicit_step(context['x'], model['item_factors'], spec['regularization'], spec['alpha'])
            result = users @ model['item_factors'].T
    else:
        with threadpool_limits(limits=4, user_api='blas'):
            result = context['x'] @ model['weights']
            if sparse.issparse(result):
                result = result.toarray()
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite recommendation scores')
    return np.asarray(result)


def topk(score: np.ndarray, k: int = 10) -> np.ndarray:
    """Catalog order is global popularity then stock code, giving common tie breaks."""
    if score.ndim != 2 or k < 1 or not np.isfinite(score).all():
        raise ValueError('Invalid ranking matrix')
    return np.argsort(-score, axis=1, kind='stable')[:, :min(k, score.shape[1])]


def recommend(model: dict, context: dict, k: int = 10) -> np.ndarray:
    return topk(scores(model, context), k)


def recommendation_table(model: dict, context: dict, ranks: np.ndarray) -> 'pd.DataFrame':
    import pandas as pd
    catalog = model['catalog']
    records = []
    no_signal = context['no_known_history'].copy()
    if model['method'] in ('itemknn', 'ease'):
        contributions = context['x'] @ model['weights']
        if sparse.issparse(contributions):
            no_signal |= np.diff(contributions.tocsr().indptr) == 0
        else:
            no_signal |= np.all(contributions == 0, axis=1)
    for row, customer in enumerate(context['customers']):
        prior = set(context['counts'].indices[context['counts'].indptr[row]:context['counts'].indptr[row + 1]])
        for rank, item in enumerate(ranks[row], 1):
            product = catalog.iloc[item]
            bought = int(item) in prior
            reason = ('No usable personalized signal; recommend popular products' if no_signal[row]
                      else 'Purchased in the past 180 days; candidate for repeat purchase' if bought
                      else 'Recommended by popularity during training period' if model['method'] == 'popular'
                      else 'Fill remaining slots with popular products when previously purchased items are insufficient' if model['method'] == 'personal_repeat'
                      else 'Recommend based on similarity of purchased products' if model['method'] == 'itemknn'
                      else 'Matching based on customer and product preference vectors' if model['method'] == 'implicit_als'
                      else 'Recommendations based on linear correlation of past purchase combinations')
            records.append(dict(snapshot_date=str(context['cutoff'].date()), customer_id=str(customer),
                                rank=rank, stock_code=str(product.stock_code), description=str(product.description),
                                purchased_in_past_180d=bought, recommendation_method=model['method'], reason=reason))
    return pd.DataFrame(records)
