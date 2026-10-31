"""Three supervised item scorers with matched parameters across four feature sets."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRanker
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from recommendation.item_features import FEATURE_INDEX, FEATURE_SETS, pair_features, GROUP_NAMES

FAMILIES = ('lr', 'classifier', 'ranker')
MODEL_NAMES = {'lr': 'logistic regression', 'classifier': 'LightGBM Classification', 'ranker': 'LightGBM sort'}
NAMES = {f'{group}_{family}': f'{GROUP_NAMES[group]} · {MODEL_NAMES[family]}'
         for group in FEATURE_SETS for family in FAMILIES}
NAMES.update(personal_repeat='Original replenishment', pure_ease='Original pure EASE', old_adaptive_ease='Previous version profile blend')


def model_specs(family: str) -> list[dict]:
    if family == 'lr':
        return [dict(id=f'lr_{c}', family=family, c=c) for c in (.01, .1, 1., 10.)]
    return [dict(id=f'{family}_{leaves}_{trees}', family=family, leaves=leaves, trees=trees)
            for leaves in (7, 15) for trees in (60, 120)]


def fit_item_model(spec: dict, group: str, training: dict) -> dict:
    columns = FEATURE_INDEX[group]
    x, y, sizes = training['x'][:, columns], training['y'], training['groups']
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    x = imputer.fit_transform(x)
    scaler = None
    family = spec['family']
    query_totals = np.add.reduceat(training['sampling_weights'], np.r_[0, np.cumsum(sizes)[:-1]])
    weights = training['sampling_weights'] / np.repeat(query_totals, sizes)
    weights *= len(weights) / len(sizes)
    if family == 'lr':
        scaler = StandardScaler()
        x = scaler.fit(x, sample_weight=weights).transform(x)
        estimator = LogisticRegression(C=spec['c'], solver='lbfgs', max_iter=600, tol=1e-4, random_state=42)
        with threadpool_limits(limits=4):
            estimator.fit(x, y, sample_weight=weights)
    else:
        params = dict(n_estimators=spec['trees'], num_leaves=spec['leaves'], learning_rate=.05,
                      min_child_samples=100, reg_lambda=5., random_state=42, n_jobs=4,
                      verbosity=-1, deterministic=True, force_col_wise=True)
        if family == 'classifier':
            estimator = LGBMClassifier(objective='binary', **params)
            estimator.fit(x, y, sample_weight=weights)
        elif family == 'ranker':
            estimator = LGBMRanker(objective='lambdarank', label_gain=[0, 1],
                                   lambdarank_truncation_level=15, lambdarank_norm=True, **params)
            estimator.fit(x, y, group=sizes)
        else:
            raise ValueError('Unknown item scoring family')
    return dict(spec=spec, group=group, features=FEATURE_SETS[group], imputer=imputer,
                scaler=scaler, estimator=estimator, fit_cutoff=training['fit_cutoff'],
                latest_label_end=training['latest_label_end'], fit_rows=len(y), fit_queries=len(sizes))


def predict_item_scores(model: dict, all_features: np.ndarray) -> np.ndarray:
    x = model['imputer'].transform(all_features[:, FEATURE_INDEX[model['group']]])
    if model['scaler'] is not None:
        x = model['scaler'].transform(x).astype(np.float64)
        with threadpool_limits(limits=4):
            result = model['estimator'].decision_function(x)
    else:
        # Raw margins for classification, ranking scores for LambdaMART: neither is a calibrated probability.
        result = model['estimator'].booster_.predict(x, raw_score=True, num_threads=4)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite item recommendation score')
    return np.round(np.asarray(result, dtype=np.float64), 12)


def score_catalog(models: dict, context: dict, batch_customers: int = 24) -> tuple[dict, dict]:
    """Exact full-catalog Top10; never samples held-out negatives or uses labels."""
    n, items = context['counts_dense'].shape
    ranks = {name: [] for name in models}
    top_scores = {name: [] for name in models}
    for start in range(0, n, batch_customers):
        stop = min(n, start + batch_customers)
        rows = np.repeat(np.arange(start, stop), items)
        columns = np.tile(np.arange(items), stop - start)
        features = pair_features(context, rows, columns)
        for name, model in models.items():
            values = predict_item_scores(model, features).reshape(stop - start, items)
            order = np.argsort(-values, axis=1, kind='stable')[:, :10]
            ranks[name].append(order)
            top_scores[name].append(np.take_along_axis(values, order, axis=1))
    return ({name: np.concatenate(parts) for name, parts in ranks.items()},
            {name: np.concatenate(parts) for name, parts in top_scores.items()})
