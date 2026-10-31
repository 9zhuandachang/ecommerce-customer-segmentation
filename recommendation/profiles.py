"""Auditable purchase-habit features and strictly forward repeat-share regression."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from recommendation.data import past_window

FEATURES = ['historical_repeat_share', 'recent_repeat_share', 'repeat_share_change',
            'top3_concentration', 'order_count', 'comparable_order_count',
            'distinct_products', 'mean_basket_size', 'recency_days', 'history_span_days']
FAMILIES = ('history', 'ridge', 'lightgbm')
PROPENSITY_NAMES = {'history': 'Historical repeat-product share', 'ridge': 'Ridge regression', 'lightgbm': 'Small LightGBM'}


def purchase_profiles(tx: pd.DataFrame, customers, cutoff: pd.Timestamp) -> pd.DataFrame:
    """First visible basket establishes history; same-time invoices are one event."""
    past = past_window(tx, cutoff)
    events = (past.groupby(['customer_id', 'invoice_date'], sort=True).stock_code
              .agg(lambda x: set(x)).reset_index())
    groups = {customer: part for customer, part in events.groupby('customer_id', sort=False)}
    rows = []
    for customer in customers:
        events_customer = groups.get(customer)
        seen, counts, rates, sizes = set(), Counter(), [], []
        if events_customer is not None:
            baskets = events_customer.stock_code.tolist()
            dates = events_customer.invoice_date.tolist()
            for number, basket in enumerate(baskets):
                if number:
                    rates.append(len(basket & seen) / len(basket))
                seen.update(basket)
                counts.update(basket)
                sizes.append(len(basket))
        else:
            dates = []
        comparable = len(rates)
        rows.append(dict(snapshot_date=cutoff, customer_id=str(customer),
                         historical_repeat_share=float(np.mean(rates)) if rates else np.nan,
                         recent_repeat_share=float(np.mean(rates[-2:])) if rates else np.nan,
                         repeat_share_change=(float(np.mean(rates[-2:]) - np.mean(rates[:-2]))
                                              if comparable >= 4 else np.nan),
                         top3_concentration=(sum(n for _, n in counts.most_common(3)) / sum(sizes)
                                             if sizes else np.nan),
                         order_count=len(dates), comparable_order_count=comparable,
                         distinct_products=len(seen), mean_basket_size=float(np.mean(sizes)) if sizes else 0.,
                         recency_days=(cutoff - dates[-1]).total_seconds() / 86400 if dates else 180.,
                         history_span_days=(dates[-1] - dates[0]).total_seconds() / 86400 if dates else 0.,
                         history_status=('Insufficient history' if comparable == 0 else 'Limited history' if comparable < 4 else 'Comparable'),
                         habit_group=('Insufficient history' if comparable == 0 else
                                      'More repeat purchases' if np.mean(rates) >= 2 / 3 else
                                      'More new-product purchases' if np.mean(rates) <= 1 / 3 else 'Mixed purchasing')))
    return pd.DataFrame(rows)


def attach_repeat_target(profiles: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    columns = ['snapshot_date', 'customer_id', 'target_items', 'repeat_target_items']
    result = profiles.merge(labels[columns], on=['snapshot_date', 'customer_id'], validate='one_to_one')
    result['target_count'] = result.target_items.map(len)
    result['repeat_share_target'] = [len(old) / len(all_items) if all_items else np.nan
                                     for old, all_items in zip(result.repeat_target_items, result.target_items)]
    result['label_end'] = result.snapshot_date + pd.Timedelta(days=30)
    return result.drop(columns=['target_items', 'repeat_target_items'])


def mature_examples(frame: pd.DataFrame, fit_cutoff: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[frame.label_end.le(fit_cutoff) & frame.repeat_share_target.notna()].copy()


def propensity_specs() -> list[dict]:
    specs = [dict(id='history', family='history')]
    specs += [dict(id=f'ridge_{alpha}', family='ridge', alpha=alpha) for alpha in (.1, 1., 10., 100.)]
    specs += [dict(id=f'lgbm_{leaves}_{trees}_{minimum}', family='lightgbm', leaves=leaves,
                   trees=trees, minimum=minimum)
              for leaves in (7, 15) for trees in (60, 120) for minimum in (50, 150)]
    return specs


def fit_propensity(spec: dict, frame: pd.DataFrame, fit_cutoff: pd.Timestamp) -> dict:
    train = mature_examples(frame, fit_cutoff)
    if train.empty:
        raise ValueError('No fully observed repeat-share training examples')
    estimator = None
    if spec['family'] == 'ridge':
        estimator = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                                  StandardScaler(), Ridge(alpha=spec['alpha']))
    elif spec['family'] == 'lightgbm':
        estimator = LGBMRegressor(n_estimators=spec['trees'], num_leaves=spec['leaves'],
                                  min_child_samples=spec['minimum'], learning_rate=.03,
                                  reg_lambda=5., objective='regression', random_state=42,
                                  n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
    elif spec['family'] != 'history':
        raise ValueError('Unknown propensity method')
    if estimator is not None:
        estimator.fit(train[FEATURES], train.repeat_share_target)
    return dict(spec=spec, estimator=estimator, prior=float(train.repeat_share_target.mean()),
                fit_cutoff=str(fit_cutoff.date()), fit_rows=len(train),
                latest_label_end=str(train.label_end.max()), features=FEATURES)


def predict_propensity(model: dict, frame: pd.DataFrame) -> np.ndarray:
    if model['spec']['family'] == 'history':
        result = frame.historical_repeat_share.fillna(model['prior']).to_numpy(dtype=float)
    else:
        result = np.asarray(model['estimator'].predict(frame[FEATURES]), dtype=float)
    result = np.clip(result, 0., 1.)
    result[frame.comparable_order_count.to_numpy() == 0] = model['prior']
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite repeat-share prediction')
    return result


def propensity_metrics(predictions: np.ndarray, frame: pd.DataFrame) -> dict:
    mask = frame.repeat_share_target.notna().to_numpy()
    error = predictions[mask] - frame.loc[mask, 'repeat_share_target'].to_numpy()
    if not len(error):
        raise ValueError('No buyer repeat-share targets')
    return dict(buyers=len(error), mae=float(np.mean(np.abs(error))),
                rmse=float(np.sqrt(np.mean(error ** 2))), bias=float(error.mean()))
