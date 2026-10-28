"""Full-catalog next-basket metrics, retaining unknown target products."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def per_customer_metrics(recommended: list[str], target: list[str], repeat: list[str],
                         new: list[str], catalog: set[str]) -> dict:
    if len(set(recommended)) != len(recommended):
        raise ValueError('Duplicate product recommendations')
    truth = set(target)
    result = dict(target_count=len(truth), repeat_target_count=len(repeat), new_target_count=len(new),
                  known_target_count=len(truth & catalog))
    for k in (5, 10):
        hits = np.array([item in truth for item in recommended[:k]], dtype=float)
        dcg = np.sum(hits / np.log2(np.arange(len(hits)) + 2))
        ideal = np.sum(1 / np.log2(np.arange(min(k, len(truth))) + 2))
        result[f'ndcg_{k}'] = float(dcg / ideal) if ideal else 0.
        result[f'recall_{k}'] = float(hits.sum() / len(truth)) if truth else 0.
        result[f'hit_{k}'] = int(hits.sum() > 0)
        result[f'precision_{k}'] = float(hits.sum() / k)
    result['repeat_recall_10'] = len(set(recommended[:10]) & set(repeat)) / len(repeat) if repeat else np.nan
    result['new_recall_10'] = len(set(recommended[:10]) & set(new)) / len(new) if new else np.nan
    return result


def evaluate(ranks: np.ndarray, context: dict, labels: pd.DataFrame, method: str,
             phase: str, config_id: str, seed: int = 42) -> tuple[dict, pd.DataFrame]:
    if list(labels.customer_id) != list(context['customers']) or len(ranks) != len(labels):
        raise ValueError('Recommendation and target cohorts differ')
    catalog = context['catalog'].stock_code.to_numpy()
    codes = set(catalog)
    all_recommended = catalog[ranks]
    records = []
    for row, label in enumerate(labels.itertuples()):
        recs = list(all_recommended[row])
        records.append(dict(snapshot_date=str(context['cutoff'].date()), customer_id=str(label.customer_id),
                            method=method, config_id=config_id, phase=phase, seed=seed,
                            recommended_items=json.dumps(recs), no_known_history=bool(context['no_known_history'][row]),
                            **per_customer_metrics(recs, label.target_items, label.repeat_target_items,
                                                   label.new_target_items, codes)))
    customers = pd.DataFrame(records)
    buyers = customers.loc[customers.target_count.gt(0)]
    if buyers.empty:
        raise ValueError('No next-order customers in evaluation month')
    summary = dict(snapshot_date=str(context['cutoff'].date()), method=method, phase=phase,
                   config_id=config_id, seed=seed, customers=len(customers), buyers=len(buyers),
                   catalog_items=len(catalog), recommended_item_coverage=len(np.unique(all_recommended)) / len(catalog),
                   all_customer_hit_10=float(customers.hit_10.mean()),
                   mean_next_basket_items=float(buyers.target_count.mean()),
                   known_target_share=float(buyers.known_target_count.sum() / buyers.target_count.sum()),
                   no_known_history_customers=int(customers.no_known_history.sum()),
                   repeat_recall_10=float(buyers.repeat_recall_10.mean()),
                   new_recall_10=float(buyers.new_recall_10.mean()))
    for metric in ['ndcg_5', 'ndcg_10', 'recall_5', 'recall_10', 'hit_5', 'hit_10', 'precision_5', 'precision_10']:
        summary[metric] = float(buyers[metric].mean())
    return summary, customers


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    means = ['ndcg_10', 'recall_10', 'hit_10', 'precision_10', 'ndcg_5', 'recall_5', 'hit_5',
             'all_customer_hit_10', 'recommended_item_coverage', 'known_target_share',
             'repeat_recall_10', 'new_recall_10']
    records = []
    for key, part in monthly.groupby(['phase', 'method', 'config_id', 'seed'], sort=False, dropna=False):
        phase, method, config, seed = key
        records.append(dict(phase=phase, method=method, config_id=config, seed=int(seed), months=len(part),
                            customers=int(part.customers.sum()), buyers=int(part.buyers.sum()),
                            **{f'mean_{metric}': float(part[metric].mean()) for metric in means}))
    return pd.DataFrame(records)


def paired_intervals(model: pd.DataFrame, baseline: pd.DataFrame,
                     resamples: int = 500, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    keys = ['snapshot_date', 'customer_id']
    columns = keys + ['target_count', 'ndcg_10', 'recall_10']
    joined = model[columns].merge(baseline[columns], on=keys, validate='one_to_one', suffixes=('_model', '_rule'))
    if len(joined) != len(model) or len(joined) != len(baseline) or not joined.target_count_model.eq(joined.target_count_rule).all():
        raise ValueError('Paired recommendation customers or targets differ')
    ids = np.sort(joined.customer_id.unique())
    code = pd.Index(ids).get_indexer(joined.customer_id)
    month_masks = [joined.snapshot_date.eq(date).to_numpy() & joined.target_count_model.gt(0).to_numpy()
                   for date in sorted(joined.snapshot_date.unique())]
    rng = np.random.default_rng(seed)
    differences = {metric: (joined[f'{metric}_model'] - joined[f'{metric}_rule']).to_numpy()
                   for metric in ('ndcg_10', 'recall_10')}
    def measure(weights):
        return {metric: float(np.mean([np.average(diff[mask], weights=weights[mask]) for mask in month_masks]))
                for metric, diff in differences.items()}
    point = measure(np.ones(len(joined)))
    records = []
    for i in range(resamples):
        counts = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        records.append(dict(resample=i + 1, **measure(counts[code])))
    draws = pd.DataFrame(records)
    intervals = {metric: dict(point=point[metric], lower_95=float(draws[metric].quantile(.025)),
                              upper_95=float(draws[metric].quantile(.975))) for metric in differences}
    return draws, dict(resamples=resamples, seed=seed, customer_clusters=len(ids), metrics=intervals,
                       scope='validation_paired_customer_resampling_after_selection')
