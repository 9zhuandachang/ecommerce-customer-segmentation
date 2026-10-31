"""Phase 4: Training period K-means clustering, center distance and RFM sorting baseline.

Run from the project root directory: python stage 4 /kmeans.py; self-check: python stage 4 /kmeans.py --check
Output: stage 4 /outputs/stage 4 /, reports, data, figures, models and running records.
Only the training period is fitted; the test period only exports clustering, distance and ranking.
Metrics are calculated without reading test tags so that stage six retains a final evaluation.
Loading: bundle = joblib.load('stage 4 /outputs/stage 4 /model/segmentation_bundle.joblib')
Distance: bundle['pipeline'].transform(frame[bundle['features']])
Grouping: bundle['pipeline'].predict(frame[bundle['features']])
During rolling verification, the fitting must be re-run in each fold training part, and the model of this full training period cannot be reused."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import unittest
from itertools import combinations
sys.dont_write_bytecode = True
from pathlib import Path

from src.paths import ROOT as PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from src.data.transforms import signed_log1p
from src.data.validation import FEATURES, RFM, KEYS, SPLITS, LABEL, load_snapshots

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager



def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')




def make_pipeline(k: int, log_columns: list[str], seed: int) -> Pipeline:
    log_indices = [FEATURES.index(col) for col in log_columns]
    identity_indices = [i for i in range(len(FEATURES)) if i not in log_indices]
    transforms = []
    if log_indices:
        transforms.append(('log1p', FunctionTransformer(np.log1p, validate=False), log_indices))
    if identity_indices:
        transforms.append(('identity', 'passthrough', identity_indices))
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('transform', ColumnTransformer(transforms, remainder='drop')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)),
    ])


def profile(data: pd.DataFrame, labels: np.ndarray, k: int) -> pd.DataFrame:
    """The number of customers is the number of duplicates in each group; the number of snapshots is used for clustering scale, and the number of customers across months cannot be added."""
    frame = data.assign(cluster=labels)
    groups = frame.groupby('cluster', sort=True)
    result = groups.agg(
        snapshot_count=('customer_id', 'size'),
        unique_customer_count=('customer_id', 'nunique'),
    )
    result['snapshot_share'] = result['snapshot_count'] / len(frame)
    for col in FEATURES:
        result[col + '_mean'] = groups[col].mean()
        result[col + '_median'] = groups[col].median()
    # Name without repurchase tag or select k.
    result.insert(0, 'k', k)
    return result.reset_index()


def name_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """After the model is fitted and the profile is formed, a verifiable descriptive name is given based on the median ranking among the groups."""
    result = {}
    ranks = profiles.set_index('cluster')[[col + '_median' for col in RFM]].rank(pct=True)
    for cluster, row in ranks.iterrows():
        recency = ('Recently' if row.iloc[0] <= 1 / 3 else
                   'medium buying interval' if row.iloc[0] <= 2 / 3 else 'Long inactive')
        frequency = 'high frequency' if row.iloc[1] > 0.5 else 'low frequency'
        monetary = ('low consumption' if row.iloc[2] <= 1 / 3 else
                    'medium consumption' if row.iloc[2] <= 2 / 3 else 'high consumption')
        result[int(cluster)] = f'C{cluster} {recency}/{frequency}/{monetary}'
    return result


def fit_rfm(train: pd.DataFrame) -> dict:
    """Fifth gear RFM: The cut point only uses the training period quantile; the same value will always be in the same gear."""
    return {
        'cutpoints': {col: train[col].quantile([.2, .4, .6, .8]).tolist() for col in RFM},
        'missing_medians': {col: float(train[col].median()) for col in RFM},
        'tie_rule': 'searchsorted(side=left); repeated cutpoints may leave empty bins',
    }


def baseline_scores(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = data[KEYS].copy()
    for col, prefix in zip(RFM, ['r', 'f', 'm']):
        values = data[col].fillna(config['missing_medians'][col]).to_numpy()
        bins = np.searchsorted(config['cutpoints'][col], values, side='left')
        out[prefix + '_score'] = 5 - bins if prefix == 'r' else 1 + bins
    out['rfm_score'] = out[['r_score', 'f_score', 'm_score']].sum(axis=1)
    # The list is sorted within eligible customers for each date, with identical scores in ascending order by the customer_id text.
    for name in ['rfm']:
        ordered = out.sort_values(
            ['snapshot_date', name + '_score', 'customer_id'],
            ascending=[True, False, True], kind='stable',
        )
        ranks = ordered.groupby('snapshot_date', sort=False).cumcount() + 1
        out[name + '_rank'] = ranks.reindex(out.index)
        sizes = out.groupby('snapshot_date')['customer_id'].transform('size')
        out[name + '_top20'] = out[name + '_rank'] <= np.ceil(sizes * .2)
    return out


def evaluate_baselines(scores: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Only calculated in train / validation; Top 20 quotas are ceil( 0.2 *n) by date."""
    frame = scores.copy()
    frame[LABEL] = labels
    rows = []
    for split in ['train', 'validation']:
        part = frame.loc[frame['split'].eq(split)]
        views = [('ALL', part)] + [
            (date.strftime('%Y-%m-%d'), group)
            for date, group in part.groupby('snapshot_date', sort=True)
        ]
        for date, group in views:
            y = group[LABEL].to_numpy()
            prevalence = float(y.mean())
            for name in ['rfm']:
                predicted = group[name + '_score'].to_numpy()
                top = group[name + '_top20'].to_numpy()
                hits = int(y[top].sum())
                precision = hits / int(top.sum())
                rows.append({
                    'split': split, 'snapshot_date': date, 'baseline': name,
                    'snapshot_count': len(y), 'positive_count': int(y.sum()),
                    'positive_rate': prevalence,
                    'average_precision': average_precision_score(y, predicted) if y.sum() else np.nan,
                    'roc_auc': roc_auc_score(y, predicted) if np.unique(y).size == 2 else np.nan,
                    'top20_count': int(top.sum()), 'top20_hits': hits,
                    'precision_at_20pct': precision,
                    'recall_at_20pct': hits / y.sum() if y.sum() else np.nan,
                    'lift_at_20pct': precision / prevalence if prevalence else np.nan,
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=PROJECT_ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'archive/artifacts/segmentation/stage4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--silhouette-sample-size', type=int, default=4000)
    parser.add_argument('--check', action='store_true', help='Check generated results without retraining')
    args = parser.parse_args()
    if args.check:
        run_self_checks(args.input, args.output.resolve())
        return
    if args.silhouette_sample_size < 10:
        parser.error('--silhouette-sample-size must be >= 10')
    output = args.output.resolve()
    for subdirectory in ['', 'data', 'figures', 'model']:
        (output / subdirectory).mkdir(parents=True, exist_ok=True)
    data, audit = load_snapshots(args.input)
    train = data.loc[data['split'].eq('train')].copy()
    if train[FEATURES].isna().all().any():
        raise ValueError('At least one training feature is entirely missing')
    if len(train) < 7:
        raise ValueError('At least seven training rows required to compare k=3..6')
    print(f'Eligible snapshots: {len(data):,}; training snapshots: {len(train):,}', flush=True)
    skew = train[FEATURES].skew()
    log_columns = skew.index[skew.gt(1)].tolist()
    sample = np.random.default_rng(args.seed).choice(
        len(train), size=min(args.silhouette_sample_size, len(train)), replace=False)
    comparisons, all_profiles, pipelines = [], [], {}
    # Single threading makes native operations stable and avoids excessive parallelism issues with MKL/OpenMP on Windows K-means.
    with threadpool_limits(limits=1):
        for k in range(3, 7):
            pipeline = make_pipeline(k, log_columns, args.seed)
            pipeline.fit(train[FEATURES])
            model = pipeline.named_steps['kmeans']
            labels = model.labels_
            if np.unique(labels).size != k:
                raise ValueError(f'k={k} produced fewer clusters; inspect degenerate features')
            transformed = pipeline[:-1].transform(train[FEATURES])
            sampled_labels = labels[sample]
            silhouette = (silhouette_score(transformed[sample], sampled_labels)
                          if 1 < np.unique(sampled_labels).size < len(sample) else np.nan)
            candidate_profile = profile(train, labels, k)
            counts = np.bincount(labels, minlength=k)
            row = {
                'k': k, 'silhouette_train': silhouette, 'inertia_train': model.inertia_,
                'min_cluster_count': int(counts.min()), 'max_cluster_count': int(counts.max()),
                'min_cluster_share': counts.min() / len(train),
                'size_floor_passed': bool(counts.min() / len(train) >= .05),
                'n_iter': int(model.n_iter_),
            }
            comparisons.append(row)
            all_profiles.append(candidate_profile)
            pipelines[k] = pipeline
            print(f"k={k}: silhouette={silhouette:.4f}; sizes={counts.tolist()}", flush=True)
        comparison = pd.DataFrame(comparisons)
        eligible = comparison.loc[comparison['size_floor_passed'] & comparison['silhouette_train'].notna()]
        size_constraint_relaxed = eligible.empty
        if eligible.empty:
            eligible = comparison.loc[comparison['silhouette_train'].notna()]
        if eligible.empty:
            raise ValueError('No valid silhouette score; increase sample size or inspect features')
        selected = int(eligible.sort_values(['silhouette_train', 'k'], ascending=[False, True]).iloc[0]['k'])
        comparison['selected'] = comparison['k'].eq(selected)
        pipeline = pipelines[selected]
        training_profile = all_profiles[selected - 3].copy()
        names = name_clusters(training_profile)
        training_profile.insert(2, 'cluster_name', training_profile['cluster'].map(names))
        labels = pipeline.predict(data[FEATURES])
        distances = pipeline.transform(data[FEATURES])
        cluster_features = data[KEYS].copy()
        cluster_features['cluster'] = labels
        cluster_features['cluster_name'] = cluster_features['cluster'].map(names)
        distance_columns = [f'cluster_distance_{i}' for i in range(selected)]
        cluster_features[distance_columns] = distances
        rfm_config = fit_rfm(train)
        rankings = baseline_scores(data, rfm_config)
        metrics = evaluate_baselines(rankings, data[LABEL].to_numpy())
        bundle = {
            'schema_version': 1, 'pipeline': pipeline, 'features': FEATURES,
            'selected_k': selected, 'cluster_names': names,
            'distance_columns': distance_columns, 'rfm_config': rfm_config, 'seed': args.seed,
        }
        # Prevent future data from changing training statistics; individual/batch inference on the same training set should be exactly the same.
        train_mask = data['split'].eq('train').to_numpy()
        np.testing.assert_allclose(pipeline.transform(train[FEATURES]), distances[train_mask], atol=1e-12)
        np.testing.assert_array_equal(np.argmin(distances, axis=1), labels)
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise AssertionError('Invalid center distances')
        pd.testing.assert_frame_equal(rankings[KEYS], cluster_features[KEYS])

    split_profiles = []
    for split in SPLITS:
        mask = data['split'].eq(split).to_numpy()
        view = profile(data.loc[mask], labels[mask], selected)
        view.insert(0, 'split', split)
        view.insert(3, 'cluster_name', view['cluster'].map(names))
        split_profiles.append(view)
    partitions = data.groupby(['split', 'snapshot_date'], sort=True).agg(
        snapshot_count=('customer_id', 'size'), unique_customer_count=('customer_id', 'nunique'),
    ).reset_index()
    partitions['label_window_end_exclusive'] = partitions['snapshot_date'] + pd.Timedelta(days=30)
    preprocessing = pd.DataFrame({
        'feature': FEATURES, 'train_skew': skew.reindex(FEATURES).to_numpy(),
        'log1p': [c in log_columns for c in FEATURES],
        'train_imputer_median': pipeline.named_steps['imputer'].statistics_,
    })
    # ColumnTransformer puts the log column in front, and the original column order needs to be restored when recording scaling statistics.
    transformed_order = log_columns + [c for c in FEATURES if c not in log_columns]
    scaler = pipeline.named_steps['scaler']
    mean_by_feature = dict(zip(transformed_order, scaler.mean_))
    scale_by_feature = dict(zip(transformed_order, scaler.scale_))
    preprocessing['transformed_train_mean'] = preprocessing['feature'].map(mean_by_feature)
    preprocessing['transformed_train_scale'] = preprocessing['feature'].map(scale_by_feature)
    manifest = {
        'input_file': (args.input.resolve().relative_to(PROJECT_ROOT).as_posix()
                       if args.input.resolve().is_relative_to(PROJECT_ROOT)
                       else str(args.input.resolve())),
        'source_urls': [
            'https://github.com/9zhuandachang/ecommerce-customer-segmentation/blob/main/data/processed/customer_snapshot_pipeline/customer_snapshots.csv',
            'https://github.com/9zhuandachang/ecommerce-customer-segmentation/blob/main/docs/project_workflow_and_acceptance_checklist.md',
        ],
        'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'features': FEATURES, 'log1p_features': log_columns, 'log_selection': 'training skew > 1',
        'selected_k': selected, 'seed': args.seed, 'n_init': 20, 'max_iter': 500,
        'silhouette_sample_size': len(sample), 'min_cluster_share_floor': .05,
        'selection_rule': 'train silhouette descending among size-floor candidates, k ascending on ties',
        'size_constraint_relaxed': size_constraint_relaxed,
        'rfm_config': rfm_config, 'audit': audit,
        'split_summary': {s: {
            'snapshot_count': int(data['split'].eq(s).sum()),
            'date_min': str(data.loc[data['split'].eq(s), 'snapshot_date'].min().date()),
            'date_max': str(data.loc[data['split'].eq(s), 'snapshot_date'].max().date()),
        } for s in SPLITS},
        'versions': {
            'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__, 'joblib': joblib.__version__,
            'matplotlib': matplotlib.__version__,
        },
        'reload_check': 'PASS: full-data predictions identical; distances within 1e-12; RFM scores identical',
        'output_checks': 'PASS: keys aligned; distances finite/nonnegative; nearest center equals label; train-only inference agrees',
        'test_label_metrics_computed': False,
    }
    results = collect_insights(data, pipelines, selected, sample, args.seed)
    report_names = render_business_report(output, data, comparison, metrics, selected, results, manifest)
    export_results(output, data, comparison, metrics, cluster_features, rankings,
                   pd.concat(split_profiles), partitions, preprocessing, bundle,
                   results, report_names, manifest)
    print(f'\nSelected k={selected}; reload / distance / baseline checks PASS', flush=True)
    print(metrics.loc[metrics['snapshot_date'].eq('ALL')].to_string(index=False), flush=True)
    print(f'Artifacts: {output}', flush=True)



EXTRA_FEATURES = ['monthly_order_change', 'recent90_order_share',
                  'history_adjusted_monthly_orders', 'has_90d_history']
DESCRIPTIVE_FEATURES = EXTRA_FEATURES + [
    'recently_observed_30d', 'one_invoice_180d', 'inactive_90d',
    'paused_after_repeat', 'accelerating_orders',
]


def derive_features(data: pd.DataFrame) -> pd.DataFrame:
    """All inputs are known as of the snapshot date; no cadence changes are judged without the full 90 days of visible history."""
    frame = data.copy()
    frame['has_90d_history'] = (frame.customer_history_days >= 90).astype(int)
    previous60 = frame.invoice_count_90d - frame.invoice_count_30d
    if (previous60 < 0).any() or (frame.invoice_count_90d > frame.invoice_frequency_180d).any():
        raise ValueError('Nested invoice windows must satisfy 30d <= 90d <= 180d')
    frame['monthly_order_change'] = (
        frame.invoice_count_30d - previous60 / 2
    ).where(frame.has_90d_history.eq(1))
    frame['recent90_order_share'] = frame.invoice_count_90d / frame.invoice_frequency_180d
    # The number of days since the first visible purchase, not the registration duration; press 30 days at the minimum to prevent the short history from being amplified.
    observed_days = frame.customer_history_days.clip(lower=30, upper=180)
    frame['history_adjusted_monthly_orders'] = frame.invoice_frequency_180d / observed_days * 30
    frame['recently_observed_30d'] = (frame.customer_history_days <= 30).astype(int)
    frame['one_invoice_180d'] = (frame.invoice_frequency_180d == 1).astype(int)
    frame['inactive_90d'] = (frame.invoice_count_90d == 0).astype(int)
    frame['paused_after_repeat'] = (
        frame.has_90d_history.eq(1) & frame.invoice_count_30d.eq(0) & frame.invoice_count_90d.ge(2)
    ).astype(int)
    frame['accelerating_orders'] = (frame.monthly_order_change >= 1).astype(int)
    return frame


def md_table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
    ])


def behavior_profile(frame, labels, k, split, base_features):
    grouped = frame.assign(cluster=labels).groupby('cluster', sort=True)
    result = grouped.agg(snapshot_count=('customer_id', 'size'),
                         unique_customer_count=('customer_id', 'nunique'))
    result['snapshot_share'] = result.snapshot_count / len(frame)
    for feature in base_features + DESCRIPTIVE_FEATURES:
        result[feature + '_median'] = grouped[feature].median()
        result[feature + '_mean'] = grouped[feature].mean()
    # Only training/verification is accepted here, and future labels will only be used for description after grouping; they will not participate in grouping or naming.
    result['observed_repurchase_rate'] = grouped['repurchase_30d'].mean()
    result.insert(0, 'split', split)
    result.insert(0, 'k', k)
    return result.reset_index()


def feature_experiments(data, pipelines, base_features, sample, seed):
    """Compare new representations of the same batch of training samples; additionally evaluate in the original 8 dimensional space to avoid cross-space mismatches."""
    train = data.loc[data.split.eq('train')]
    columns = base_features + EXTRA_FEATURES
    log_cols = train[columns].skew().abs().loc[lambda s: s > 1].index.tolist()
    logged = [columns.index(c) for c in log_cols]
    unchanged = [i for i in range(len(columns)) if i not in logged]
    original_space = pipelines[3][:-1].transform(train[base_features])
    rows, models, profiles = [], {}, []
    for k in range(3, 7):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('transform', ColumnTransformer([
                ('signed_log1p', FunctionTransformer(signed_log1p), logged),
                ('identity', 'passthrough', unchanged),
            ])),
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=seed)),
        ])
        labels = pipe.fit_predict(train[columns])
        counts = np.bincount(labels, minlength=k)
        transformed = pipe[:-1].transform(train[columns])
        base_labels = pipelines[k].predict(train[base_features])
        rows.append({
            'k': k, 'new_space_silhouette': silhouette_score(transformed[sample], labels[sample]),
            'original_space_silhouette': silhouette_score(original_space[sample], labels[sample]),
            'min_cluster_share': counts.min() / len(train),
            'agreement_with_original_same_k': adjusted_rand_score(base_labels, labels),
        })
        models[k] = pipe
        profiles.append(behavior_profile(train, labels, k, 'train', base_features))
    results = pd.DataFrame(rows)
    valid = results.loc[results.min_cluster_share >= .05]
    if valid.empty:
        valid = results
    chosen = int(valid.sort_values(['new_space_silhouette', 'k'], ascending=[False, True]).iloc[0]['k'])
    results['exploratory_selected'] = results.k.eq(chosen)
    bundle = {
        'pipeline': models[chosen], 'features': columns, 'selected_k': chosen,
        'log_features': log_cols, 'purpose': 'exploratory feature representation; not main model',
        'feature_builder': 'kmeans.derive_features',
    }
    return results, pd.concat(profiles, ignore_index=True), bundle



def collect_insights(data, pipelines, selected, sample, seed):
    """Intermediate calculations are kept in memory and only useful final results are given to the unified export function."""
    frame = derive_features(data)
    profiles = []
    with threadpool_limits(limits=1):
        for k, pipe in pipelines.items():
            labels = pipe.predict(frame[FEATURES])
            for split in ['train', 'validation']:
                mask = frame.split.eq(split)
                profiles.append(behavior_profile(frame.loc[mask], labels[mask], k, split, FEATURES))
            if k == selected:
                frame['cluster'] = labels
        experiments, experiment_profiles, exploratory = feature_experiments(
            frame, pipelines, FEATURES, sample, seed)
    train = frame.loc[frame.split.eq('train')]
    latest = train.loc[train.snapshot_date.eq(train.snapshot_date.max())]
    snapshot = latest.groupby('cluster').agg(
        customers=('customer_id', 'size'), spend_180d=('monetary_value_180d', 'sum')).reset_index()
    snapshot['customer_share'] = snapshot.customers / snapshot.customers.sum()
    snapshot['spend_share'] = snapshot.spend_180d / snapshot.spend_180d.sum()
    snapshot.insert(0, 'snapshot_date', latest.snapshot_date.iloc[0])
    slices = []
    for split in ['train', 'validation']:
        part = frame.loc[frame.split.eq(split) & frame.has_90d_history.eq(1)]
        for feature in ['accelerating_orders', 'paused_after_repeat']:
            for (cluster, flag), group in part.groupby(['cluster', feature]):
                slices.append({
                    'split': split, 'cluster': cluster, 'feature': feature, 'flag': flag,
                    'snapshot_count': len(group), 'unique_customer_count': group.customer_id.nunique(),
                    'observed_repurchase_rate': group.repurchase_30d.mean(),
                })
    monthly = frame.loc[frame.split.isin(['train', 'validation'])].groupby(
        ['split', 'snapshot_date', 'cluster']).agg(
            snapshot_count=('customer_id', 'size'),
            observed_repurchase_rate=('repurchase_30d', 'mean')).reset_index()
    return {
        'frame': frame, 'profiles': pd.concat(profiles, ignore_index=True),
        'experiments': experiments, 'experiment_profiles': experiment_profiles,
        'exploratory': exploratory, 'snapshot': snapshot,
        'slices': pd.DataFrame(slices), 'monthly': monthly,
    }


def json_records(frame):
    """Dates are converted to ISO format and missing values are converted to JSON null."""
    return json.loads(frame.to_json(orient='records', date_format='iso'))


def export_results(output, data, comparison, metrics, cluster_features, rankings,
                   split_profiles, partitions, preprocessing, bundle, results, names, manifest):
    """Unified export: four tables, an evidence record, a model package and a running record."""
    customer = data[KEYS + FEATURES].merge(cluster_features, on=KEYS, validate='one_to_one')
    customer = customer.merge(rankings, on=KEYS, validate='one_to_one')
    customer = customer.merge(results['frame'][KEYS + DESCRIPTIVE_FEATURES],
                              on=KEYS, validate='one_to_one')
    customer['report_name'] = customer.cluster.map(names)
    if len(customer) != len(data):
        raise AssertionError('Customer consolidation changed the sample count')
    pd.testing.assert_frame_equal(customer[KEYS], data[KEYS])
    save_csv(customer, output / 'data/customer_results.csv')

    original_profiles = results['profiles'].assign(feature_set='original_8')
    test_profiles = split_profiles.loc[split_profiles.split.eq('test')].drop(columns=['cluster_name'])
    test_profiles = test_profiles.assign(feature_set='original_8')
    exploratory_profiles = results['experiment_profiles'].assign(feature_set='enhanced_12')
    profiles = pd.concat([original_profiles, test_profiles, exploratory_profiles], ignore_index=True)
    profiles['is_main_model'] = profiles.feature_set.eq('original_8') & profiles.k.eq(bundle['selected_k'])
    profiles['report_name'] = profiles.cluster.map(names).where(profiles.is_main_model, '')
    front = ['feature_set', 'k', 'split', 'cluster', 'report_name', 'is_main_model',
             'snapshot_count', 'unique_customer_count', 'snapshot_share']
    profiles = profiles[front + [c for c in profiles if c not in front]]
    save_csv(profiles, output / 'data/cluster_profiles.csv')

    original_comparison = comparison.rename(columns={'silhouette_train': 'silhouette_in_own_space'})
    original_comparison = original_comparison.assign(
        feature_set='original_8', silhouette_in_original_space=comparison.silhouette_train,
        agreement_with_original_same_k=1.0,
    )
    enhanced_comparison = results['experiments'].rename(columns={
        'new_space_silhouette': 'silhouette_in_own_space',
        'original_space_silhouette': 'silhouette_in_original_space',
        'exploratory_selected': 'selected',
    }).assign(feature_set='enhanced_12')
    enhanced_comparison['size_floor_passed'] = enhanced_comparison.min_cluster_share >= .05
    comparisons = pd.concat([original_comparison, enhanced_comparison], ignore_index=True)
    comparisons = comparisons[['feature_set', 'k'] + [c for c in comparisons if c not in ['feature_set', 'k']]]
    save_csv(comparisons, output / 'data/model_comparison.csv')
    save_csv(metrics, output / 'data/baseline_metrics.csv')

    evidence = {
        'description': 'Same-day consumption share, purchase rhythm subgroups, and monthly repurchase comparisons in the report; do not include test label evaluations.',
        'latest_training_snapshot': json_records(results['snapshot']),
        'behavior_subgroups': json_records(results['slices']),
        'monthly_repurchase': json_records(results['monthly']),
    }
    (output / 'data/behavior_evidence.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    bundle['report_names'] = names
    bundle['exploratory'] = results['exploratory']
    bundle['schema_version'] = 2
    model_path = output / 'model/segmentation_bundle.joblib'
    joblib.dump(bundle, model_path)
    loaded = joblib.load(model_path)
    with threadpool_limits(limits=1):
        np.testing.assert_allclose(
            loaded['pipeline'].transform(data[FEATURES]),
            customer[bundle['distance_columns']], rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(loaded['pipeline'].predict(data[FEATURES]), customer.cluster)
        extra = results['frame'][loaded['exploratory']['features']]
        np.testing.assert_allclose(loaded['exploratory']['pipeline'].transform(extra),
                                   bundle['exploratory']['pipeline'].transform(extra), atol=1e-12)
    pd.testing.assert_frame_equal(baseline_scores(data, loaded['rfm_config']), rankings)
    manifest['cutoff_partitions'] = json_records(partitions)
    manifest['preprocessing_audit'] = json_records(preprocessing)
    manifest['insights'] = {
        'extra_model_features': EXTRA_FEATURES,
        'feature_experiment_selected_k': results['exploratory']['selected_k'],
        'main_model_changed_by_feature_experiment': False,
        'test_labels_used_for_insights': False,
        'momentum_scope': 'customer_history_days >= 90; otherwise missing',
    }
    manifest['output_layout'] = {
        'report': 'stage4_report.md', 'customer_details': 'data/customer_results.csv',
        'profiles': 'data/cluster_profiles.csv', 'comparison': 'data/model_comparison.csv',
        'baselines': 'data/baseline_metrics.csv', 'evidence': 'data/behavior_evidence.json',
        'models': 'model/segmentation_bundle.joblib',
        'figures': ['figures/customer_group_story.png', 'figures/cluster_followup_story.png'],
    }
    (output / 'run_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    verify_report_links(output)


def verify_report_links(output):
    import re
    report = (output / 'stage4_report.md').read_text(encoding='utf-8')
    for target in re.findall(r'\]\(([^)]+)\)', report):
        if not target.startswith(('https://', 'http://', '#')):
            if not (output / target).is_file():
                raise AssertionError(f'Broken report link: {target}')


def render_business_report(output, data, comparison, metrics, selected, results, manifest):
    """The report template below is generated from the data, ensuring that reruns do not revert to old term stack versions."""
    profiles = results['profiles']
    train_profiles = profiles.loc[(profiles.k == selected) & profiles.split.eq('train')].copy()
    # The report is arranged by recent to long-term purchases, and the group number is retained for easy comparison with the original export.
    train_profiles = train_profiles.sort_values('recency_days_median')
    ids = train_profiles.cluster.astype(int).tolist()
    if selected == 3:
        names = dict(zip(ids, ['Frequent buyers', 'Infrequent buyers', 'Long inactive']))
    else:
        names = {int(row.cluster): f'Group {int(row.cluster) + 1} ( {row.recency_days_median:.0f} days apart)'
                 for row in train_profiles.itertuples()}
    # Save the naming basis and use the name for this report without changing the exported center number.
    make_story_figures(output, train_profiles, results['profiles'], names, selected, results['snapshot'])
    p = train_profiles.set_index('cluster')
    validation = profiles.loc[(profiles.k == selected) & profiles.split.eq('validation')].set_index('cluster')
    high, middle, quiet = ids[0], ids[len(ids) // 2], ids[-1]
    hi, mid, low = p.loc[high], p.loc[middle], p.loc[quiet]
    latest = results['snapshot'].set_index('cluster')
    latest_date = results['snapshot'].snapshot_date.iloc[0].strftime('%Y year %m month %d day')
    k4 = profiles.loc[profiles.k.eq(4) & profiles.split.eq('train')].copy()
    k5 = profiles.loc[profiles.k.eq(5) & profiles.split.eq('train')].copy()
    k6 = profiles.loc[profiles.k.eq(6) & profiles.split.eq('train')].copy()
    big_ticket = k4.loc[k4.average_order_value_180d_median.idxmax()]
    frequent = k4.loc[k4.invoice_frequency_180d_median.idxmax()]
    quiet4 = k4.loc[k4.recency_days_median.idxmax()]
    remaining = k4.loc[~k4.cluster.isin([big_ticket.cluster, frequent.cluster, quiet4.cluster])]
    light4 = remaining.iloc[0] if len(remaining) else k4.iloc[0]
    cheapest5 = k5.loc[k5.monetary_value_180d_median.idxmin()]
    biggest6 = k6.loc[k6.invoice_frequency_180d_median.idxmax()]
    pair_candidates = []
    for a, b in combinations(k6.to_dict('records'), 2):
        if (a['invoice_frequency_180d_median'] == b['invoice_frequency_180d_median']
                and a['invoice_frequency_180d_median'] > 1
                and abs(a['recency_days_median'] - b['recency_days_median']) > 20):
            gap = abs(np.log(a['monetary_value_180d_median'] / b['monetary_value_180d_median']))
            pair_candidates.append((gap, a, b))
    pair_text = ''
    if pair_candidates:
        _, a, b = min(pair_candidates, key=lambda value: value[0])
        a, b = sorted([a, b], key=lambda r: r['recency_days_median'])
        pair_text = (
            f"The six-group plan also found a pair of groups worthy of comparison: both groups bought about **{a['invoice_frequency_180d_median']:.0f} times** in half a year,"
            f"The amount is also close to (**{a['monetary_value_180d_median']:,.0f} / {b['monetary_value_180d_median']:,.0f}**),"
            f"However, it is **{a['recency_days_median']:.0f} days / {b['recency_days_median']:.0f} days** since the last purchase."
            'Their consumption scale in the past was similar, but their current purchasing stages are different. This is a difference that is easily overlooked when stratified only by total consumption.'
        )
    slices = results['slices']
    def slice_row(cluster, feature, flag):
        view = slices.loc[slices.split.eq('validation') & slices.cluster.eq(cluster)
                          & slices.feature.eq(feature) & slices.flag.eq(flag)]
        return view.iloc[0] if len(view) else None
    acceleration = [slice_row(high, 'accelerating_orders', flag) for flag in [1, 0]]
    pausing = [slice_row(middle, 'paused_after_repeat', flag) for flag in [1, 0]]
    selected_experiment = results['experiments'].loc[results['experiments'].exploratory_selected].iloc[0]
    experiment_profiles = results['experiment_profiles']
    exp = experiment_profiles.loc[experiment_profiles.k.eq(selected_experiment.k)]
    short_history = exp.loc[exp.customer_history_days_median.idxmin()]
    base_score = float(comparison.loc[comparison.k.eq(selected), 'silhouette_train'].iloc[0])
    new_same_space = float(results['experiments'].loc[results['experiments'].k.eq(selected), 'original_space_silhouette'].iloc[0])
    base_valid = metrics.loc[metrics.split.eq('validation') & metrics.snapshot_date.eq('ALL')].set_index('baseline')
    rfm = base_valid.loc['rfm']

    group_rows = []
    for cluster in ids:
        row = p.loc[cluster]
        group_rows.append([
            f'{names[cluster]}（C{cluster}）', f'{row.snapshot_share:.1%}',
            f'{row.recency_days_median:.0f} days', f'{row.invoice_frequency_180d_median:.0f} times',
            f'{row.monetary_value_180d_median:,.0f}',
        ])
    comparison_details = {
        3: 'Establish a general outline: frequent purchases, low-frequency purchases, and no purchases for a long time.',
        4: f'A "high single transaction, low frequency" group appears: a typical single transaction is {big_ticket.average_order_value_180d_median:.0f}, purchased {big_ticket.invoice_frequency_180d_median:.0f} times in half a year, and {big_ticket.recency_days_median:.0f} days since the last purchase.',
        5: f'Identify a group of customers with low consumption and few types of goods purchased: half-year amount {cheapest5.monetary_value_180d_median:.0f}, type of goods {cheapest5.product_diversity_180d_median:.0f}.',
        6: f'High-frequency customers continue to be segmented: the most frequent group buys {biggest6.invoice_frequency_180d_median:.0f} times in half a year, accounting for {biggest6.snapshot_share:.1%}; groups with similar consumption and different recent activities are also separated.',
    }
    comparison_rows = [[f'{int(r.k)} group', f'{r.silhouette_train:.4f}', f'{r.min_cluster_share:.1%}',
                        comparison_details[int(r.k)]] for r in comparison.itertuples()]
    k4_rows = []
    for title, row in [('Frequent purchase type', frequent), ('High single pen, low frequency type', big_ticket),
                       ('Low frequency, short visible history', light4), ('Long inactive', quiet4)]:
        k4_rows.append([f'{title}（C{int(row.cluster)}）', f'{row.snapshot_share:.1%}',
                        f'{row.recency_days_median:.0f} days', f'{row.invoice_frequency_180d_median:.0f} times',
                        f'{row.average_order_value_180d_median:,.0f}'])
    followup_rows = [[names[c], f'{p.loc[c, "observed_repurchase_rate"]:.1%}',
                     f'{validation.loc[c, "observed_repurchase_rate"]:.1%}'] for c in ids]
    monthly = results['monthly'].loc[results['monthly'].split.eq('validation')]
    ranges = monthly.groupby('cluster').observed_repurchase_rate.agg(['min', 'max'])
    lines = [
        '#Customer segmentation: Who buys frequently, who only comes occasionally, and who hasn\'t bought for a long time?', '',
        '> What this report wants to answer is what are the differences between customers that deserve attention, and what these differences can bring to subsequent analysis.', '',
        '## Let\'s look at the three most noteworthy discoveries first.', '',
        f'1. **People who buy often may not necessarily be particularly expensive every time.** The number of purchases in half a year for the Frequent Visitors and Frequent Buyers group is about the same as that of the Low Frequency Buyers group.'
        f'{hi.invoice_frequency_180d_median / mid.invoice_frequency_180d_median:.1f} times,'
        f'But the typical single transaction amount is only about {hi.average_order_value_180d_median / mid.average_order_value_180d_median:.1f} times. Frequency of purchase is a very prominent difference between the two groups.',
        f'2. **A single transaction costs a lot, and it may not be active in the near future.** When divided into four groups, the typical single transaction amount of one group is'
        f'{big_ticket.average_order_value_180d_median:,.0f}, higher than the most frequently purchased group'
        f'{frequent.average_order_value_180d_median:,.0f}; but the time since the last purchase was'
        f'{big_ticket.recency_days_median:.0f} days and {frequent.recency_days_median:.0f} days.',
        '3. **Just because I have bought a lot recently does not mean I will definitely continue to buy next month.** The newly added "Purchase Rhythm Change" feature discovered a counter-intuitive phenomenon, and specific comparisons are given later.', '',
        'Reading caliber: A customer records one record every month, which is called "Customer Monthly Record". The same person may belong to different groups in different months.'
        'Unless otherwise specified, the profiles are recorded using the training period ( 2010 year 6 month - 2011 year 5 month); the "typical value" refers to the median, which is the number in the middle after sorting. The amount follows the original unit of CSV.', '',
        '## 1. Divided into three groups, what kind of people are they?', '',
        '**k Just want to divide into several groups. k= 3 means divided into three groups; C 0, C 1, C 2 are just numbers, there is no order of priority.**', '',
        md_table(['group', 'Record proportion', 'since last purchase', 'Number of purchases in half a year', 'Half-year consumption amount'], group_rows), '',
        '![Comparison of purchasing behavior of three types of customers ](figures/customer_group_story.png)', '',
        f'### {names[high]}: The core feature is to buy frequently and buy more products.', '',
        f'The typical number of purchases for this group of people in half a year is **{hi.invoice_frequency_180d_median:.0f} times**,'
        f'Purchased **{hi.product_diversity_180d_median:.0f} products**; the low-frequency groups are'
        f'{mid.invoice_frequency_180d_median:.0f} times, {mid.product_diversity_180d_median:.0f} kinds.'
        f'The typical single amount is **{hi.average_order_value_180d_median:,.0f}**, and the low-frequency group is **{mid.average_order_value_180d_median:,.0f}**.'
        'This is more like a group of customers who continue to buy and have a wide range of products. The single amount is the median of the average order amount per customer, and several medians cannot be directly multiplied.', '',
        f'Let\'s look at another example of not counting customers twice: **{latest_date}**, this group only'
        f'**{int(latest.loc[high, "customers"]):,} people**, accounted for the customers of the day **{latest.loc[high, "customer_share"]:.1%}**,'
        f'However, it accounts for **{latest.loc[high, "spend_share"]:.1%}** of the spending amount of these customers in the past 180 days.'
        'This ratio indicates that maintaining the continuity of their purchases is a concern; it is the share of consumption within the customer range for that day.', '',
        f'### {names[middle]}: There are many people and they have not yet formed the habit of frequent purchases.', '',
        f'This group accounts for all records **{mid.snapshot_share:.1%}**, among which **{mid.one_invoice_180d_mean:.1%}**'
        f'Only purchased once in the past six months; first visible purchase of **{mid.recently_observed_30d_mean:.1%}** was within the last 30 days.'
        'Therefore, this group includes both customers who have recently appeared in the data, as well as older customers who do not buy frequently.', '',
        '**You can continue to ask:** For people who only have one purchase record, what historical behaviors are related to the second purchase?'
        '"First visible purchase" is the first purchase record in the data and cannot be directly regarded as the actual registration time.', '',
        f'### {names[quiet]}: The main difference is the long silence time', '',
        f'It has been **{low.recency_days_median:.0f} days** since the last time this group purchased a typical product,'
        f'The record for **{low.inactive_90d_mean:.1%}** had no purchases in the past 90 days.'
        f'But its typical single amount is **{low.average_order_value_180d_median:,.0f}**,'
        f'Close to **{mid.average_order_value_180d_median:,.0f}** of the low frequency group.'
        'Describing it as "not purchased for a long time" is more accurate than just labeling it "low value".', '',
        '**You can continue to ask:** Are there different replenishment cycles for people who have not purchased for a long time? The existing snapshot does not have the complete order interval and product category, so we can only raise this pending verification issue for the time being.', '',
        '## 2. Why should we watch the fourth, fifth and sixth groups?', '',
        'Divide into more groups to see the profile more closely. It will reclassify all customers, so the four-group solution does not mean simply splitting one of the three groups.', '',
        'The "degree of separation" in the table corresponds to the silhouette coefficient: the higher it is, the more similar the same group of customers are and the easier it is for different groups to separate. It\'s not accuracy.', '',
        md_table(['Plan', 'degree of separation', 'Minimum group proportion', 'What did you see more than the overview?'], comparison_rows), '',
        f'**The current main model follows the {selected} group.** The only difference between the scores of the third group and the fourth group is'
        f'{abs(float(comparison.loc[comparison.k.eq(3), "silhouette_train"].iloc[0]) - float(comparison.loc[comparison.k.eq(4), "silhouette_train"].iloc[0])):.4f}，'
        'So it cannot be said that the three groups clearly win. Three groups are suitable for talking about the overall structure; four groups provide valuable business details that are worth retaining in the display. The five and six groups can be used as a more detailed observation perspective, and the degree of separation has not continued to improve.', '',
        '### Among the four group solutions, the "high single transaction, low frequency" group deserves to be discussed separately.', '',
        md_table(['Groups in the four-group scheme', 'Record proportion', 'since last purchase', 'Number of purchases in half a year', 'Typical single transaction amount'], k4_rows), '',
        f'"High single transaction, low frequency type" accounts for **{big_ticket.snapshot_share:.1%}**, and the typical half-year consumption amount is'
        f'**{big_ticket.monetary_value_180d_median:,.0f}**. This reminds us: high consumption amount can come from "buying frequently" or "buying more each time".'
        'Whether the two types of customers are suitable for the same maintenance rhythm requires further research.', '',
        pair_text, '',
        '## 3. What features were newly discovered?', '',
        'The original features tell us "how much was bought". This time I added "Have there been any changes recently, and how often have I purchased it in the limited history?"'
        'These features are calculated from existing fields and are new expressions; real order intervals, category preferences, and return preferences still require transaction details.', '',
        md_table(['New features', 'Popular meaning and algorithm', 'How to understand'], [
            ['Changes in buying pace', 'The number of purchases in the last 30 days − the average number of purchases per month in the previous 60 days',
             'For example, in the last 30 days, 3 times were purchased, and in the previous 60 days, a total of 2 times were purchased, and the change is + 2. Only calculated for people who have at least 90 days of visible history.'],
            ['Proportion of recent purchases', 'Number of purchases in the last 90 days ÷ Number of purchases in the last 180 days',
             'Close to 1: purchases within half a year are concentrated in the recent period; for 0: no purchases were made in the last 90 days. Customers with short history may naturally have higher prices.'],
            ['Number of monthly purchases converted based on visible history', 'Number of purchases in half a year ÷ Number of visible historical days × 30; limited number of days 30 — 180 days',
             'For the same purchase 2 times, the purchase frequency should be treated differently if it occurs for two months and for half a year.'],
            ['Is history long enough?', 'Is there at least 90 days of visible purchase history?',
             'Prompt whether there is enough observation time for changes in rhythm to avoid taking the lack of history as no purchase.'],
        ]), '',
        'In addition, two markers have been added to facilitate finding people: **Recent acceleration** = rhythm changes have increased by at least 1 times/month;'
        '**Temporarily suspended after repeated purchases** = There is a visible history of 90 days, and purchases have been made at least 2 times in the past 90 days, but no purchases have been made in the last 30 days.', '',
        '### Do new features really make clustering better? Actually tried it', '',
        f'After adding the above 4 item to the original 8 item, compare the 3 — 6 group. The new feature scheme selected **{int(selected_experiment.k)} group**.'
        f'It singles out a group that accounts for **{short_history.snapshot_share:.1%}**,'
        f'First visible purchase is typically only **{short_history.customer_history_days_median:.0f} days** away,'
        f'Of which **{1 - short_history.has_90d_history_mean:.1%}** has less than 90 days of visible history.'
        'Explain that the new features allow the model to emphasize "what observation stage the customer is in".', '',
        f'Measured by the newly added 12 features, the new solution scores **{selected_experiment.new_space_silhouette:.4f}**.'
        'But the "ruler" of measurement has changed, and this number cannot be directly compared with the original score.'
        f'Reuse the original 8 features to measure the new classification method for the same {selected} group, and the score is **{new_same_space:.4f}**,'
        f'The original plan is **{base_score:.4f}**.'
        'Therefore, the gain at this stage is that a new observation dimension has been found, and there is no evidence that the original model should be replaced with a new solution.', '',
        '## 4. Will these characteristics still hold true over time?', '',
        'Use the original model to group customers from 2011 years to 6 to 8 months, and then see if they actually purchase again after 30 days. The following results are only used for post-mortem inspection, and future purchase records will not be included in the grouping.', '',
        md_table(['group', 'Actual repurchase ratio during training period', 'Actual repurchase ratio during verification period'], followup_rows), '',
        '![Actual repurchase comparison of each group in the next period of time ](figures/cluster_followup_story.png)', '',
        f'The repurchase ratio of frequent visitors and frequent buyers in the three verification months is **{ranges.loc[high, "min"]:.1%} — {ranges.loc[high, "max"]:.1%}**,'
        f'The group that has not purchased for a long time is **{ranges.loc[quiet, "min"]:.1%} — {ranges.loc[quiet, "max"]:.1%}**.'
        'This difference exists in the three verification months, indicating that the profiles have subsequent behavioral correspondence. It also shows that "I haven\'t bought it for a long time" does not mean that I will never buy it again.', '',
        '### A counter-intuitive discovery: recent acceleration may not necessarily continue to accelerate', '',
    ]
    if all(row is not None for row in acceleration):
        yes, no = acceleration
        lines += [
            f'Within **{names[high]}**, only records with visible history that are 90 days old are compared:', '',
            md_table(['Validation period subgroup', 'Number of customer records per month', 'Actual repurchase ratio after 30 days'], [
                ['The number of recent purchases has increased significantly', f'{int(yes.snapshot_count):,}', f'{yes.observed_repurchase_rate:.1%}'],
                ['did not reach the above-mentioned increase rate', f'{int(no.snapshot_count):,}', f'{no.observed_repurchase_rate:.1%}'],
            ]), '',
            'This result reminds us: there is a rhythm to buying, and the recent concentrated buying may not continue into the next month.'
            'Whether it is related to concentrated replenishment, seasonality or customer composition, the current data cannot be confirmed. "Changes in purchasing rhythm" can be submitted to the stage five model for testing, but cannot be written directly as "the more active you are, the easier it is to repurchase".', '',
        ]
    if all(row is not None for row in pausing):
        yes, no = pausing
        lines += [
            '### Another detail: People who haven\'t bought yet may still have the basis to buy.', '',
            f'In **{names[middle]}**, the record of "Temporary interruption after repeated purchase" is'
            f'**{int(yes.snapshot_count):,} items**, the repurchase ratio of 30 days after the verification period is **{yes.observed_repurchase_rate:.1%}**;'
            f'The remaining visible records in the same group that have a history of 90 days are **{no.observed_repurchase_rate:.1%}** ( {int(no.snapshot_count):,} records).', '',
            'This tag includes both "purchased more than once in the past" and "recently paused," so the higher repurchase rate may come from an existing purchase base.'
            'It is suitable for distinguishing customer status, but it cannot be used to interpret "pause" as the reason for repurchase.', '',
        ]
    lines += [
        'The above are exploratory observations. The same customer can contribute multiple monthly records and cannot be used as an independent sample for ordinary significance testing;'
        '"Salient features" here refer to observable differences without claims of statistical significance or effectiveness of marketing measures. The testing period is still left for final evaluation.', '',
        '## 5. What reference does simple sorting set for subsequent prediction models?', '',
        'Assume that you can only view 20% customers every month: RFM At the same time, look at "whether you bought recently, how many times you bought, and how much you spent" as a simple sorting reference for subsequent prediction models.', '',
        md_table(['Sorting method', 'A total of records selected during the verification period', 'which will be purchased again after 30 days', 'Repurchase ratio within the list'], [
            ['Recent, times and amount ( RFM )', f'{int(rfm.top20_count):,}', f'{int(rfm.top20_hits):,}', f'{rfm.precision_at_20pct:.1%}'],
        ]), '',
        'Subsequent complex models should be compared with this simple approach to determine whether the added complexity is worth it. This is a look back at the historical list and does not mean that marketing campaigns resulted in these purchases.', '',
        '## 6. What can you say when reporting?', '',
        '> We first use three groups to capture the overall structure: frequent buyers, low-frequency buyers, and long-term non-buyers.'
        'Then we refine the grouping and find that high consumption may come from high-frequency purchases or higher single amounts; people with similar consumption may have different levels of recent activity.'
        'After further constructing the buying rhythm and visible historical characteristics, it is found that the recent acceleration does not necessarily extend into the next month.'
        'Therefore, the value of customer segmentation is to clarify these differences and raise testable questions for subsequent repurchase predictions.', '',
        '## Data files and reproduction', '',
        f'- All qualified snapshots **{len(data):,}**; the training period is used to discover groupings, and the verification period is used for subsequent behavioral control.',
        '- Run `python archive/segmentation/kmeans.py` in the project root directory to regenerate analysis and reports; run `python archive/segmentation/kmeans.py --check` to check existing results.',
        '- `data/` holds the result table, `figures/` holds the two pictures quoted in the text, and `model/` holds the loadable model.', '',
        md_table(['File', 'Content and purpose'], [
            ['[Customer results summary table ](data/customer_results.csv)', 'One row per customer per month: original features, new features, clustering, center distance, two sortings, and top 20% tags.'],
            ['[Group profile summary ](data/cluster_profiles.csv)', 'Profiles of each group in the original and newly added feature schemes; filter by feature_set, k, split.'],
            ['[Comparison of grouping schemes ](data/model_comparison.csv)', 'Comparison of k= 3 — 6 features of the original 8 term and the newly added 12 term, totaling 8 solutions.'],
            ['[Sort Baseline Performance ](data/baseline_metrics.csv)', 'Month-by-month and summary sorting performance during training and validation periods.'],
            ['[Basis for behavioral discovery ](data/behavior_evidence.json)', 'Same-day consumption share, purchase rhythm subgroups, and monthly repurchase comparisons are combined into one record.'],
            ['[Model package ](model/segmentation_bundle.joblib)', 'The main model, RFM cut point, and new feature exploration model share one file.'],
            ['[Run record ](run_manifest.json)', 'Enter hashes, versions, configurations, date partitioning, and preprocessing statistics.'],
        ]), '',
        'In the profile table, `feature_set=original_8` is the original plan, and `enhanced_12` is the new feature exploration plan;'
        '`is_main_model=True` is the main model used in the main text. `split` differentiates between training, validation and testing;'
        '`snapshot_count` is the number of monthly customer records, `snapshot_share` is the proportion, `_median` is the median, and `_mean` is the mean.'
        'The actual repurchase ratio during the test period is left blank.', '',
        '<details>', '<summary>Instructions for using methods and models (click to expand)</summary>', '',
        '**Training boundaries.** Following the upstream time division, filling, transformation and scaling are only fitted during the training period; customer ID, date and future labels are not entered for grouping.'
        'Whether the original transaction and the last period of observation are complete still need to be corroborated by the transaction details of stages two and three.', '',
        f'**Group method.** The original 8 feature is filled with the median of the training period, and the column with a skewness greater than 1 is log 1 p, and then the scale is unified.'
        f'Four k use the same batch of {manifest["silhouette_sample_size"]:,} training records to compare the degree of separation;'
        f'Random seed {manifest["seed"]}, try 20 initial center for each solution. First, the minimum group proportion is required to reach 5%, and then the solution with the highest degree of separation is selected.'
        'The new feature scheme allows purchase rhythm changes to be negative, using a logarithmic transformation that preserves the sign, and is fitted independently.', '',
        '**Sort caliber.** The recent purchase score is the inverse of the number of days since last purchase. RFM is scored according to the quintile cut-off point during the training period.'
        'R reverse, F/M forward, each 1 - 5 is divided and then added together; the same value is in the same gear. Each date is rounded up to the nearest 20% (the number of people is rounded up),'
        'Sort by customer ID text in ascending order. The summary list performance is based on the total of the selected records month by month; the score is the basis for sorting.', '',
        '**Load model.** Run in the project root directory:', '',
        '```python', 'import joblib',
        "bundle = joblib.load('archive/artifacts/segmentation/stage4/model/segmentation_bundle.joblib')",
        "X = snapshots[bundle['features']]",
        "clusters = bundle['pipeline'].predict(X)",
        "distances = bundle['pipeline'].transform(X)",
        "# The exploration model is located in bundle['exploratory'], and the input is first generated using stage 4.kmeans.derive_features.", '```', '',
        'The center distance of the main model is located in the feature space after unified scale, which can be used in stage five; rolling verification should be refitted during each fold training period.'
        'The exploration models in the model package use the importable transformation functions in stage 4.transforms.'
        'Dependency versions are recorded in run_manifest.json.', '',
        '</details>', '',
    ]
    (output / 'stage4_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return names


def make_story_figures(output, profiles, all_profiles, names, selected, snapshot):
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((f for f in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'] if f in available), 'DejaVu Sans')
    with plt.rc_context({'font.family': font, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 11}):
        colors = ['#227C9D', '#D99429', '#8B929A']
        ids = profiles.cluster.astype(int).tolist()
        labels = [names[c] for c in ids]
        fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), layout='constrained')
        for ax, col, title, unit in zip(axes,
                ['recency_days_median', 'invoice_frequency_180d_median', 'monetary_value_180d_median'],
                ['Days since last purchase', '180-day order count', '180-day spending'], ['Days', 'Orders', 'GBP']):
            vals = profiles[col].to_numpy()
            ax.barh(labels, vals, color=colors[:len(ids)])
            ax.invert_yaxis()
            ax.set_title(title, fontweight='bold', pad=12)
            ax.set_xlabel(unit)
            ax.set_xlim(0, vals.max() * 1.3)
            for i, value in enumerate(vals):
                ax.text(value + vals.max() * .025, i, f'{value:,.0f}', va='center')
        fig.suptitle('Purchase patterns across three segments' if selected == 3 else 'Typical buying behavior of each group', fontsize=17)
        fig.savefig(output / 'figures' / 'customer_group_story.png', dpi=170, facecolor='white')
        plt.close(fig)
        validation = all_profiles.loc[(all_profiles.k == selected) & all_profiles.split.eq('validation')].set_index('cluster')
        fig, ax = plt.subplots(figsize=(8.6, 4.3), layout='constrained')
        x = np.arange(len(ids))
        train = profiles.set_index('cluster').loc[ids, 'observed_repurchase_rate'] * 100
        valid = validation.loc[ids, 'observed_repurchase_rate'] * 100
        for shift, values, label, color in [(-.18, train, 'Training', '#93BCD0'), (.18, valid, 'Validation', '#227C9D')]:
            bars = ax.bar(x + shift, values, .34, label=label, color=color)
            ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=4)
        ax.set_xticks(x, labels)
        ax.set_ylabel('Repurchase within 30 days (%)')
        ax.set_ylim(0, max(train.max(), valid.max()) * 1.28)
        ax.set_title('Segment patterns in training and validation', fontsize=16, pad=15)
        ax.legend(frameon=False)
        fig.savefig(output / 'figures' / 'cluster_followup_story.png', dpi=170, facecolor='white')
        plt.close(fig)

def run_self_checks(source: Path, output: Path) -> None:
    class StageFourTests(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.data, _ = load_snapshots(source)
            cls.bundle = joblib.load(output / 'model/segmentation_bundle.joblib')

        def test_saved_pipeline_matches_export_in_fresh_process(self):
            expected = pd.read_csv(output / 'data/customer_results.csv',
                                   dtype={'customer_id': 'string'})
            pipeline = self.bundle['pipeline']
            with threadpool_limits(limits=1):
                distances = pipeline.transform(self.data[FEATURES])
                labels = pipeline.predict(self.data[FEATURES])
            np.testing.assert_allclose(distances, expected[self.bundle['distance_columns']],
                                       rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(labels, expected['cluster'])
            self.assertEqual(pipeline.feature_names_in_.tolist(), FEATURES)

        def test_heldout_extremes_and_missing_values_do_not_refit(self):
            pipeline = self.bundle['pipeline']
            imputer = pipeline.named_steps['imputer']
            scaler = pipeline.named_steps['scaler']
            medians, means, scales = (imputer.statistics_.copy(), scaler.mean_.copy(), scaler.scale_.copy())
            heldout = self.data.loc[self.data['split'].eq('test'), FEATURES].head(10).astype(float)
            heldout.iloc[0, :] = np.nan
            heldout.iloc[1, :] = 1e9
            with threadpool_limits(limits=1):
                distances = pipeline.transform(heldout)
            self.assertTrue(np.isfinite(distances).all())
            np.testing.assert_array_equal(imputer.statistics_, medians)
            np.testing.assert_array_equal(scaler.mean_, means)
            np.testing.assert_array_equal(scaler.scale_, scales)
            train = self.data.loc[self.data['split'].eq('train'), FEATURES]
            np.testing.assert_allclose(medians, train.median().to_numpy())

        def test_cross_partition_label_window_rejected(self):
            raw = pd.read_csv(source)
            row = raw.index[raw['split'].eq('train')][-1]
            raw.loc[row, 'snapshot_date'] = '2011/5/15 0:00'
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'bad.csv'
                raw.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, 'Label window crosses'):
                    load_snapshots(path)

        def test_duplicate_snapshot_key_rejected(self):
            raw = pd.read_csv(source).head(3)
            raw = pd.concat([raw, raw.iloc[[0]]])
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'duplicate.csv'
                raw.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, 'Duplicate snapshot'):
                    load_snapshots(path)

        def test_rfm_ties_train_cutpoints_and_daily_top20(self):
            train = self.data.loc[self.data['split'].eq('train')]
            config = fit_rfm(train)
            self.assertEqual(config, self.bundle['rfm_config'])
            fixture = self.data.iloc[:10].copy()
            fixture.loc[:, 'snapshot_date'] = pd.to_datetime(['2011-06-01'] * 5 + ['2011-07-01'] * 5)
            fixture.loc[:, 'split'] = 'validation'
            fixture.loc[:, RFM] = [20, 1, 100]
            scores = baseline_scores(fixture, config)
            self.assertEqual(scores['rfm_score'].nunique(), 1)
            self.assertEqual(set(scores.columns) - set(KEYS),
                             {'r_score', 'f_score', 'm_score', 'rfm_score',
                              'rfm_rank', 'rfm_top20'})
            for name in ['rfm']:
                counts = scores.groupby('snapshot_date')[name + '_top20'].sum()
                self.assertEqual(counts.tolist(), [1, 1])
            # Entering row order cannot change the score, ranking, or selection status of the same customer.
            shuffled = baseline_scores(fixture.sample(frac=1, random_state=7), config)
            pd.testing.assert_frame_equal(scores.sort_index(), shuffled.sort_index())

        def test_behavior_features_handle_short_history_and_momentum(self):
            fixture = self.data.head(3).copy()
            fixture['customer_history_days'] = [180, 20, 120]
            fixture['invoice_frequency_180d'] = [8, 2, 6]
            fixture['invoice_count_90d'] = [5, 2, 4]
            fixture['invoice_count_30d'] = [3, 2, 0]
            derived = derive_features(fixture)
            self.assertEqual(derived.monthly_order_change.iloc[0], 2)
            self.assertTrue(np.isnan(derived.monthly_order_change.iloc[1]))
            self.assertEqual(derived.has_90d_history.iloc[1], 0)
            self.assertEqual(derived.monthly_order_change.iloc[2], -2)
            self.assertEqual(derived.paused_after_repeat.tolist(), [0, 0, 1])
            self.assertEqual(derived.recent90_order_share.iloc[0], 5 / 8)
            np.testing.assert_allclose(derived.history_adjusted_monthly_orders, [8 / 6, 2, 1.5])

        def test_derived_features_do_not_use_future_labels(self):
            fixture = self.data.head(50).copy()
            expected = derive_features(fixture)[DESCRIPTIVE_FEATURES]
            fixture['repurchase_30d'] = 1 - fixture['repurchase_30d']
            actual = derive_features(fixture)[DESCRIPTIVE_FEATURES]
            pd.testing.assert_frame_equal(actual, expected)
            fixture['invoice_count_30d'] = fixture.invoice_count_90d + 1
            with self.assertRaisesRegex(ValueError, 'Nested invoice windows'):
                derive_features(fixture)

        def test_new_feature_model_imputation_uses_training_only(self):
            bundle = joblib.load(output / 'model/segmentation_bundle.joblib')['exploratory']
            derived = derive_features(self.data)
            train = derived.loc[derived.split.eq('train'), bundle['features']]
            expected = train.median().to_numpy()
            np.testing.assert_allclose(bundle['pipeline'].named_steps['imputer'].statistics_, expected)
            self.assertEqual(bundle['features'], FEATURES + EXTRA_FEATURES)
            heldout = derived.loc[derived.split.eq('test'), bundle['features']].head(12)
            before = bundle['pipeline'].named_steps['scaler'].mean_.copy()
            with threadpool_limits(limits=1):
                distances = bundle['pipeline'].transform(heldout)
            self.assertTrue(np.isfinite(distances).all())
            np.testing.assert_array_equal(bundle['pipeline'].named_steps['scaler'].mean_, before)


    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StageFourTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
