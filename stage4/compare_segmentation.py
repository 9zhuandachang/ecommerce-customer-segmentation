"""Compare three customer-state clusterers without changing the legacy model."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import BisectingKMeans, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv'
OUTPUT = ROOT / 'stage4/outputs/segmentation_comparison'
PROTOCOL = ROOT / 'stage4/segmentation_comparison_protocol.md'
RAW_COLUMNS = (
    'recency_days', 'invoice_frequency_180d', 'average_order_value_180d',
    'product_diversity_180d', 'invoice_count_90d',
)
FEATURES = (
    'recency_days', 'invoice_frequency_180d', 'average_order_value_180d',
    'products_per_invoice_180d', 'recent90_invoice_share',
)
LOG_COLUMNS = (
    'invoice_frequency_180d', 'average_order_value_180d',
    'products_per_invoice_180d',
)
K_VALUES = range(2, 7)
HOLDOUT_SEED = 20261027
RESAMPLE_SEEDS = (3101, 3102, 3103, 3104, 3105)
MIN_MONTHLY_GROUP = 50
SILHOUETTE_SAMPLE = 2000


@dataclass(frozen=True)
class Candidate:
    algorithm: str
    k: int
    covariance_type: str = ''

    @property
    def name(self) -> str:
        suffix = f'_{self.covariance_type}' if self.covariance_type else ''
        return f'{self.algorithm}_k{self.k}{suffix}'


def candidates() -> list[Candidate]:
    return (
        [Candidate('kmeans', k) for k in K_VALUES]
        + [Candidate('bisecting_kmeans', k) for k in K_VALUES]
        + [Candidate('gmm', k, covariance) for covariance in ('diag', 'full')
           for k in K_VALUES]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_snapshots() -> pd.DataFrame:
    fields = ['snapshot_date', 'customer_id', 'split', 'use_for_modeling',
              *RAW_COLUMNS, 'monetary_value_180d', 'customer_history_days']
    data = pd.read_csv(SNAPSHOTS, usecols=fields,
                       dtype={'customer_id': 'string'}, parse_dates=['snapshot_date'])
    if data[['snapshot_date', 'customer_id', 'split']].isna().any().any():
        raise ValueError('Missing snapshot key or split')
    if data.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate customer-month key')
    flags = data.use_for_modeling.astype(str).str.lower()
    if not flags.isin(('true', 'false', '1', '0')).all():
        raise ValueError('Invalid use_for_modeling flag')
    data = data.loc[flags.isin(('true', '1'))].copy()
    expected = {'train': ('2010-06-01', '2011-05-01'),
                'validation': ('2011-06-01', '2011-08-01'),
                'test': ('2011-09-01', '2011-11-01')}
    if set(data.split) != set(expected):
        raise ValueError('Unexpected partition names')
    for split, (start, end) in expected.items():
        dates = data.loc[data.split.eq(split), 'snapshot_date']
        if (dates.min(), dates.max()) != (pd.Timestamp(start), pd.Timestamp(end)):
            raise ValueError(f'Unexpected date range for {split}')
    if data[list(RAW_COLUMNS)].isna().any().any():
        raise ValueError('Clustering inputs contain missing values')
    return data.sort_values(['snapshot_date', 'customer_id'], kind='stable').reset_index(drop=True)


def behavior_features(data: pd.DataFrame) -> pd.DataFrame:
    raw = data[list(RAW_COLUMNS)].astype(float)
    if not np.isfinite(raw.to_numpy()).all() or raw.lt(0).any().any():
        raise ValueError('Clustering inputs must be finite and nonnegative')
    frequency = raw.invoice_frequency_180d
    recent = raw.invoice_count_90d
    if frequency.lt(1).any() or recent.gt(frequency).any():
        raise ValueError('Invalid 180-day or 90-day invoice counts')
    result = pd.DataFrame({
        'recency_days': raw.recency_days,
        'invoice_frequency_180d': frequency,
        'average_order_value_180d': raw.average_order_value_180d,
        'products_per_invoice_180d': raw.product_diversity_180d / frequency,
        'recent90_invoice_share': recent / frequency,
    }, index=data.index)
    if not np.isfinite(result.to_numpy()).all() or result.lt(0).any().any():
        raise ValueError('Derived features must be finite and nonnegative')
    if result.recent90_invoice_share.gt(1).any():
        raise ValueError('Recent share cannot exceed one')
    return result[list(FEATURES)]


def transformed_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    result[list(LOG_COLUMNS)] = np.log1p(result[list(LOG_COLUMNS)])
    return result


def make_model(spec: Candidate, seed: int):
    if spec.algorithm == 'kmeans':
        return KMeans(n_clusters=spec.k, n_init=20, max_iter=500,
                      random_state=seed)
    if spec.algorithm == 'bisecting_kmeans':
        return BisectingKMeans(n_clusters=spec.k, init='k-means++',
                               n_init=10, bisecting_strategy='biggest_inertia',
                               random_state=seed)
    if spec.algorithm == 'gmm' and spec.covariance_type in ('diag', 'full'):
        return GaussianMixture(n_components=spec.k,
                               covariance_type=spec.covariance_type,
                               n_init=5, reg_covar=1e-4, max_iter=300,
                               random_state=seed)
    raise ValueError(f'Unknown candidate: {spec}')


def fit_model(spec: Candidate, features: pd.DataFrame, seed: int) -> dict:
    scaler = StandardScaler()
    x = scaler.fit_transform(transformed_features(features))
    model = make_model(spec, seed)
    with threadpool_limits(limits=4):
        model.fit(x)
    if spec.algorithm == 'gmm' and not model.converged_:
        raise ValueError(f'{spec.name} failed to converge')
    return {'candidate': asdict(spec), 'features': list(FEATURES),
            'log_columns': list(LOG_COLUMNS), 'scaler': scaler, 'model': model}


def transformed_for_bundle(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    return bundle['scaler'].transform(transformed_features(features))


def assign(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    x = transformed_for_bundle(bundle, features)
    return bundle['model'].predict(x).astype(int)


def customer_holdout(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    customers = np.sort(train.customer_id.unique().astype(str))
    rng = np.random.default_rng(HOLDOUT_SEED)
    held = set(rng.choice(customers, size=int(round(len(customers) * .20)),
                              replace=False))
    hold_mask = train.customer_id.isin(held).to_numpy()
    if not hold_mask.any() or hold_mask.all():
        raise ValueError('Invalid customer holdout')
    return np.flatnonzero(~hold_mask), np.flatnonzero(hold_mask)


def one_state_per_customer(holdout: pd.DataFrame) -> pd.DataFrame:
    """Keep one fixed historical state for each held-out customer."""
    result = holdout.groupby('customer_id', sort=True, observed=True).sample(
        n=1, random_state=HOLDOUT_SEED)
    return result.sort_values('customer_id', kind='stable').reset_index(drop=True)


def monthly_sizes(data: pd.DataFrame, labels: np.ndarray,
                  spec: Candidate) -> pd.DataFrame:
    frame = data[['snapshot_date', 'customer_id']].copy()
    frame['cluster'] = labels
    rows = []
    for date, month in frame.groupby('snapshot_date', sort=True):
        counts = month.cluster.value_counts().reindex(range(spec.k), fill_value=0)
        for cluster, count in counts.items():
            rows.append({'candidate': spec.name, 'snapshot_date': date.date().isoformat(),
                         'cluster': int(cluster), 'customers': int(count),
                         'share': float(count / len(month))})
    return pd.DataFrame(rows)


def profiles(data: pd.DataFrame, features: pd.DataFrame,
             labels: np.ndarray, spec: Candidate, period: str) -> pd.DataFrame:
    frame = pd.concat([data[['snapshot_date', 'customer_id',
                             'monetary_value_180d', 'customer_history_days']],
                       features], axis=1).copy()
    frame['cluster'] = labels
    rows = []
    for (date, cluster), group in frame.groupby(['snapshot_date', 'cluster'], sort=True):
        row = {'period': period, 'candidate': spec.name,
               'snapshot_date': date.date().isoformat(), 'cluster': int(cluster),
               'customers': int(group.customer_id.nunique())}
        for column in [*FEATURES, 'monetary_value_180d', 'customer_history_days']:
            row[column + '_median'] = float(group[column].median())
        rows.append(row)
    return pd.DataFrame(rows)


def compare() -> None:
    data = load_snapshots()
    train = data.loc[data.split.eq('train')].reset_index(drop=True)
    validation = data.loc[data.split.eq('validation')].reset_index(drop=True)
    train_features = behavior_features(train)
    validation_features = behavior_features(validation)
    fit_indices, hold_indices = customer_holdout(train)
    fit = train.iloc[fit_indices].reset_index(drop=True)
    hold = one_state_per_customer(train.iloc[hold_indices].reset_index(drop=True))
    fit_features = train_features.iloc[fit_indices].reset_index(drop=True)
    hold_features = behavior_features(hold)
    if set(fit.customer_id) & set(hold.customer_id):
        raise AssertionError('Training and structure holdout share customers')
    rng = np.random.default_rng(HOLDOUT_SEED + 1)
    sample = np.sort(rng.choice(len(hold), size=min(SILHOUETTE_SAMPLE, len(hold)),
                                replace=False))
    fit_customers = np.sort(fit.customer_id.unique().astype(str))
    rows, all_sizes, all_profiles = [], [], []
    for spec in candidates():
        bundle = fit_model(spec, fit_features, seed=42)
        hold_labels = assign(bundle, hold_features)
        val_labels = assign(bundle, validation_features)
        fit_labels = assign(bundle, fit_features)
        if np.unique(fit_labels).size != spec.k:
            raise ValueError(f'{spec.name} formed fewer groups than requested')
        sizes = monthly_sizes(validation, val_labels, spec)
        all_sizes.append(sizes)
        all_profiles.append(profiles(validation, validation_features,
                                     val_labels, spec, 'validation'))
        x_hold = transformed_for_bundle(bundle, hold_features)
        score = (float(silhouette_score(x_hold[sample], hold_labels[sample]))
                 if np.unique(hold_labels[sample]).size > 1 else float('nan'))
        stability = []
        for seed in RESAMPLE_SEEDS:
            sample_customers = set(np.random.default_rng(seed).choice(
                fit_customers, size=int(round(len(fit_customers) * .80)),
                replace=False))
            sampled = fit.customer_id.isin(sample_customers).to_numpy()
            repeated = fit_model(spec, fit_features.loc[sampled].reset_index(drop=True),
                                 seed=seed)
            stability.append(float(adjusted_rand_score(
                hold_labels, assign(repeated, hold_features))))
        model = bundle['model']
        row = {'candidate': spec.name, 'algorithm': spec.algorithm, 'k': spec.k,
               'covariance_type': spec.covariance_type,
               'fit_customer_count': int(fit.customer_id.nunique()),
               'holdout_customer_count': int(hold.customer_id.nunique()),
               'holdout_silhouette': score,
               'stability_ari_median': float(np.median(stability)),
               'stability_ari_min': float(min(stability)),
               'stability_ari_max': float(max(stability)),
               'validation_min_monthly_group': int(sizes.customers.min()),
               'validation_min_monthly_share': float(sizes.share.min()),
               'size_gate_passed': bool(sizes.customers.ge(MIN_MONTHLY_GROUP).all()),
               'gmm_bic_fit_only': float(model.bic(transformed_for_bundle(
                   bundle, fit_features))) if spec.algorithm == 'gmm' else np.nan,
               'gmm_mean_max_membership': float(np.mean(np.max(model.predict_proba(
                   transformed_for_bundle(bundle, validation_features)), axis=1)))
               if spec.algorithm == 'gmm' else np.nan,
               'model_converged': bool(model.converged_) if spec.algorithm == 'gmm' else True,
            }
        rows.append(row)
        print(f'{spec.name}: silhouette={score:.4f}; ARI={row["stability_ari_median"]:.4f}; '
              f'min monthly group={row["validation_min_monthly_group"]}', flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {'candidates': OUTPUT / 'candidates.csv',
             'monthly_sizes': OUTPUT / 'monthly_sizes.csv',
             'validation_profiles': OUTPUT / 'validation_profiles.csv'}
    pd.DataFrame(rows).to_csv(paths['candidates'], index=False, encoding='utf-8-sig')
    pd.concat(all_sizes, ignore_index=True).to_csv(
        paths['monthly_sizes'], index=False, encoding='utf-8-sig')
    pd.concat(all_profiles, ignore_index=True).to_csv(
        paths['validation_profiles'], index=False, encoding='utf-8-sig')
    manifest = {'snapshot_sha256': sha256(SNAPSHOTS),
                'protocol_sha256': sha256(PROTOCOL),
                'script_sha256': sha256(Path(__file__)),
                'holdout_seed': HOLDOUT_SEED,
                'resample_seeds': list(RESAMPLE_SEEDS),
                'silhouette_sample_size': len(sample),
                'fit_rows': len(fit), 'holdout_rows': len(hold),
                'validation_rows': len(validation),
                'outputs_sha256': {name: sha256(path) for name, path in paths.items()}}
    (OUTPUT / 'comparison_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def select(spec: Candidate) -> None:
    if spec not in candidates():
        raise ValueError('Selection must be one of the predeclared candidates')
    manifest = json.loads((OUTPUT / 'comparison_manifest.json').read_text(encoding='utf-8'))
    if (sha256(SNAPSHOTS) != manifest['snapshot_sha256']
            or sha256(PROTOCOL) != manifest['protocol_sha256']
            or sha256(Path(__file__)) != manifest['script_sha256']):
        raise ValueError('Comparison input or protocol changed after comparison')
    for name, digest in manifest['outputs_sha256'].items():
        if sha256(OUTPUT / f'{name}.csv') != digest:
            raise ValueError(f'Comparison output changed: {name}')
    comparison = pd.read_csv(OUTPUT / 'candidates.csv')
    candidate = comparison.loc[comparison.candidate.eq(spec.name)]
    if len(candidate) != 1 or not bool(candidate.iloc[0].size_gate_passed):
        raise ValueError('Candidate is missing or failed the monthly size gate')
    data = load_snapshots()
    train = data.loc[data.split.eq('train')].reset_index(drop=True)
    bundle = fit_model(spec, behavior_features(train), seed=42)
    assignments, profile_rows = [], []
    for split in ('train', 'validation', 'test'):
        part = data.loc[data.split.eq(split)].reset_index(drop=True)
        features = behavior_features(part)
        labels = assign(bundle, features)
        result = part[['snapshot_date', 'customer_id', 'split']].copy()
        result['cluster'] = labels
        if spec.algorithm == 'gmm':
            probabilities = bundle['model'].predict_proba(
                transformed_for_bundle(bundle, features))
            result['max_membership'] = probabilities.max(axis=1)
        assignments.append(result)
        profile_rows.append(profiles(part, features, labels, spec, split))
    model_path = OUTPUT / 'selected_model.joblib'
    joblib.dump(bundle, model_path, compress=3)
    assignments_path = OUTPUT / 'selected_assignments.csv'
    pd.concat(assignments, ignore_index=True).to_csv(
        assignments_path, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
    profiles_path = OUTPUT / 'selected_profiles.csv'
    pd.concat(profile_rows, ignore_index=True).to_csv(
        profiles_path, index=False, encoding='utf-8-sig')
    decision = {'selected': asdict(spec), 'candidate': spec.name,
                'features': list(FEATURES), 'log_columns': list(LOG_COLUMNS),
                'model_parameters': bundle['model'].get_params(deep=False),
                'preprocessing': 'log1p on log_columns, then StandardScaler fitted on training rows',
                'fit_rows': len(train),
                'source_scope': 'historical_reanalysis_not_new_independent_test',
                'downstream_integrated': False,
                'comparison_manifest_sha256': sha256(OUTPUT / 'comparison_manifest.json'),
                'model_sha256': sha256(model_path),
                'assignments_sha256': sha256(assignments_path),
                'profiles_sha256': sha256(profiles_path)}
    (OUTPUT / 'selected.json').write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Selected {spec.name}; model saved to {model_path}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    sub.add_parser('compare')
    choose = sub.add_parser('select')
    choose.add_argument('--algorithm', choices=('kmeans', 'bisecting_kmeans', 'gmm'),
                        required=True)
    choose.add_argument('--k', type=int, choices=K_VALUES, required=True)
    choose.add_argument('--covariance-type', choices=('', 'diag', 'full'), default='')
    args = parser.parse_args()
    if args.mode == 'compare':
        compare()
    else:
        select(Candidate(args.algorithm, args.k, args.covariance_type))


if __name__ == '__main__':
    main()
