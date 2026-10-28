"""阶段四：训练期 K-means 分群、中心距离及 RFM 排序基线。

从项目根目录运行：python stage4/kmeans.py；自检：python stage4/kmeans.py --check
输出：stage4/outputs/stage4/，报告、data、figures、model 和运行记录。
仅训练期拟合；测试期只导出分群、距离和排序，
不读取测试标签计算指标，以便阶段六保留一次最终评估。
加载：bundle = joblib.load('stage4/outputs/stage4/model/segmentation_bundle.joblib')
距离：bundle['pipeline'].transform(frame[bundle['features']])
分群：bundle['pipeline'].predict(frame[bundle['features']])
滚动验证时必须在各折训练部分重新运行拟合，不可复用本次全训练期模型。
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from stage4.transforms import signed_log1p

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

FEATURES = [
    'recency_days', 'invoice_frequency_180d', 'monetary_value_180d',
    'average_order_value_180d', 'product_diversity_180d',
    'invoice_count_30d', 'invoice_count_90d', 'customer_history_days',
]
RFM = ['recency_days', 'invoice_frequency_180d', 'monetary_value_180d']
KEYS = ['snapshot_date', 'customer_id', 'split']
SPLITS = ['train', 'validation', 'test']
LABEL = 'repurchase_30d'


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')


def load_snapshots(path: Path) -> tuple[pd.DataFrame, dict]:
    """沿用上游划分，验证样本唯一性、时间顺序和跨分区标签窗口。"""
    data = pd.read_csv(path, dtype={'customer_id': 'string'})
    required = set(KEYS + FEATURES + ['use_for_modeling', LABEL])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    original_count = len(data)
    flags = data['use_for_modeling'].astype(str).str.lower()
    if not flags.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Invalid use_for_modeling flag')
    data['snapshot_date'] = pd.to_datetime(data['snapshot_date'], errors='raise')
    if data[KEYS].isna().any().any():
        raise ValueError('Missing snapshot key / split')
    if data.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate snapshot_date + customer_id')
    data = data.loc[flags.isin(['true', '1'])].copy()
    if set(data['split']) != set(SPLITS):
        raise ValueError('Expected nonempty train, validation and test partitions')
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors='raise')
    values = data[FEATURES].to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError('Features must be nonnegative finite values or missing')
    if data[RFM].isna().any().any():
        raise ValueError('R/F/M is required for eligibility, profiles and baselines')
    if not data['recency_days'].between(0, 180).all():
        raise ValueError('Recency outside the 180-day observation window')
    if (data['invoice_frequency_180d'] < 1).any():
        raise ValueError('Eligible customers must have at least one invoice')
    # 标签只作为训练/验证排序评估的结果使用，不进入聚类或预处理。
    if not data[LABEL].isin([0, 1]).all():
        raise ValueError('Expected binary repurchase_30d')
    for earlier, later in zip(SPLITS, SPLITS[1:]):
        previous = data.loc[data['split'].eq(earlier), 'snapshot_date']
        boundary = data.loc[data['split'].eq(later), 'snapshot_date'].min()
        if previous.max() >= boundary:
            raise ValueError('Temporal partitions overlap')
        if (previous + pd.Timedelta(days=30) > boundary).any():
            raise ValueError('Label window crosses the next partition; fix upstream eligibility')
    for column in ['last_purchase_datetime', 'first_purchase_datetime']:
        if column in data:
            purchase = pd.to_datetime(data[column], errors='raise')
            if purchase.isna().any() or (purchase >= data['snapshot_date']).any():
                raise ValueError(f'{column} must be strictly before snapshot_date')
    data = data.sort_values(KEYS[:2], kind='stable').reset_index(drop=True)
    audit = {
        'source_rows': original_count, 'eligible_rows': len(data),
        'excluded_by_use_for_modeling': original_count - len(data),
        'missing_features': data[FEATURES].isna().sum().to_dict(),
        'unique_snapshot_keys': True, 'temporal_order_and_label_boundary_check': True,
        'upstream_transaction_and_complete_future_window_reaudit':
            'Unavailable from snapshots alone; requires stage 2/3 transaction artifacts.',
    }
    return data, audit


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
    """客户数是各群去重人数；快照数用于聚类规模，跨月客户数不可相加。"""
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
    # 不使用复购标签命名或选 k。
    result.insert(0, 'k', k)
    return result.reset_index()


def name_clusters(profiles: pd.DataFrame) -> dict[int, str]:
    """模型拟合并形成画像后，按群间中位数排名给出可核对的描述名。"""
    result = {}
    ranks = profiles.set_index('cluster')[[col + '_median' for col in RFM]].rank(pct=True)
    for cluster, row in ranks.iterrows():
        recency = ('近期' if row.iloc[0] <= 1 / 3 else
                   '中等购买间隔' if row.iloc[0] <= 2 / 3 else '较久未购')
        frequency = '高频' if row.iloc[1] > 0.5 else '低频'
        monetary = ('低消费' if row.iloc[2] <= 1 / 3 else
                    '中等消费' if row.iloc[2] <= 2 / 3 else '高消费')
        result[int(cluster)] = f'C{cluster} {recency}/{frequency}/{monetary}'
    return result


def fit_rfm(train: pd.DataFrame) -> dict:
    """五档 RFM：切点仅用训练期分位数；相同数值永远同档。"""
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
    # 名单在每个日期的合格客户内排序，相同分数按 customer_id 文本升序稳定破同分。
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
    """只在 train / validation 计算；Top20 名额按日期分别 ceil(0.2*n)。"""
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
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'outputs' / 'stage4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--silhouette-sample-size', type=int, default=4000)
    parser.add_argument('--check', action='store_true', help='检查已生成结果，不重新训练')
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
    # 单线程使本机运算稳定，并避免 Windows K-means 的 MKL/OpenMP 过度并行问题。
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
        # 防止未来数据改变训练统计量；同一训练集单独/整批推理应完全一致。
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
    # ColumnTransformer 将 log 列放在前面，记录缩放统计时需恢复原始列序。
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
            'https://github.com/9zhuandachang/ecommerce-customer-segmentation/blob/main/docs/项目任务流程与验收清单.md',
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
    """所有输入截至快照日已知；没有完整90天可见历史时不判断节奏变化。"""
    frame = data.copy()
    frame['has_90d_history'] = (frame.customer_history_days >= 90).astype(int)
    previous60 = frame.invoice_count_90d - frame.invoice_count_30d
    if (previous60 < 0).any() or (frame.invoice_count_90d > frame.invoice_frequency_180d).any():
        raise ValueError('Nested invoice windows must satisfy 30d <= 90d <= 180d')
    frame['monthly_order_change'] = (
        frame.invoice_count_30d - previous60 / 2
    ).where(frame.has_90d_history.eq(1))
    frame['recent90_order_share'] = frame.invoice_count_90d / frame.invoice_frequency_180d
    # 首次可见购买距今天数，不是注册时长；最短按30天防止短历史被放大。
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
    # 这里只接收训练/验证，未来标签只作分群后的描述；不参与分群或命名。
    result['observed_repurchase_rate'] = grouped['repurchase_30d'].mean()
    result.insert(0, 'split', split)
    result.insert(0, 'k', k)
    return result.reset_index()


def feature_experiments(data, pipelines, base_features, sample, seed):
    """同一批训练样本比较新表示；额外在原8维空间评估避免跨空间误比。"""
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
    """中间计算保存在内存，仅将有用途的最终结果交给统一导出函数。"""
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
    """日期转换为 ISO 格式，缺失值转换为 JSON null。"""
    return json.loads(frame.to_json(orient='records', date_format='iso'))


def export_results(output, data, comparison, metrics, cluster_features, rankings,
                   split_profiles, partitions, preprocessing, bundle, results, names, manifest):
    """统一导出：四张表、一份证据记录、一个模型包和一份运行记录。"""
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
        'description': '报告中的同日消费份额、购买节奏子组和按月复购对照；不含测试标签评估。',
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
    """下方报告模板由数据生成，确保重跑不会恢复旧的术语堆叠版本。"""
    profiles = results['profiles']
    train_profiles = profiles.loc[(profiles.k == selected) & profiles.split.eq('train')].copy()
    # 报告按近期到久未购买排列，群编号保留便于对照原始导出。
    train_profiles = train_profiles.sort_values('recency_days_median')
    ids = train_profiles.cluster.astype(int).tolist()
    if selected == 3:
        names = dict(zip(ids, ['常来常买型', '低频购买型', '长时间未购型']))
    else:
        names = {int(row.cluster): f'第{int(row.cluster) + 1}组（间隔{row.recency_days_median:.0f}天）'
                 for row in train_profiles.itertuples()}
    # 保存命名依据，名称用于本报告，不改变已输出的中心编号。
    make_story_figures(output, train_profiles, results['profiles'], names, selected, results['snapshot'])
    p = train_profiles.set_index('cluster')
    validation = profiles.loc[(profiles.k == selected) & profiles.split.eq('validation')].set_index('cluster')
    high, middle, quiet = ids[0], ids[len(ids) // 2], ids[-1]
    hi, mid, low = p.loc[high], p.loc[middle], p.loc[quiet]
    latest = results['snapshot'].set_index('cluster')
    latest_date = results['snapshot'].snapshot_date.iloc[0].strftime('%Y年%m月%d日')
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
            f"六群方案还发现一对值得对比的群体：两群半年都买了约 **{a['invoice_frequency_180d_median']:.0f}次**，"
            f"金额也接近（**{a['monetary_value_180d_median']:,.0f} / {b['monetary_value_180d_median']:,.0f}**），"
            f"但距上次购买分别为 **{a['recency_days_median']:.0f}天 / {b['recency_days_median']:.0f}天**。"
            '它们过去的消费规模相似，当前所处的购买阶段却不同。这是只按消费总额分层容易忽略的差别。'
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
            f'{row.recency_days_median:.0f}天', f'{row.invoice_frequency_180d_median:.0f}次',
            f'{row.monetary_value_180d_median:,.0f}',
        ])
    comparison_details = {
        3: '建立总体轮廓：常来常买、低频购买、长时间未购。',
        4: f'出现“高单笔、低频”群：典型单笔{big_ticket.average_order_value_180d_median:.0f}，半年买{big_ticket.invoice_frequency_180d_median:.0f}次，距上次购买{big_ticket.recency_days_median:.0f}天。',
        5: f'识别一批低消费、购买商品种类少的客户：半年金额{cheapest5.monetary_value_180d_median:.0f}，商品种类{cheapest5.product_diversity_180d_median:.0f}种。',
        6: f'高频客户继续细分：最频繁一群半年买{biggest6.invoice_frequency_180d_median:.0f}次，占{biggest6.snapshot_share:.1%}；也分开了消费接近、近期活跃不同的群体。',
    }
    comparison_rows = [[f'{int(r.k)}群', f'{r.silhouette_train:.4f}', f'{r.min_cluster_share:.1%}',
                        comparison_details[int(r.k)]] for r in comparison.itertuples()]
    k4_rows = []
    for title, row in [('频繁购买型', frequent), ('高单笔、低频型', big_ticket),
                       ('低频、可见历史较短型', light4), ('长时间未购型', quiet4)]:
        k4_rows.append([f'{title}（C{int(row.cluster)}）', f'{row.snapshot_share:.1%}',
                        f'{row.recency_days_median:.0f}天', f'{row.invoice_frequency_180d_median:.0f}次',
                        f'{row.average_order_value_180d_median:,.0f}'])
    followup_rows = [[names[c], f'{p.loc[c, "observed_repurchase_rate"]:.1%}',
                     f'{validation.loc[c, "observed_repurchase_rate"]:.1%}'] for c in ids]
    monthly = results['monthly'].loc[results['monthly'].split.eq('validation')]
    ranges = monthly.groupby('cluster').observed_repurchase_rate.agg(['min', 'max'])
    lines = [
        '# 客户分群：谁在常买，谁只是偶尔来，谁已经很久没买？', '',
        '> 本报告要回答的是客户之间有哪些值得关注的差别，以及这些差别能给后续分析带来什么。', '',
        '## 先看最值得讲的三个发现', '',
        f'1. **常买的人不一定每笔都特别贵。** 常来常买群的半年购买次数约是低频群的 '
        f'{hi.invoice_frequency_180d_median / mid.invoice_frequency_180d_median:.1f}倍，'
        f'但典型单笔金额仅约 {hi.average_order_value_180d_median / mid.average_order_value_180d_median:.1f}倍。购买频次是两群之间很突出的差别。',
        f'2. **单笔花得多，近期也未必活跃。** 分成四群时，有一群典型单笔金额为 '
        f'{big_ticket.average_order_value_180d_median:,.0f}，高于最频繁购买群的 '
        f'{frequent.average_order_value_180d_median:,.0f}；但距上次购买分别为 '
        f'{big_ticket.recency_days_median:.0f}天和{frequent.recency_days_median:.0f}天。',
        '3. **最近买得多，不等于下个月一定继续买。** 新增的“购买节奏变化”特征发现了一个反直觉现象，后文给出具体对照。', '',
        '阅读口径：一个客户每个月记一条记录，称为“客户月记录”。同一个人可能在不同月份属于不同群体。'
        '除特别说明外，画像使用训练期（2010年6月—2011年5月）记录；“典型值”指中位数，也就是排序后居中的数。金额沿用原始 CSV 的单位。', '',
        '## 1. 分成三群，分别是什么样的人？', '',
        '**k 就是想分成几群。k=3 表示分成三群；C0、C1、C2 只是编号，没有高低顺序。**', '',
        md_table(['群体', '记录占比', '距上次购买', '半年购买次数', '半年消费金额'], group_rows), '',
        '![三类客户的购买行为对比](figures/customer_group_story.png)', '',
        f'### {names[high]}：核心特点是常买，买的商品也更多', '',
        f'这群人典型的半年购买次数为 **{hi.invoice_frequency_180d_median:.0f}次**，'
        f'购买过 **{hi.product_diversity_180d_median:.0f}种商品**；低频群分别为 '
        f'{mid.invoice_frequency_180d_median:.0f}次、{mid.product_diversity_180d_median:.0f}种。'
        f'典型单笔金额为 **{hi.average_order_value_180d_median:,.0f}**，低频群为 **{mid.average_order_value_180d_median:,.0f}**。'
        '这更像一群持续购买、接触商品较广的客户。单笔金额是每位客户平均订单金额的中位数，不能把几个中位数直接相乘。', '',
        f'再看一个不重复计算客户的例子：**{latest_date}**，这群只有 '
        f'**{int(latest.loc[high, "customers"]):,}人**，占当日客户 **{latest.loc[high, "customer_share"]:.1%}**，'
        f'却占这些客户过去180天消费金额的 **{latest.loc[high, "spend_share"]:.1%}**。'
        '这个比例说明维护其购买连续性值得关注；它是该日客户范围内的消费份额。', '',
        f'### {names[middle]}：人数多，尚未形成频繁购买习惯', '',
        f'这群占全部记录 **{mid.snapshot_share:.1%}**，其中 **{mid.one_invoice_180d_mean:.1%}** '
        f'在过去半年只买过一次；**{mid.recently_observed_30d_mean:.1%}** 的首次可见购买在最近30天内。'
        '因此，这一群同时包含最近才出现在数据中的客户，以及买得不频繁的老客户。', '',
        '**可以继续追问：** 对只有一次购买记录的人，哪些历史行为与第二次购买相关？'
        '“首次可见购买”是数据里的第一次购买记录，不能直接当作真实注册时间。', '',
        f'### {names[quiet]}：主要区别是沉默时间长', '',
        f'这群距上次购买典型已有 **{low.recency_days_median:.0f}天**，'
        f'**{low.inactive_90d_mean:.1%}** 的记录在过去90天没有购买。'
        f'但其典型单笔金额 **{low.average_order_value_180d_median:,.0f}**，'
        f'与低频群的 **{mid.average_order_value_180d_median:,.0f}** 接近。'
        '用“长时间未购买”描述它，比只贴“低价值”标签更准确。', '',
        '**可以继续追问：** 长时间未购的人是否存在不同的补货周期？现有快照没有完整订单间隔和商品类别，暂时只能提出这个待验证问题。', '',
        '## 2. 为什么还要看四群、五群、六群？', '',
        '多分几群，是把画像看得更细。它会重新划分全部客户，所以四群方案不等于简单地把三群中的某一群劈开。', '',
        '表里的“分离程度”对应轮廓系数：越高，表示同群客户更相似、不同群体更容易分开。它不是准确率。', '',
        md_table(['方案', '分离程度', '最小群占比', '比总览多看到了什么'], comparison_rows), '',
        f'**当前主模型沿用 {selected} 群。** 三群与四群得分只差 '
        f'{abs(float(comparison.loc[comparison.k.eq(3), "silhouette_train"].iloc[0]) - float(comparison.loc[comparison.k.eq(4), "silhouette_train"].iloc[0])):.4f}，'
        '所以不能说三群明显胜出。三群适合讲总体结构；四群提供了有价值的业务细节，值得在展示中保留。五群、六群可作为更细的观察视角，分离程度并未持续改善。', '',
        '### 四群方案中，值得单独讲的“高单笔、低频”群', '',
        md_table(['四群方案中的群体', '记录占比', '距上次购买', '半年购买次数', '典型单笔金额'], k4_rows), '',
        f'“高单笔、低频型”占 **{big_ticket.snapshot_share:.1%}**，典型半年消费金额为 '
        f'**{big_ticket.monetary_value_180d_median:,.0f}**。这提示我们：消费金额高可以来自“买得频繁”，也可以来自“每笔买得较多”。'
        '两类客户是否适合相同的维护节奏，需要进一步研究。', '',
        pair_text, '',
        '## 3. 新挖掘了哪些特征？', '',
        '原来的特征告诉我们“买了多少”。这次补充“最近有没有改变、在有限历史里买得有多频繁”。'
        '这些特征由现有字段计算，属于新的表达方式；真实订单间隔、品类偏好、退货偏好仍需要交易明细。', '',
        md_table(['新增特征', '通俗含义与算法', '怎么理解'], [
            ['购买节奏变化', '最近30天购买次数 − 再往前60天的月均购买次数',
             '例如最近30天买3次、前60天共买2次，变化为 +2。只给已有至少90天可见历史的人计算。'],
            ['近期购买占比', '最近90天购买次数 ÷ 最近180天购买次数',
             '接近1：半年内的购买集中在较近一段；为0：最近90天没有买。短历史客户天然可能偏高。'],
            ['按可见历史折算的月购次数', '半年购买次数 ÷ 可见历史天数 × 30；天数限定30—180天',
             '同样买2次，出现了两个月与出现了半年，购买频率应区别看待。'],
            ['历史是否足够长', '是否已有至少90天可见购买历史',
             '提示节奏变化是否有足够观察时间，避免把缺少历史当成没有购买。'],
        ]), '',
        '另外增加了两个便于找人的标记：**近期加速**＝节奏变化至少增加1次/月；'
        '**重复购买后暂时中断**＝已有90天可见历史、过去90天买过至少2次，但最近30天没有买。', '',
        '### 新特征真的让分群更好吗？实际试了一遍', '',
        f'将上述4项加入原有8项后，再比较3—6群。新增特征方案选择了 **{int(selected_experiment.k)}群**。'
        f'它单独识别出一个占 **{short_history.snapshot_share:.1%}** 的群体，'
        f'首次可见购买距今典型只有 **{short_history.customer_history_days_median:.0f}天**，'
        f'其中 **{1 - short_history.has_90d_history_mean:.1%}** 的可见历史不足90天。'
        '说明新增特征让模型更强调“客户处在什么观察阶段”。', '',
        f'用新增后的12项特征衡量，新方案得分 **{selected_experiment.new_space_silhouette:.4f}**。'
        '但衡量的“尺子”变了，不能直接拿这个数与原来的得分比较。'
        f'重新用原来8项特征衡量同为 {selected} 群的新分法，得分为 **{new_same_space:.4f}**，'
        f'原方案为 **{base_score:.4f}**。'
        '因此，现阶段的收获是找到了新的观察维度，还没有证据说明应该用新方案替换原模型。', '',
        '## 4. 这些特点换一段时间还成立吗？', '',
        '用原模型给2011年6—8月的客户分群，再看他们之后30天实际有没有再次购买。以下结果只用于事后检查画像，未来购买记录没有参与分群。', '',
        md_table(['群体', '训练期实际复购比例', '验证期实际复购比例'], followup_rows), '',
        '![各群体在下一段时间的实际复购对照](figures/cluster_followup_story.png)', '',
        f'常来常买群在三个验证月份的复购比例为 **{ranges.loc[high, "min"]:.1%}—{ranges.loc[high, "max"]:.1%}**，'
        f'长时间未购群为 **{ranges.loc[quiet, "min"]:.1%}—{ranges.loc[quiet, "max"]:.1%}**。'
        '这种差异在三个验证月份都存在，说明画像有后续行为上的对应。也说明“很久没买”不等于以后再也不会买。', '',
        '### 一个反直觉发现：近期加速，不一定持续加速', '',
    ]
    if all(row is not None for row in acceleration):
        yes, no = acceleration
        lines += [
            f'在 **{names[high]}** 内，只比较可见历史满90天的记录：', '',
            md_table(['验证期子组', '客户月记录数', '之后30天实际复购比例'], [
                ['最近购买次数明显增加', f'{int(yes.snapshot_count):,}', f'{yes.observed_repurchase_rate:.1%}'],
                ['没有达到上述增加幅度', f'{int(no.snapshot_count):,}', f'{no.observed_repurchase_rate:.1%}'],
            ]), '',
            '这个结果提醒我们：购买有节奏，最近的集中购买未必会持续到下个月。'
            '是否与集中补货、季节性或客户构成有关，当前数据还不能确认。可以把“购买节奏变化”交给阶段五模型检验，而不能直接写成“越活跃越容易复购”。', '',
        ]
    if all(row is not None for row in pausing):
        yes, no = pausing
        lines += [
            '### 另一个细节：暂时没买的人，也可能仍有购买基础', '',
            f'在 **{names[middle]}** 中，“重复购买后暂时中断”的记录有 '
            f'**{int(yes.snapshot_count):,}条**，验证期之后30天复购比例为 **{yes.observed_repurchase_rate:.1%}**；'
            f'同群其余可见历史满90天的记录为 **{no.observed_repurchase_rate:.1%}**（{int(no.snapshot_count):,}条）。', '',
            '这个标记同时包含“过去买过不止一次”和“最近暂停”，所以较高复购率可能来自既有的购买基础。'
            '它适合用来区分客户状态，尚不能据此把“暂停”解释为复购的原因。', '',
        ]
    lines += [
        '以上属于探索性观察。同一客户可贡献多个月份记录，不能当作独立样本做普通显著性检验；'
        '这里的“突出特点”指可观察的差异，没有声称统计显著或营销措施有效。测试期仍留给最终评估。', '',
        '## 5. 简单排序给后续预测模型定了什么参照？', '',
        '假设每个月只能查看20%的客户：RFM 同时看“最近买没买、买过几次、花了多少”，作为后续预测模型的简单排序参照。', '',
        md_table(['排序方法', '验证期共选出的记录', '其中之后30天再次购买', '名单内复购比例'], [
            ['近期、次数和金额（RFM）', f'{int(rfm.top20_count):,}', f'{int(rfm.top20_hits):,}', f'{rfm.precision_at_20pct:.1%}'],
        ]), '',
        '后续复杂模型应该与这个简单办法比较，确认增加复杂度是否值得。这里是历史名单的回看，不代表营销活动带来了这些购买。', '',
        '## 6. 汇报时可以怎么说？', '',
        '> 我们先用三群抓住总体结构：常来常买、低频购买和长时间未购。'
        '然后把分群细化，发现高消费既可能来自高频购买，也可能来自较高的单笔金额；消费相似的人，近期活跃程度也可能不同。'
        '进一步构造购买节奏和可见历史特征后，发现近期加速不一定延续到下个月。'
        '因此，客户分群的价值是把这些差别说清楚，为后续复购预测提出可检验的问题。', '',
        '## 数据文件与复现', '',
        f'- 全部合格快照 **{len(data):,}条**；训练期用于发现分群，验证期用于后续行为对照。',
        '- 在项目根目录运行 `python stage4/kmeans.py` 重新生成分析与报告；运行 `python stage4/kmeans.py --check` 检查已有结果。',
        '- `data/` 放结果表，`figures/` 放正文引用的两张图，`model/` 放可加载模型。', '',
        md_table(['文件', '内容与用途'], [
            ['[客户结果总表](data/customer_results.csv)', '每个客户每月一行：原始特征、新特征、分群、中心距离、两种排序和前20%标记。'],
            ['[群体画像汇总](data/cluster_profiles.csv)', '原始与新增特征方案的各群体画像；按 feature_set、k、split 筛选。'],
            ['[分群方案比较](data/model_comparison.csv)', '原始8项和新增后12项特征的 k=3—6 比较，合计8个方案。'],
            ['[排序基线表现](data/baseline_metrics.csv)', '训练、验证期的逐月与汇总排序表现。'],
            ['[行为发现的依据](data/behavior_evidence.json)', '同日消费份额、购买节奏子组、按月复购对照，合在一份记录中。'],
            ['[模型包](model/segmentation_bundle.joblib)', '主模型、RFM切点、新特征探索模型，共用一个文件。'],
            ['[运行记录](run_manifest.json)', '输入哈希、版本、配置、日期划分和预处理统计。'],
        ]), '',
        '画像表中，`feature_set=original_8` 是原始方案，`enhanced_12` 是新特征探索方案；'
        '`is_main_model=True` 是正文使用的主模型。`split` 区分训练、验证和测试；'
        '`snapshot_count` 是客户月记录数，`snapshot_share` 是占比，`_median` 是中位数，`_mean` 是均值。'
        '测试期的实际复购比例留空。', '',
        '<details>', '<summary>方法与模型使用说明（点击展开）</summary>', '',
        '**训练边界。** 沿用上游时间划分，填补、变换和缩放只在训练期拟合；客户ID、日期和未来标签不作分群输入。'
        '原始交易与最后一期观察是否完整，仍需阶段二、三的交易明细佐证。', '',
        f'**分群方法。** 原始8项特征以训练期中位数填补，偏度大于1的列做 log1p，再统一尺度。'
        f'四个 k 使用同一批 {manifest["silhouette_sample_size"]:,} 条训练记录比较分离程度；'
        f'随机种子 {manifest["seed"]}，每个方案尝试20次初始中心。先要求最小群占比达到5%，再选择分离程度最高的方案。'
        '新特征方案允许购买节奏变化为负数，使用保留正负号的对数变换，独立拟合。', '',
        '**排序口径。** 最近购买时间分数为距上次购买天数的相反数。RFM按训练期五分位切点计分，'
        'R反向、F/M正向，各1—5分后相加；相同值同档。每个日期分别取前20%（人数向上取整），'
        '同分按客户ID文本升序。汇总名单表现采用逐月入选记录的合计；分数是排序依据。', '',
        '**加载模型。** 在项目根目录运行：', '',
        '```python', 'import joblib',
        "bundle = joblib.load('stage4/outputs/stage4/model/segmentation_bundle.joblib')",
        "X = snapshots[bundle['features']]",
        "clusters = bundle['pipeline'].predict(X)",
        "distances = bundle['pipeline'].transform(X)",
        "# 探索模型位于 bundle['exploratory']，输入先用 stage4.kmeans.derive_features 生成。", '```', '',
        '主模型中心距离位于统一尺度后的特征空间，可供阶段五使用；滚动验证应在各折训练期重新拟合。'
        '模型包中的探索模型使用 stage4.transforms 中可导入的变换函数。'
        '依赖版本记录在 run_manifest.json。', '',
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
                ['多久没买了？', '半年买了几次？', '半年花了多少？'], ['天', '次', '金额单位']):
            vals = profiles[col].to_numpy()
            ax.barh(labels, vals, color=colors[:len(ids)])
            ax.invert_yaxis()
            ax.set_title(title, fontweight='bold', pad=12)
            ax.set_xlabel(unit)
            ax.set_xlim(0, vals.max() * 1.3)
            for i, value in enumerate(vals):
                ax.text(value + vals.max() * .025, i, f'{value:,.0f}', va='center')
        fig.suptitle('三类客户，最直观的区别' if selected == 3 else '各群体的典型购买行为', fontsize=17)
        fig.savefig(output / 'figures' / 'customer_group_story.png', dpi=170, facecolor='white')
        plt.close(fig)
        validation = all_profiles.loc[(all_profiles.k == selected) & all_profiles.split.eq('validation')].set_index('cluster')
        fig, ax = plt.subplots(figsize=(8.6, 4.3), layout='constrained')
        x = np.arange(len(ids))
        train = profiles.set_index('cluster').loc[ids, 'observed_repurchase_rate'] * 100
        valid = validation.loc[ids, 'observed_repurchase_rate'] * 100
        for shift, values, label, color in [(-.18, train, '训练期', '#93BCD0'), (.18, valid, '验证期', '#227C9D')]:
            bars = ax.bar(x + shift, values, .34, label=label, color=color)
            ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=4)
        ax.set_xticks(x, labels)
        ax.set_ylabel('未来30天实际再次购买的比例（%）')
        ax.set_ylim(0, max(train.max(), valid.max()) * 1.28)
        ax.set_title('换一段时间，群体之间的差异仍然存在', fontsize=16, pad=15)
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
            # 输入行顺序不能改变同一客户的分数、名次或入选状态。
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
