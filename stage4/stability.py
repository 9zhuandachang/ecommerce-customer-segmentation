"""Check k=3/4 K-means stability using training snapshots only.

Run from the project root: python stage4/stability.py
All models are refitted; the fixed evaluation records are excluded from fitting.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from threadpoolctl import threadpool_limits

from stage4.kmeans import FEATURES, RFM, load_snapshots, make_pipeline

INPUT = ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv'
OUTPUT = ROOT / 'stage4/outputs/stage4'
EVAL_SIZE = 2000
SAMPLE_FRACTION = 0.8
SEEDS = (42, 7, 23)


def align_clusters(reference: np.ndarray, labels: np.ndarray, k: int) -> dict[int, int]:
    """Map arbitrary cluster numbers to reference numbers by maximum overlap."""
    overlap = np.zeros((k, k), dtype=int)
    np.add.at(overlap, (reference, labels), 1)
    reference_ids, candidate_ids = linear_sum_assignment(-overlap)
    return dict(zip(candidate_ids.tolist(), reference_ids.tolist()))


def analyze(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if len(train) < EVAL_SIZE + 100:
        raise ValueError('Not enough training snapshots for disjoint fit/evaluation sets')
    rng = np.random.default_rng(2026)
    held_out = rng.choice(len(train), EVAL_SIZE, replace=False)
    fit_pool = train.drop(train.index[held_out]).reset_index(drop=True)
    evaluation = train.iloc[held_out].reset_index(drop=True)
    log_columns = fit_pool[FEATURES].skew().index[fit_pool[FEATURES].skew().gt(1)].tolist()
    rows, detail_rows, labels_by_run = [], [], {}
    with threadpool_limits(limits=1):
        for k in (3, 4):
            for kind, run_seeds in [('reference', (42,)), ('initialization', SEEDS[1:]),
                                    ('training_sample', SEEDS)]:
                for seed in run_seeds:
                    run_id = f'k{k}_{kind}_{seed}'
                    if kind == 'training_sample':
                        sample_rng = np.random.default_rng(seed)
                        index = sample_rng.choice(len(fit_pool), int(len(fit_pool) * SAMPLE_FRACTION), replace=False)
                        fitting = fit_pool.iloc[index]
                    else:
                        fitting = fit_pool
                    model_seed = 42 if kind == 'training_sample' else seed
                    model = make_pipeline(k, log_columns, model_seed)
                    model.fit(fitting[FEATURES])
                    predicted = model.predict(evaluation[FEATURES])
                    transformed = model[:-1].transform(evaluation[FEATURES])
                    silhouette = silhouette_samples(transformed, predicted)
                    reference = labels_by_run.get(f'k{k}_reference_42')
                    mapping = (align_clusters(reference, predicted, k) if reference is not None
                               else {i: i for i in range(k)})
                    labels_by_run[run_id] = predicted
                    aligned = np.array([mapping[int(value)] for value in predicted])
                    counts = np.bincount(model.named_steps['kmeans'].labels_, minlength=k)
                    rows.append({
                        'k': k, 'variation': kind, 'seed': seed, 'run_id': run_id,
                        'fit_snapshot_count': len(fitting), 'evaluation_snapshot_count': len(evaluation),
                        'evaluation_silhouette': float(silhouette.mean()),
                        'ari_to_reference': (adjusted_rand_score(reference, predicted)
                                             if reference is not None else 1.0),
                        'min_fit_cluster_share': float(counts.min() / len(fitting)),
                        'negative_silhouette_share': float((silhouette < 0).mean()),
                    })
                    for cluster in range(k):
                        mask = aligned == cluster
                        part = evaluation.loc[mask]
                        detail_rows.append({
                            'k': k, 'run_id': run_id, 'reference_cluster': cluster,
                            'evaluation_snapshot_count': int(mask.sum()),
                            'evaluation_unique_customer_count': int(part.customer_id.nunique()),
                            'evaluation_share': float(mask.mean()),
                            'mean_silhouette': float(silhouette[mask].mean()),
                            'negative_silhouette_share': float((silhouette[mask] < 0).mean()),
                            **{f'{feature}_median': float(part[feature].median()) for feature in RFM},
                        })
                    print(f'{run_id}: silhouette={silhouette.mean():.4f}, '
                          f'ARI={rows[-1]["ari_to_reference"]:.4f}', flush=True)
    config = {
        'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluation_rule': 'fixed random 2000 training snapshots, excluded from every fit',
        'evaluation_seed': 2026, 'training_sample_fraction': SAMPLE_FRACTION,
        'initialization_seeds': list(SEEDS), 'sampling_seeds': list(SEEDS),
        'kmeans_n_init': 20, 'kmeans_max_iter': 500,
        'log1p_features': log_columns, 'scikit_learn_version': sklearn.__version__,
        'validation_and_test_labels_used': False,
    }
    return pd.DataFrame(rows), pd.DataFrame(detail_rows), config


def write_report(runs: pd.DataFrame) -> str:
    lines = [
        '# 阶段四：三群与四群稳定性检查', '',
        '从训练期留出固定的 2,000 条客户月记录；所有重复模型都只用其余训练记录拟合。'
        '“只换随机种子”保持拟合记录相同；“换训练抽样”每次抽取剩余记录的 80%，固定 K-means 种子为 42。'
        '每次都重新拟合填补、变换、缩放和 K-means。没有使用验证期或测试期标签。', '',
        '轮廓系数在同一批留出记录上计算；ARI 衡量它们与本检查参考模型的分组一致性，1 表示完全一致，'
        '聚类编号互换不影响 ARI。参考模型也未见过这批留出记录。', '',
        '| k | 变化方式 | 留出轮廓系数范围 | 相对参考模型 ARI 范围 | 最小训练群占比范围 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for k in (3, 4):
        for kind, label in [('initialization', '只换初始化种子'), ('training_sample', '换训练抽样')]:
            part = runs.loc[(runs.k == k) & (runs.variation == kind)]
            lines.append(f'| {k} | {label} | {part.evaluation_silhouette.min():.4f}–'
                         f'{part.evaluation_silhouette.max():.4f} | {part.ari_to_reference.min():.4f}–'
                         f'{part.ari_to_reference.max():.4f} | {part.min_fit_cluster_share.min():.1%}–'
                         f'{part.min_fit_cluster_share.max():.1%} |')
    paired = runs.pivot(index=['variation', 'seed'], columns='k', values='evaluation_silhouette')
    difference = paired[3] - paired[4]
    lines += [
        '', f'同一次设置下，三群减四群的留出轮廓系数范围为 '
        f'**{difference.min():+.4f} 至 {difference.max():+.4f}**；'
        f'三群较高 {int((difference > 0).sum())}/{len(difference)} 次。',
        '', '各群留出记录数、去重客户数、平均轮廓系数、负轮廓占比及 RFM 中位数，'
        '见 [stability_cluster_details.csv](data/stability_cluster_details.csv)。'
        '这里只检查随机因素和训练样本扰动；同一客户可跨月出现，因此留出记录可能与拟合记录共享客户，'
        '这不是跨客户的泛化测试。',
        '', '完整运行参数见 [stability_config.json](stability_config.json)；'
        '逐次结果见 [stability_runs.csv](data/stability_runs.csv)。',
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    data, _ = load_snapshots(INPUT)
    train = data.loc[data.split.eq('train')].reset_index(drop=True)
    runs, details, config = analyze(train)
    output = OUTPUT
    (output / 'data').mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / 'data/stability_runs.csv', index=False, encoding='utf-8-sig')
    details.to_csv(output / 'data/stability_cluster_details.csv', index=False, encoding='utf-8-sig')
    (output / 'stability_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'stability_report.md').write_text(write_report(runs), encoding='utf-8')


if __name__ == '__main__':
    main()
