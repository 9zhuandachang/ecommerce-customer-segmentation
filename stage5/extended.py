"""Stage-five extension: behavior features, rolling backtests and small GBDT tuning.

Run from the project root: python stage5/extended.py
The original six-experiment outputs remain in stage5/outputs/stage5/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from src.modelling import build_gbdt_pipeline, build_lr_pipeline
from stage4.kmeans import FEATURES, KEYS, LABEL, baseline_scores, fit_rfm, load_snapshots, make_pipeline
from stage4.kmeans import EXTRA_FEATURES, derive_features
from stage5.experiments import add_distances, evaluate, top20_metrics

INPUT = ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv'
STAGE4 = ROOT / 'stage4/outputs/stage4'
ORIGINAL = ROOT / 'stage5/outputs/stage5'
OUTPUT = ROOT / 'stage5/outputs/extended'
FOLD_DATES = pd.to_datetime(['2011-03-01', '2011-04-01', '2011-05-01'])
FEATURE_SETS = {
    'A_base_8': FEATURES,
    'B_base_8_plus_3_distances': FEATURES + [f'cluster_distance_{i}' for i in range(3)],
    'C_base_8_plus_4_behavior': FEATURES + EXTRA_FEATURES,
}
LEAF_OPTIONS = (15, 31)
MAX_TREES = 500
EARLY_STOPPING_ROUNDS = 30
WEIGHTS = (0.25, 0.50, 0.75)


def make_fold(train: pd.DataFrame, validation_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A training label's entire 30-day window must end by validation cutoff."""
    date = pd.Timestamp(validation_date)
    fit = train.loc[(train.snapshot_date < date) &
                    (train.snapshot_date + pd.Timedelta(days=30) <= date)].copy()
    validation = train.loc[train.snapshot_date.eq(date)].copy()
    if fit.empty or validation.empty:
        raise ValueError(f'Empty rolling fold at {date:%Y-%m-%d}')
    if (fit.snapshot_date + pd.Timedelta(days=30) > date).any():
        raise AssertionError('Training label window crosses rolling validation boundary')
    return fit.reset_index(drop=True), validation.reset_index(drop=True)


def fold_features(fit: pd.DataFrame, validation: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Refit K-means inside each B fold; C uses only features known at cutoff."""
    columns = FEATURE_SETS[feature_set]
    if feature_set == 'B_base_8_plus_3_distances':
        skew = fit[FEATURES].skew()
        log_columns = skew.index[skew.gt(1)].tolist()
        cluster = make_pipeline(3, log_columns, 42)
        cluster.fit(fit[FEATURES])
        fit = fit.assign(**{f'cluster_distance_{i}': values for i, values in
                            enumerate(cluster.transform(fit[FEATURES]).T)})
        validation = validation.assign(**{f'cluster_distance_{i}': values for i, values in
                                          enumerate(cluster.transform(validation[FEATURES]).T)})
    elif feature_set == 'C_base_8_plus_4_behavior':
        fit, validation = derive_features(fit), derive_features(validation)
    return fit, validation, columns


def fit_early_stopped_gbdt(x_fit: pd.DataFrame, y_fit: np.ndarray,
                           x_validation: pd.DataFrame, y_validation: np.ndarray,
                           leaves: int) -> tuple[float, int]:
    """Pick tree count on an inner month, never on the outer fold month."""
    pipeline = build_gbdt_pipeline()
    imputer = pipeline.named_steps['imputer']
    model = pipeline.named_steps['clf']
    model.set_params(num_leaves=leaves, n_estimators=MAX_TREES, metric='average_precision')
    x_fit_array = imputer.fit_transform(x_fit)
    x_validation_array = imputer.transform(x_validation)
    model.fit(x_fit_array, y_fit,
              eval_X=x_validation_array, eval_y=y_validation,
              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                            first_metric_only=True, verbose=False)])
    if 'average_precision' not in model.evals_result_.get('valid_0', {}):
        raise AssertionError('LightGBM early stopping did not monitor average precision')
    iteration = int(model.best_iteration_ or model.n_estimators_)
    scores = model.predict_proba(x_validation_array)[:, 1]
    return float(average_precision_score(y_validation, scores)), iteration


def select_rolling_configuration(tuning: pd.DataFrame,
                                 scores: dict[tuple[str, str, int | str], tuple[np.ndarray, np.ndarray]]) -> tuple[dict, pd.DataFrame]:
    """Select leaves by mean GBDT AP; then soft-vote weight by mean fold AP."""
    selected = {}
    candidate_rows = []
    for feature_set in FEATURE_SETS:
        part = tuning.loc[tuning.feature_set.eq(feature_set)]
        by_leaves = part.groupby('num_leaves').validation_average_precision.mean()
        leaves = int(by_leaves.sort_values(ascending=False, kind='stable').index[0])
        iterations = int(np.median(part.loc[part.num_leaves.eq(leaves), 'best_iteration']))
        fold_aps = {}
        for weight in WEIGHTS:
            values = []
            for date in FOLD_DATES:
                fold = date.strftime('%Y-%m-%d')
                y, lr = scores[(feature_set, fold, 'LR')]
                _, gbdt = scores[(feature_set, fold, leaves)]
                values.append(float(average_precision_score(y, weight * lr + (1 - weight) * gbdt)))
            fold_aps[weight] = values
            candidate_rows.append({'feature_set': feature_set, 'num_leaves': leaves,
                                   'lr_weight': weight, 'mean_fold_ap': float(np.mean(values)),
                                   'min_fold_ap': float(np.min(values))})
        weight = max(WEIGHTS, key=lambda w: (np.mean(fold_aps[w]), -abs(w - .5), -w))
        selected[feature_set] = {'num_leaves': leaves, 'n_estimators': iterations,
                                 'lr_weight': weight,
                                 'mean_tuning_gbdt_ap': float(by_leaves.loc[leaves])}
    candidates = pd.DataFrame(candidate_rows)
    candidates['selected'] = candidates.apply(
        lambda row: row.lr_weight == selected[row.feature_set]['lr_weight'], axis=1)
    return selected, candidates


def rolling_validation(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    tuning_rows, fold_rows, scores, baseline_rows = [], [], {}, []
    with threadpool_limits(limits=1):
        for date in FOLD_DATES:
            fit, validation = make_fold(train, date)
            inner_date = fit.snapshot_date.max()
            inner_fit, inner_validation = make_fold(fit, inner_date)
            fold = date.strftime('%Y-%m-%d')
            y_fit = fit[LABEL].to_numpy(dtype=int)
            y_validation = validation[LABEL].to_numpy(dtype=int)
            y_inner_fit = inner_fit[LABEL].to_numpy(dtype=int)
            y_inner_validation = inner_validation[LABEL].to_numpy(dtype=int)
            print(f'Rolling fold {fold}: fit={len(fit)}, validation={len(validation)}', flush=True)
            baseline = baseline_scores(validation, fit_rfm(fit))
            for name in ('rfm',):
                baseline_rows.append({'fold': fold, 'feature_set': 'baseline', 'model': name,
                                      'fit_snapshot_count': len(fit),
                                      'validation_snapshot_count': len(validation),
                                      **evaluate(validation, baseline[name + '_score'].to_numpy(), y_validation)})
            for feature_set in FEATURE_SETS:
                x_fit, x_validation, columns = fold_features(fit, validation, feature_set)
                x_inner_fit, x_inner_validation, _ = fold_features(
                    inner_fit, inner_validation, feature_set)
                lr = build_lr_pipeline()
                lr.fit(x_fit[columns], y_fit)
                lr_scores = lr.predict_proba(x_validation[columns])[:, 1]
                scores[(feature_set, fold, 'LR')] = (y_validation, lr_scores)
                for leaves in LEAF_OPTIONS:
                    inner_ap, iteration = fit_early_stopped_gbdt(
                        x_inner_fit[columns], y_inner_fit,
                        x_inner_validation[columns], y_inner_validation, leaves)
                    gbdt = build_gbdt_pipeline()
                    gbdt.set_params(clf__num_leaves=leaves, clf__n_estimators=iteration)
                    gbdt.fit(x_fit[columns], y_fit)
                    gbdt_scores = gbdt.predict_proba(x_validation[columns])[:, 1]
                    scores[(feature_set, fold, leaves)] = (y_validation, gbdt_scores)
                    tuning_rows.append({'fold': fold, 'feature_set': feature_set,
                                        'num_leaves': leaves, 'best_iteration': iteration,
                                        'inner_validation_date': inner_date.strftime('%Y-%m-%d'),
                                        'inner_fit_snapshot_count': len(inner_fit),
                                        'inner_validation_snapshot_count': len(inner_validation),
                                        'latest_inner_fit_label_end': str((inner_fit.snapshot_date + pd.Timedelta(days=30)).max().date()),
                                        'inner_validation_average_precision': inner_ap,
                                        'validation_average_precision': float(average_precision_score(y_validation, gbdt_scores)),
                                        'fit_snapshot_count': len(fit),
                                        'validation_snapshot_count': len(validation),
                                        'latest_fit_label_end': str((fit.snapshot_date + pd.Timedelta(days=30)).max().date())})
    tuning = pd.DataFrame(tuning_rows)
    selected, candidates = select_rolling_configuration(tuning, scores)
    for date in FOLD_DATES:
        fit, validation = make_fold(train, date)
        fold = date.strftime('%Y-%m-%d')
        for feature_set in FEATURE_SETS:
            config = selected[feature_set]
            y, lr_scores = scores[(feature_set, fold, 'LR')]
            _, gbdt_scores = scores[(feature_set, fold, config['num_leaves'])]
            for name, prediction in [('LR', lr_scores), ('GBDT', gbdt_scores),
                                     ('SoftVote', config['lr_weight'] * lr_scores +
                                      (1 - config['lr_weight']) * gbdt_scores)]:
                fold_rows.append({'fold': fold, 'feature_set': feature_set, 'model': name,
                                  'fit_snapshot_count': len(fit),
                                  'validation_snapshot_count': len(validation),
                                  **evaluate(validation, prediction, y)})
    fold_table = pd.DataFrame(fold_rows + baseline_rows)
    summary = fold_table.groupby(['feature_set', 'model'], sort=False).agg(
        mean_fold_ap=('validation_average_precision', 'mean'),
        min_fold_ap=('validation_average_precision', 'min'),
        max_fold_ap=('validation_average_precision', 'max'),
        mean_precision_at_20pct=('precision_at_20pct', 'mean'),
    ).reset_index()
    return fold_table, tuning, candidates, selected, summary


def official_validation(data: pd.DataFrame, selected: dict, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = data.loc[data.split.eq('train')].reset_index(drop=True)
    validation = data.loc[data.split.eq('validation')].reset_index(drop=True)
    y_fit = train[LABEL].to_numpy(dtype=int)
    y_validation = validation[LABEL].to_numpy(dtype=int)
    results, predictions, monthly = [], [], []
    for feature_set, columns in FEATURE_SETS.items():
        if feature_set == 'B_base_8_plus_3_distances':
            x_fit, x_validation = add_distances(train, bundle), add_distances(validation, bundle)
        elif feature_set == 'C_base_8_plus_4_behavior':
            x_fit, x_validation = derive_features(train), derive_features(validation)
        else:
            x_fit, x_validation = train, validation
        config = selected[feature_set]
        lr, gbdt = build_lr_pipeline(), build_gbdt_pipeline()
        gbdt.set_params(clf__num_leaves=config['num_leaves'], clf__n_estimators=config['n_estimators'])
        lr.fit(x_fit[columns], y_fit)
        gbdt.fit(x_fit[columns], y_fit)
        lr_scores = lr.predict_proba(x_validation[columns])[:, 1]
        gbdt_scores = gbdt.predict_proba(x_validation[columns])[:, 1]
        weight = config['lr_weight']
        for model, scores in [('LR', lr_scores), ('GBDT', gbdt_scores),
                              ('SoftVote', weight * lr_scores + (1 - weight) * gbdt_scores)]:
            results.append({'feature_set': feature_set, 'model': model,
                            'feature_count': len(columns), 'lr_weight': weight if model == 'SoftVote' else np.nan,
                            'num_leaves': config['num_leaves'] if model != 'LR' else np.nan,
                            'n_estimators': config['n_estimators'] if model != 'LR' else np.nan,
                            'train_snapshot_count': len(train), 'validation_snapshot_count': len(validation),
                            **evaluate(validation, scores, y_validation)})
            details = validation[KEYS].copy()
            details[LABEL] = y_validation
            details['feature_set'], details['model'], details['score'] = feature_set, model, scores
            predictions.append(details)
            for date, group in details.groupby('snapshot_date', sort=True):
                values = group.score.to_numpy()
                labels = group[LABEL].to_numpy(dtype=int)
                monthly.append({'feature_set': feature_set, 'model': model,
                                'snapshot_date': date.strftime('%Y-%m-%d'),
                                'snapshot_count': len(group), 'positive_count': int(labels.sum()),
                                **evaluate(group, values, labels)})
            print(f'{feature_set} {model}: official validation AP={results[-1]["validation_average_precision"]:.4f}', flush=True)
    baseline = baseline_scores(validation, fit_rfm(train))
    for name in ('rfm',):
        scores = baseline[name + '_score'].to_numpy(dtype=float)
        results.append({'feature_set': 'baseline', 'model': name,
                        'feature_count': np.nan, 'lr_weight': np.nan,
                        'num_leaves': np.nan, 'n_estimators': np.nan,
                        'train_snapshot_count': len(train), 'validation_snapshot_count': len(validation),
                        **evaluate(validation, scores, y_validation)})
        details = validation[KEYS].copy()
        details[LABEL], details['score'] = y_validation, scores
        for date, group in details.groupby('snapshot_date', sort=True):
            labels = group[LABEL].to_numpy(dtype=int)
            monthly.append({'feature_set': 'baseline', 'model': name,
                            'snapshot_date': date.strftime('%Y-%m-%d'),
                            'snapshot_count': len(group), 'positive_count': int(labels.sum()),
                            **evaluate(group, group.score.to_numpy(), labels)})
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True), pd.DataFrame(monthly)


def write_report(comparison: pd.DataFrame, rolling: pd.DataFrame,
                 selected: dict, original: pd.DataFrame) -> str:
    lines = [
        '# 阶段五扩展：行为特征、时间滚动验证和 LightGBM 调整', '',
        '原有六组实验及其文件保留。新增 C 组：8 项基础特征 + 4 项截至快照日已知的购买节奏特征，'
        '共比较 9 组模型。短于 90 天的记录不计算订单变化，缺失值仅用对应训练部分的中位数填补。', '',
        '在训练期内部以 2011 年 3、4、5 月各做一次滚动验证；每轮清除跨越该月起点的 30 天训练标签，'
        '并重新拟合填补、缩放、三群 K-means（B 组）和分类器。LightGBM 在每轮训练期内部再留一个月，'
        '比较 15/31 个叶子、最多 500 棵树，连续 30 轮内部验证 AP 不改善时早停；'
        '随后用完整的该轮训练部分重训，在外层验证月评分。用三轮外层平均 AP 选叶子数，'
        '用所选设置的最佳轮数中位数定最终树数。软投票权重也按三轮平均 AP 选择。'
        '这些滚动指标参与选参，不能当作独立测试结果。', '',
        '| 特征组 | 模型 | 滚动 AP 均值 | 官方验证 AP | 官方 Precision@20% |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    merged = comparison.merge(rolling[['feature_set', 'model', 'mean_fold_ap']],
                              on=['feature_set', 'model'], how='left')
    for row in merged.itertuples():
        lines.append(f'| {row.feature_set} | {row.model} | {row.mean_fold_ap:.4f} | '
                     f'{row.validation_average_precision:.4f} | {row.precision_at_20pct:.4f} |')
    lines += ['', '滚动验证选定的 LightGBM 参数与软投票权重：', '',
              '| 特征组 | 叶子数 | 树数 | LR 权重 |', '| --- | ---: | ---: | ---: |']
    for feature_set, config in selected.items():
        lines.append(f'| {feature_set} | {config["num_leaves"]} | {config["n_estimators"]} | '
                     f'{config["lr_weight"]:.2f} |')
    old_formal = original.loc[original.feature_set.ne('baseline')]
    new_formal = comparison.loc[comparison.feature_set.ne('baseline')]
    old_best = old_formal.loc[old_formal.validation_average_precision.idxmax()]
    new_best = new_formal.loc[new_formal.validation_average_precision.idxmax()]
    c_best = new_formal.loc[new_formal.feature_set.eq('C_base_8_plus_4_behavior'),
                            'validation_average_precision'].max()
    lines += ['', f'原六组中验证 AP 最高的是 {old_best.feature_set} / {old_best.model}：'
              f'{old_best.validation_average_precision:.4f}。扩展实验当前最高的是 '
              f'{new_best.feature_set} / {new_best.model}：{new_best.validation_average_precision:.4f}，'
              f'差值 {new_best.validation_average_precision - old_best.validation_average_precision:+.4f}。'
              f'C 组最高 AP 为 {c_best:.4f}，未超过扩展实验的最高值。'
              '这些都是验证期结果，最终测试尚未进行。',
              '', '官方验证期为 2011 年 6—8 月，只在滚动验证选定配置后评估。'
              '9 月以后的测试期未用于调参或计算本报告指标。'
              '逐月结果见 `monthly_validation.csv`，每折与候选参数见 `rolling_folds.csv`、'
              '`lightgbm_tuning.csv`、`weight_candidates.csv`，逐条评分见 `validation_predictions.csv`。'
              '若新特征或调整后的模型没有改善，应以结果为准。', '']
    return '\n'.join(lines)


def main() -> None:
    data, _ = load_snapshots(INPUT)
    train = data.loc[data.split.eq('train')].reset_index(drop=True)
    original_manifest = json.loads((STAGE4 / 'run_manifest.json').read_text(encoding='utf-8'))
    source_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    if original_manifest['input_sha256'] != source_hash:
        raise ValueError('Stage-four model was fitted with a different snapshot file')
    if original_manifest['versions']['scikit-learn'] != sklearn.__version__:
        raise ValueError('Saved stage-four model uses a different scikit-learn version')
    bundle = joblib.load(STAGE4 / 'model/segmentation_bundle.joblib')
    if list(bundle['distance_columns']) != FEATURE_SETS['B_base_8_plus_3_distances'][len(FEATURES):]:
        raise ValueError('Unexpected stage-four distance columns')
    rolling, tuning, candidates, selected, summary = rolling_validation(train)
    comparison, predictions, monthly = official_validation(data, selected, bundle)
    original = pd.read_csv(ORIGINAL / 'validation_comparison.csv')
    if not {'A_base_8', 'B_base_8_plus_3_distances'}.issubset(set(original.feature_set)):
        raise ValueError('Original six-experiment reference is missing')
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    for filename, frame in [
        ('rolling_folds.csv', rolling), ('lightgbm_tuning.csv', tuning),
        ('weight_candidates.csv', candidates), ('rolling_summary.csv', summary),
        ('validation_comparison.csv', comparison), ('validation_predictions.csv', predictions),
        ('monthly_validation.csv', monthly),
    ]:
        frame.to_csv(output / filename, index=False, encoding='utf-8-sig')
    config = {
        'snapshot_sha256': source_hash,
        'stage4_model_sha256': hashlib.sha256((STAGE4 / 'model/segmentation_bundle.joblib').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_stage5_comparison_sha256': hashlib.sha256((ORIGINAL / 'validation_comparison.csv').read_bytes()).hexdigest(),
        'feature_sets': FEATURE_SETS, 'fold_dates': [d.strftime('%Y-%m-%d') for d in FOLD_DATES],
        'purge_rule': 'train snapshot_date + 30 days <= fold validation date',
        'gbdt_leaf_options': list(LEAF_OPTIONS), 'gbdt_max_trees': MAX_TREES,
        'gbdt_early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'gbdt_iteration_selection': 'inner last eligible month with 30-day purge; outer fold month untouched until evaluation',
        'gbdt_selection': 'highest mean fold AP, lower leaf count on tie; median best iteration',
        'soft_vote_weights': list(WEIGHTS), 'soft_vote_selection': 'highest mean fold AP, prefer 0.50 on tie',
        'selected': selected, 'versions': {'python': platform.python_version(),
                                          'numpy': np.__version__, 'pandas': pd.__version__,
                                          'scikit-learn': sklearn.__version__, 'lightgbm': lgb.__version__},
        'official_validation_dates': ['2011-06-01', '2011-07-01', '2011-08-01'],
        'test_labels_used': False,
    }
    (output / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'extended_report.md').write_text(write_report(comparison, summary, selected, original), encoding='utf-8')


if __name__ == '__main__':
    main()
