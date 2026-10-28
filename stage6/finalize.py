"""Stage 6. Run `prepare` once, then `evaluate` on the held-out months.

`prepare` chooses the highest validation AP from the recorded stage-five
experiments, fits only on training snapshots, and calibrates/sets the binary
threshold on validation snapshots. `evaluate` verifies the frozen artifact
before reading test labels.
"""

from __future__ import annotations

import argparse
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
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from src.modelling import build_gbdt_pipeline, build_lr_pipeline
from stage4.kmeans import FEATURES, KEYS, LABEL, baseline_scores, load_snapshots
from stage4.kmeans import EXTRA_FEATURES, derive_features
from stage5.experiments import add_distances, top20_metrics

INPUT = ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv'
STAGE4 = ROOT / 'stage4/outputs/stage4'
STAGE5 = ROOT / 'stage5/outputs/extended'
OUTPUT = ROOT / 'stage6/outputs/final'
MODEL = OUTPUT / 'model/final_bundle.joblib'
LOCK = OUTPUT / 'locked_config.json'
TEST_DATES = ['2011-09-01', '2011-10-01', '2011-11-01']
VERSION = 'stage6-v1'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_model(comparison: pd.DataFrame) -> pd.Series:
    """Use validation AP only, with a deterministic name tie break."""
    formal = comparison.loc[comparison.feature_set.ne('baseline')].copy()
    if formal.empty or formal.validation_average_precision.isna().any():
        raise ValueError('Missing stage-five model comparison')
    return formal.sort_values(['validation_average_precision', 'feature_set', 'model'],
                              ascending=[False, True, True], kind='stable').iloc[0]


def choose_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Freeze the validation F1 threshold; ranking still uses monthly Top 20%."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    return float(thresholds[best[-1]])  # conservative tie break


def model_input(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    feature_set = bundle['feature_set']
    if feature_set == 'B_base_8_plus_3_distances':
        frame = add_distances(frame, bundle['segmenter'])
    elif feature_set == 'C_base_8_plus_4_behavior':
        frame = derive_features(frame)
    elif feature_set != 'A_base_8':
        raise ValueError(f'Unknown feature set: {feature_set}')
    return frame[bundle['features']]


def score_snapshots(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Same entry point for evaluation and the next-stage dashboard."""
    x = model_input(frame, bundle)
    raw = bundle['classifier'].predict_proba(x)[:, 1]
    calibrated = bundle['calibrator'].predict_proba(x)[:, 1]
    groups = bundle['segmenter']['pipeline'].predict(frame[FEATURES])
    return pd.DataFrame({'cluster': groups.astype(int), 'raw_score': raw,
                         'score': calibrated}, index=frame.index)


def top20_mask(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    ranked = frame[KEYS].copy()
    ranked['score'] = scores
    ranked['row_number'] = np.arange(len(frame))
    ranked = ranked.sort_values(['snapshot_date', 'score', 'customer_id'],
                                ascending=[True, False, True], kind='stable')
    position = ranked.groupby('snapshot_date', sort=False).cumcount() + 1
    quota = np.ceil(ranked.groupby('snapshot_date').customer_id.transform('size') * .2)
    mask = np.zeros(len(frame), dtype=bool)
    mask[ranked.loc[position <= quota, 'row_number'].to_numpy(dtype=int)] = True
    return mask


def evaluate_scores(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    y = frame[LABEL].to_numpy(dtype=int)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'snapshot_count': len(frame), 'positive_count': int(y.sum()),
            'positive_rate': float(y.mean()),
            'average_precision': float(average_precision_score(y, scores)),
            'roc_auc': float(roc_auc_score(y, scores)),
            'brier': float(brier_score_loss(y, scores)),
            'threshold': threshold,
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            **top20_metrics(frame, scores, y)}


def calibration_bins(frame: pd.DataFrame, scores: np.ndarray,
                     period: str, kind: str) -> pd.DataFrame:
    bins = pd.qcut(pd.Series(scores), q=10, labels=False, duplicates='drop')
    table = pd.DataFrame({'bin': bins, 'score': scores,
                          'label': frame[LABEL].to_numpy(dtype=int)})
    out = table.groupby('bin', sort=True).agg(
        count=('label', 'size'), mean_score=('score', 'mean'),
        observed_rate=('label', 'mean')).reset_index()
    out.insert(0, 'period', period)
    out.insert(1, 'kind', kind)
    out['absolute_gap'] = (out.mean_score - out.observed_rate).abs()
    return out


def prepare() -> None:
    if (OUTPUT / 'test_comparison.csv').exists():
        raise RuntimeError('Test results already exist; do not reselect after seeing the test set')
    stage5_config = json.loads((STAGE5 / 'run_config.json').read_text(encoding='utf-8'))
    stage4_config = json.loads((STAGE4 / 'run_manifest.json').read_text(encoding='utf-8'))
    source_hash = sha256(INPUT)
    segmenter_path = STAGE4 / 'model/segmentation_bundle.joblib'
    if source_hash != stage5_config['snapshot_sha256'] or source_hash != stage4_config['input_sha256']:
        raise ValueError('Snapshot source differs from stages four/five')
    if sha256(segmenter_path) != stage5_config['stage4_model_sha256']:
        raise ValueError('Stage-four segmenter differs from stage-five validation')
    if stage5_config['versions']['scikit-learn'] != sklearn.__version__:
        raise ValueError('Saved pipelines require the same scikit-learn version')
    comparison = pd.read_csv(STAGE5 / 'validation_comparison.csv')
    winner = select_model(comparison)
    feature_set, model_name = winner.feature_set, winner.model
    settings = stage5_config['selected'][feature_set]
    data, _ = load_snapshots(INPUT)
    train = data.loc[data.split.eq('train')].reset_index(drop=True)
    validation = data.loc[data.split.eq('validation')].reset_index(drop=True)
    segmenter = joblib.load(segmenter_path)
    bundle = {'version': VERSION, 'feature_set': feature_set, 'model_name': model_name,
              'features': stage5_config['feature_sets'][feature_set],
              'segmenter': segmenter, 'rfm_config': stage4_config['rfm_config']}
    if segmenter['features'] != FEATURES or segmenter['selected_k'] != 3:
        raise ValueError('Unexpected stage-four segmentation configuration')
    if model_name == 'GBDT':
        classifier = build_gbdt_pipeline()
        classifier.set_params(clf__num_leaves=settings['num_leaves'],
                              clf__n_estimators=settings['n_estimators'])
    elif model_name == 'LR':
        classifier = build_lr_pipeline()
    else:
        raise ValueError('Stage-six single-model persistence expects LR or GBDT')
    classifier.fit(model_input(train, bundle), train[LABEL].to_numpy(dtype=int))
    bundle['classifier'] = classifier
    raw_validation = classifier.predict_proba(model_input(validation, bundle))[:, 1]
    reproduced_ap = average_precision_score(validation[LABEL], raw_validation)
    if abs(reproduced_ap - float(winner.validation_average_precision)) > 1e-10:
        raise ValueError('Stage-five winner cannot be reproduced from its recorded configuration')
    calibrator = CalibratedClassifierCV(FrozenEstimator(classifier), method='sigmoid')
    calibrator.fit(model_input(validation, bundle), validation[LABEL].to_numpy(dtype=int))
    bundle['calibrator'] = calibrator
    scored = score_snapshots(bundle, validation)
    threshold = choose_f1_threshold(validation[LABEL].to_numpy(dtype=int),
                                    scored.score.to_numpy())
    bundle['threshold'] = threshold
    OUTPUT.mkdir(parents=True, exist_ok=True)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL, compress=3)
    lock = {'version': VERSION, 'selected_feature_set': feature_set,
            'selected_model': model_name, 'selection_metric': 'maximum official validation AP',
            'stage5_validation_ap': float(winner.validation_average_precision),
            'gbdt_settings': settings if model_name == 'GBDT' else None,
            'train_snapshot_count': len(train), 'validation_snapshot_count': len(validation),
            'threshold_selection': 'maximum validation F1 on sigmoid-calibrated score; highest threshold on tie',
            'threshold': threshold, 'calibration': 'sigmoid on validation; frozen train-fitted classifier',
            'snapshot_sha256': source_hash,
            'stage4_model_sha256': sha256(segmenter_path),
            'stage5_comparison_sha256': sha256(STAGE5 / 'validation_comparison.csv'),
            'stage5_config_sha256': sha256(STAGE5 / 'run_config.json'),
            'stage6_script_sha256': sha256(Path(__file__)),
            'model_sha256': sha256(MODEL),
            'test_dates_reserved': TEST_DATES, 'test_labels_used': False,
            'versions': {'python': platform.python_version(),
                         'numpy': np.__version__, 'pandas': pd.__version__,
                         'scikit-learn': sklearn.__version__,
                         'lightgbm': lightgbm.__version__}}
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    validation_output = validation[KEYS + [LABEL]].copy()
    for column in scored:
        validation_output[column] = scored[column]
    validation_output['top20_selected'] = top20_mask(validation, scored.score.to_numpy())
    validation_output.to_csv(OUTPUT / 'validation_predictions.csv', index=False, encoding='utf-8-sig')
    raw_metrics = evaluate_scores(validation, scored.raw_score.to_numpy(), threshold)
    calibrated_metrics = evaluate_scores(validation, scored.score.to_numpy(), threshold)
    pd.DataFrame([{'period': 'validation', 'kind': 'raw', **raw_metrics},
                  {'period': 'validation', 'kind': 'sigmoid', **calibrated_metrics}]).to_csv(
                      OUTPUT / 'validation_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat([calibration_bins(validation, scored.raw_score.to_numpy(), 'validation', 'raw'),
               calibration_bins(validation, scored.score.to_numpy(), 'validation', 'sigmoid')],
              ignore_index=True).to_csv(OUTPUT / 'validation_calibration.csv', index=False,
                                        encoding='utf-8-sig')
    print(f'Frozen {feature_set}/{model_name}; validation AP={reproduced_ap:.4f}; '
          f'threshold={threshold:.6f}; model={MODEL}', flush=True)


def evaluate() -> None:
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    expected = [(INPUT, 'snapshot_sha256'), (MODEL, 'model_sha256'),
                (STAGE4 / 'model/segmentation_bundle.joblib', 'stage4_model_sha256'),
                (STAGE5 / 'validation_comparison.csv', 'stage5_comparison_sha256'),
                (STAGE5 / 'run_config.json', 'stage5_config_sha256'),
                (Path(__file__), 'stage6_script_sha256')]
    for path, key in expected:
        if sha256(path) != lock[key]:
            raise ValueError(f'Frozen input changed: {path}')
    if lock['version'] != VERSION or lock['test_dates_reserved'] != TEST_DATES:
        raise ValueError('Unexpected lock configuration')
    bundle = joblib.load(MODEL)
    if (bundle['version'], bundle['feature_set'], bundle['model_name'], bundle['threshold']) != (
            VERSION, lock['selected_feature_set'], lock['selected_model'], lock['threshold']):
        raise ValueError('Saved model does not match frozen configuration')
    data, _ = load_snapshots(INPUT)
    test = data.loc[data.split.eq('test')].reset_index(drop=True)
    dates = sorted(test.snapshot_date.dt.strftime('%Y-%m-%d').unique().tolist())
    if dates != TEST_DATES:
        raise ValueError(f'Unexpected test months: {dates}')
    scored = score_snapshots(bundle, test)
    reloaded = joblib.load(MODEL)
    repeat = score_snapshots(reloaded, test)
    np.testing.assert_allclose(scored[['raw_score', 'score']], repeat[['raw_score', 'score']],
                               rtol=0, atol=0)
    np.testing.assert_array_equal(scored.cluster, repeat.cluster)
    test_output = test[KEYS + [LABEL]].copy()
    for column in scored:
        test_output[column] = scored[column]
    test_output['top20_selected'] = top20_mask(test, scored.score.to_numpy())
    test_output.to_csv(OUTPUT / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    raw = evaluate_scores(test, scored.raw_score.to_numpy(), bundle['threshold'])
    final = evaluate_scores(test, scored.score.to_numpy(), bundle['threshold'])
    baseline = baseline_scores(test, bundle['rfm_config'])
    rfm = baseline.rfm_score.to_numpy(dtype=float)
    y = test[LABEL].to_numpy(dtype=int)
    comparison = pd.DataFrame([
        {'period': 'test', 'model': bundle['model_name'], 'feature_set': bundle['feature_set'],
         'score_kind': 'sigmoid', **final},
        {'period': 'test', 'model': 'RFM', 'feature_set': 'baseline',
         'score_kind': 'rank_only', 'snapshot_count': len(test),
         'positive_count': int(y.sum()), 'positive_rate': float(y.mean()),
         'average_precision': float(average_precision_score(y, rfm)),
         'roc_auc': float(roc_auc_score(y, rfm)), **top20_metrics(test, rfm, y)}])
    comparison.to_csv(OUTPUT / 'test_comparison.csv', index=False, encoding='utf-8-sig')
    monthly = pd.DataFrame([{'snapshot_date': date.strftime('%Y-%m-%d'),
                             **evaluate_scores(group, scored.loc[group.index, 'score'].to_numpy(),
                                               bundle['threshold'])}
                            for date, group in test.groupby('snapshot_date', sort=True)])
    monthly.to_csv(OUTPUT / 'monthly_test.csv', index=False, encoding='utf-8-sig')
    groups = []
    for cluster, part in test_output.groupby('cluster', sort=True):
        labels = part[LABEL].to_numpy(dtype=int)
        groups.append({'cluster': int(cluster), 'snapshot_count': len(part),
                       'positive_count': int(labels.sum()), 'positive_rate': float(labels.mean()),
                       'average_precision': float(average_precision_score(labels, part.score)),
                       'roc_auc': float(roc_auc_score(labels, part.score)),
                       'brier': float(brier_score_loss(labels, part.score)),
                       'top20_selected_count': int(part.top20_selected.sum()),
                       'top20_hits': int(part.loc[part.top20_selected, LABEL].sum())})
    pd.DataFrame(groups).to_csv(OUTPUT / 'cluster_test.csv', index=False, encoding='utf-8-sig')
    pd.concat([calibration_bins(test, scored.raw_score.to_numpy(), 'test', 'raw'),
               calibration_bins(test, scored.score.to_numpy(), 'test', 'sigmoid')],
              ignore_index=True).to_csv(OUTPUT / 'test_calibration.csv', index=False,
                                        encoding='utf-8-sig')
    validation = pd.read_csv(OUTPUT / 'validation_metrics.csv')
    val = validation.loc[validation.kind.eq('sigmoid')].iloc[0]
    calibration_gain = raw['brier'] - final['brier']
    lines = [
        '# 阶段六：冻结模型与独立测试结果', '',
        f'验证期（2011 年 6–8 月）按 AP 选定 {bundle["feature_set"]} / {bundle["model_name"]}。'
        f'基础分类器仅在训练期拟合；验证期用于 sigmoid 校准及固定 F1 分类阈值 '
        f'{bundle["threshold"]:.6f}。测试期为 2011 年 9–11 月，未参与上述选择。'
        '测试前已写入带哈希的 locked_config.json 和可加载模型。', '',
        '| 指标 | 验证期 | 独立测试期 | 测试期 RFM |',
        '| --- | ---: | ---: | ---: |',
        f'| AP | {val.average_precision:.4f} | {final["average_precision"]:.4f} | {comparison.iloc[1].average_precision:.4f} |',
        f'| ROC-AUC | {val.roc_auc:.4f} | {final["roc_auc"]:.4f} | {comparison.iloc[1].roc_auc:.4f} |',
        f'| Top 20% Precision | {val.precision_at_20pct:.2%} | {final["precision_at_20pct"]:.2%} | {comparison.iloc[1].precision_at_20pct:.2%} |',
        f'| Top 20% Recall | {val.recall_at_20pct:.2%} | {final["recall_at_20pct"]:.2%} | {comparison.iloc[1].recall_at_20pct:.2%} |',
        f'| Top 20% Lift | {val.lift_at_20pct:.3f} | {final["lift_at_20pct"]:.3f} | {comparison.iloc[1].lift_at_20pct:.3f} |',
        '', f'测试期 {len(test):,} 条客户月快照，正例 {int(y.sum()):,} 条（{y.mean():.2%}）。'
        f'每月分别选 ceil(当月客户数 × 20%)，共 {final["top20_count"]:,} 个名额，'
        f'命中 {final["top20_hits"]:,} 条；RFM 命中 {int(comparison.iloc[1].top20_hits):,} 条。'
        '同一客户跨月可能出现多次，统计单位是客户月快照。', '',
        f'固定阈值下：Precision {final["precision"]:.4f}，Recall {final["recall"]:.4f}，'
        f'F1 {final["f1"]:.4f}；混淆矩阵 TN={final["tn"]}, FP={final["fp"]}, '
        f'FN={final["fn"]}, TP={final["tp"]}。Top 20% 名单独立于该阈值。', '',
        f'测试 Brier：原始评分 {raw["brier"]:.4f}，sigmoid 后 {final["brier"]:.4f}，'
        f'差值（原始减校准）{calibration_gain:+.4f}。分箱可靠性见 test_calibration.csv。'
        '目前将输出称为模型评分；它表示历史数据上的复购风险排序，'
        '不代表营销活动的因果增量或收入提升。', '',
        '逐月、分群和逐条结果分别见 monthly_test.csv、cluster_test.csv、test_predictions.csv。'
        '独立测试结果只用于报告，不回头修改本次模型或阈值。', ''
    ]
    (OUTPUT / 'stage6_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Test AP={final["average_precision"]:.4f}; RFM={comparison.iloc[1].average_precision:.4f}; '
          f'Top20 hits={final["top20_hits"]}/{final["top20_count"]}', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=['prepare', 'evaluate'])
    step = parser.parse_args().step
    if step == 'prepare':
        prepare()
    else:
        evaluate()


if __name__ == '__main__':
    main()
