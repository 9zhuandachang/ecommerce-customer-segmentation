"""Stage 5: two feature sets x LR, GBDT and validation-selected soft vote.

Run from the project root: python stage5/experiments.py
Test-period labels and scores are not used for model selection.
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
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modelling import build_gbdt_pipeline, build_lr_pipeline
from stage4.kmeans import FEATURES, KEYS, LABEL, baseline_scores, fit_rfm, load_snapshots

INPUT = ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv'
STAGE4 = ROOT / 'stage4/outputs/stage4'
OUTPUT = ROOT / 'stage5/outputs/stage5'
WEIGHTS = (0.25, 0.50, 0.75)


def choose_weight(y: np.ndarray, lr: np.ndarray, gbdt: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose only by validation AP; prefer 0.50 on an exact tie."""
    rows = [{'lr_weight': weight,
             'validation_average_precision': float(average_precision_score(
                 y, weight * lr + (1 - weight) * gbdt))}
            for weight in WEIGHTS]
    table = pd.DataFrame(rows)
    winner = table.sort_values(
        ['validation_average_precision', 'lr_weight'], ascending=[False, True],
        kind='stable',
    ).iloc[0]
    maximum = winner.validation_average_precision
    ties = table.loc[table.validation_average_precision.eq(maximum)]
    selected = min(ties.lr_weight, key=lambda weight: (abs(weight - 0.5), weight))
    table['selected'] = table.lr_weight.eq(selected)
    return float(selected), table


def add_distances(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    features = bundle['features']
    if features != FEATURES or bundle['selected_k'] != 3:
        raise ValueError('Stage-four model does not match the declared base features / k=3')
    distances = bundle['pipeline'].transform(frame[features])
    columns = list(bundle['distance_columns'])
    if distances.shape != (len(frame), len(columns)) or not np.isfinite(distances).all():
        raise ValueError('Invalid center distances')
    return frame.assign(**{name: distances[:, i] for i, name in enumerate(columns)})


def top20_metrics(frame: pd.DataFrame, scores: np.ndarray, y: np.ndarray) -> dict:
    """Allocate ceil(20%) slots within each monthly cutoff, then pool hits."""
    ranked = frame[KEYS].copy()
    ranked['score'] = scores
    ranked['label'] = y
    ranked = ranked.sort_values(
        ['snapshot_date', 'score', 'customer_id'],
        ascending=[True, False, True], kind='stable',
    )
    position = ranked.groupby('snapshot_date', sort=False).cumcount() + 1
    quota = np.ceil(ranked.groupby('snapshot_date').customer_id.transform('size') * .2)
    chosen = ranked.loc[position <= quota]
    hits = int(chosen.label.sum())
    positives = int(np.sum(y))
    precision = hits / len(chosen)
    prevalence = positives / len(y)
    return {'top20_count': len(chosen), 'top20_hits': hits,
            'precision_at_20pct': precision,
            'recall_at_20pct': hits / positives if positives else np.nan,
            'lift_at_20pct': precision / prevalence if prevalence else np.nan}


def evaluate(frame: pd.DataFrame, scores: np.ndarray, y: np.ndarray) -> dict:
    return {
        'validation_average_precision': float(average_precision_score(y, scores)),
        'validation_roc_auc': float(roc_auc_score(y, scores)),
        **top20_metrics(frame, scores, y),
    }


def main() -> None:
    data, _ = load_snapshots(INPUT)
    # The test partition is deliberately discarded before feature generation or metrics.
    train = data.loc[data.split.eq('train')].reset_index(drop=True)
    validation = data.loc[data.split.eq('validation')].reset_index(drop=True)
    manifest = json.loads((STAGE4 / 'run_manifest.json').read_text(encoding='utf-8'))
    source_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    if manifest['input_sha256'] != source_hash:
        raise ValueError('Snapshot file differs from the stage-four training input')
    if manifest['versions']['scikit-learn'] != sklearn.__version__:
        raise ValueError('scikit-learn version differs from the saved clustering model')
    bundle = joblib.load(STAGE4 / 'model/segmentation_bundle.joblib')
    train = add_distances(train, bundle)
    validation = add_distances(validation, bundle)
    distance_columns = list(bundle['distance_columns'])
    y_train = train[LABEL].to_numpy(dtype=int)
    y_validation = validation[LABEL].to_numpy(dtype=int)
    results, weight_rows, predictions = [], [], []
    for set_name, columns in [('A_base_8', FEATURES), ('B_base_8_plus_3_distances', FEATURES + distance_columns)]:
        lr = build_lr_pipeline()
        gbdt = build_gbdt_pipeline()
        lr.fit(train[columns], y_train)
        gbdt.fit(train[columns], y_train)
        lr_scores = lr.predict_proba(validation[columns])[:, 1]
        gbdt_scores = gbdt.predict_proba(validation[columns])[:, 1]
        weight, candidates = choose_weight(y_validation, lr_scores, gbdt_scores)
        candidates.insert(0, 'feature_set', set_name)
        weight_rows.append(candidates)
        for model_name, scores, selected_weight in [
            ('LR', lr_scores, np.nan),
            ('GBDT', gbdt_scores, np.nan),
            ('SoftVote', weight * lr_scores + (1 - weight) * gbdt_scores, weight),
        ]:
            results.append({'feature_set': set_name, 'model': model_name,
                            'feature_count': len(columns), 'lr_weight': selected_weight,
                            'train_snapshot_count': len(train),
                            'validation_snapshot_count': len(validation),
                            **evaluate(validation, scores, y_validation)})
            predicted = validation[KEYS].copy()
            predicted[LABEL] = y_validation
            predicted['feature_set'] = set_name
            predicted['model'] = model_name
            predicted['score'] = scores
            predictions.append(predicted)
            print(f'{set_name} {model_name}: AP={results[-1]["validation_average_precision"]:.4f}', flush=True)

    # Identical snapshots, dates, labels and Top20 allocation as the formal models.
    baseline = baseline_scores(validation, fit_rfm(train))
    for name in ('rfm',):
        scores = baseline[name + '_score'].to_numpy(dtype=float)
        results.append({'feature_set': 'baseline', 'model': name,
                        'feature_count': np.nan, 'lr_weight': np.nan,
                        'train_snapshot_count': len(train),
                        'validation_snapshot_count': len(validation),
                        **evaluate(validation, scores, y_validation)})
    table = pd.DataFrame(results)
    weights = pd.concat(weight_rows, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True)
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / 'validation_comparison.csv', index=False, encoding='utf-8-sig')
    weights.to_csv(output / 'weight_selection.csv', index=False, encoding='utf-8-sig')
    prediction_table.to_csv(output / 'validation_predictions.csv', index=False, encoding='utf-8-sig')
    config = {
        'snapshot_sha256': source_hash,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'stage4_model_sha256': hashlib.sha256((STAGE4 / 'model/segmentation_bundle.joblib').read_bytes()).hexdigest(),
        'stage4_selected_k': bundle['selected_k'], 'stage4_seed': bundle['seed'],
        'feature_sets': {'A_base_8': FEATURES, 'B_base_8_plus_3_distances': FEATURES + distance_columns},
        'models': ['LR', 'GBDT', 'SoftVote'],
        'base_model_builders': 'src.modelling.build_lr_pipeline / build_gbdt_pipeline',
        'lr_classifier_params': {name: lr.named_steps['clf'].get_params()[name] for name in
                                 ('max_iter', 'class_weight', 'random_state', 'solver')},
        'gbdt_classifier_params': {name: gbdt.named_steps['clf'].get_params()[name] for name in
                                   ('n_estimators', 'learning_rate', 'num_leaves', 'class_weight', 'random_state')},
        'soft_vote_weights': list(WEIGHTS),
        'weight_selection': 'maximum validation average precision, prefer 0.50 on exact tie',
        'partitions': {'train': {'snapshots': len(train), 'dates': sorted(train.snapshot_date.dt.strftime('%Y-%m-%d').unique().tolist())},
                       'validation': {'snapshots': len(validation), 'dates': sorted(validation.snapshot_date.dt.strftime('%Y-%m-%d').unique().tolist())}},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                     'lightgbm': lightgbm.__version__},
        'test_labels_used': False,
    }
    (output / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    base = table.loc[table.feature_set.eq('A_base_8')].set_index('model')
    added = table.loc[table.feature_set.eq('B_base_8_plus_3_distances')].set_index('model')
    lines = [
        '# Stage 5: Six groups of repurchase prediction experiments', '',
        'The two sets of features use the same 37,019 customer-month records in the training period and 8,146 customer-month records in the validation period.'
        'A is the 8 basic transaction characteristics of stage four; B plus the three cluster center distances that have been fitted in the training period.'
        'LR and GBDT are each trained once, and the soft voting weight of each group is selected from 0.25, 0.50 and 0.75 using only the verification set AP.'
        'Test period metrics are not calculated.', '',
        '| Feature Group | Model | Validation AP | ROC-AUC | Precision@ 20% | Recall@20% |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in table.itertuples():
        lines.append(f'| {row.feature_set} | {row.model} | {row.validation_average_precision:.4f} | '
                     f'{row.validation_roc_auc:.4f} | {row.precision_at_20pct:.4f} | '
                     f'{row.recall_at_20pct:.4f} |')
    lines += ['', 'The weight selected for the verification period of LR in soft voting: A is'
              f'{float(weights.loc[(weights.feature_set == "A_base_8") & weights.selected, "lr_weight"].iloc[0]):.2f}，'
              'B is'
              f'{float(weights.loc[(weights.feature_set == "B_base_8_plus_3_distances") & weights.selected, "lr_weight"].iloc[0]):.2f}。',
        '', 'Changes in AP after adding clustering distance (B minus A):' + '；'.join(
        f'{name} {added.loc[name, "validation_average_precision"] - base.loc[name, "validation_average_precision"]:+.4f}'
        for name in ('LR', 'GBDT', 'SoftVote')) + '。',
        '', 'The Top 20% quota is calculated as ceil ( 0.2 × number of customers in the month) for each verification month, and then the number of hits is summarized.'
        'Results only represent ranking performance for the historical validation period; probability calibration, thresholds, and final testing are left to phase six.',
        '', 'See `validation_predictions.csv` for item-by-item verification scores; see `weight_selection.csv` for the weights and selection values of the three candidates;'
        'See `run_config.json` for complete characteristics, versions, and source hashes.',
    ]
    (output / 'stage5_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
