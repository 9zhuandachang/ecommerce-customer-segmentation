"""Package the newly selected four-cluster and repurchase models for replay.

Run from the project root: python -B -m src.integration.models
The 2011-09 through 2011-11 results are retrospective diagnostics because
these months were viewed during earlier project work.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src.segmentation.comparison import assign, behavior_features
from src.repurchase.comparison import (FEATURE_SETS, SNAPSHOTS, TRANSACTIONS,
                                        add_features, evaluate, make_model,
                                        read_invoice_dates)
from src.data.validation import FEATURES, KEYS, LABEL, load_snapshots

from src.paths import ROOT
SEGMENT_DIR = ROOT / 'artifacts/segmentation/current'
REPURCHASE_DIR = ROOT / 'artifacts/repurchase/current'
OUTPUT = ROOT / 'artifacts/integration/current'
VERSION = 'reanalysis-v2'
TEST_DATES = ('2011-09-01', '2011-10-01', '2011-11-01')
SEGMENT_NAMES = {
    0: 'Group 0 | Recent low order value',
    1: 'Group 1 | Long inactive',
    2: 'Group 2 | Recent frequent buyers',
    3: 'Group 3 | Recent high order value',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_source_artifacts() -> tuple[dict, dict]:
    segmentation = json.loads((SEGMENT_DIR / 'selected.json').read_text(encoding='utf-8'))
    repurchase = json.loads((REPURCHASE_DIR / 'run_manifest.json').read_text(encoding='utf-8'))
    if (segmentation['selected'] != {'algorithm': 'kmeans', 'k': 4,
                                    'covariance_type': ''}
            or segmentation['model_sha256'] != sha256(SEGMENT_DIR / 'selected_model.joblib')
            or segmentation['assignments_sha256'] != sha256(SEGMENT_DIR / 'selected_assignments.csv')
            or repurchase['selected']['model'] != 'XGBoost'
            or repurchase['selected']['feature_set'] != 'base_8_plus_trend_and_rhythm'):
        raise ValueError('New stage-four or stage-five selection changed')
    if (repurchase['snapshot_sha256'] != sha256(SNAPSHOTS)
            or repurchase['clean_transactions_sha256'] != sha256(TRANSACTIONS)):
        raise ValueError('Stage-five source data changed')
    return segmentation, repurchase


def score(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Inputs are historical snapshots with rhythm features already generated."""
    groups = assign(bundle['segmenter'], behavior_features(frame))
    scores = bundle['repurchase_model'].predict_proba(
        frame[bundle['repurchase_features']])[:, 1]
    if (len(groups) != len(frame) or len(scores) != len(frame)
            or not set(groups).issubset(SEGMENT_NAMES)
            or not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()):
        raise AssertionError('Invalid integrated scores or clusters')
    return pd.DataFrame({'cluster': groups, 'score': scores}, index=frame.index)


def verify_prior_outputs(bundle: dict, validation: pd.DataFrame,
                         test: pd.DataFrame) -> None:
    """Check the new package reproduces the earlier stage-four/five decisions."""
    assignments = pd.read_csv(
        SEGMENT_DIR / 'selected_assignments.csv',
        dtype={'customer_id': 'string'}, parse_dates=['snapshot_date'])
    for part in (validation, test):
        prior = assignments.loc[
            assignments.split.eq(part.split.iloc[0]), KEYS + ['cluster']
        ].sort_values(KEYS[:2], kind='stable').reset_index(drop=True)
        ordered = part.sort_values(KEYS[:2], kind='stable').reset_index(drop=True)
        if not ordered[KEYS].equals(prior[KEYS]):
            raise AssertionError('Stage-four assignment keys changed')
        np.testing.assert_array_equal(score(bundle, ordered).cluster.to_numpy(),
                                      prior.cluster.to_numpy())

    prior_scores = pd.read_csv(
        REPURCHASE_DIR / 'validation_predictions.csv',
        dtype={'customer_id': 'string'}, parse_dates=['snapshot_date'])
    prior_scores = prior_scores.loc[
        prior_scores.feature_set.eq(bundle['feature_set'])
        & prior_scores.model.eq(bundle['model_name'])
    ].sort_values(KEYS[:2], kind='stable').reset_index(drop=True)
    ordered = validation.sort_values(KEYS[:2], kind='stable').reset_index(drop=True)
    if not ordered[KEYS + [LABEL]].equals(prior_scores[KEYS + [LABEL]]):
        raise AssertionError('Stage-five validation keys or labels changed')
    np.testing.assert_allclose(score(bundle, ordered).score.to_numpy(),
                               prior_scores.score.to_numpy(), rtol=0, atol=1e-12)


def as_dashboard_rows(frame: pd.DataFrame, scored: pd.DataFrame,
                      segment_names: dict[int, str]) -> pd.DataFrame:
    rows = frame[KEYS + FEATURES].copy()
    rows['cluster'] = scored.cluster.to_numpy(dtype=int)
    rows['segment_name'] = rows.cluster.map(segment_names)
    rows['repurchase_score'] = scored.score.to_numpy(dtype=float)
    if rows.segment_name.isna().any():
        raise AssertionError('Unlabelled customer group')
    return rows


def monthly_top20_mask(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    ranked = frame[KEYS].copy()
    ranked['score'] = scores
    ranked['row_number'] = np.arange(len(frame))
    ranked = ranked.sort_values(['snapshot_date', 'score', 'customer_id'],
                                ascending=[True, False, True], kind='stable')
    position = ranked.groupby('snapshot_date', sort=False).cumcount() + 1
    quota = np.ceil(ranked.groupby('snapshot_date').customer_id.transform('size') * .2)
    mask = np.zeros(len(frame), dtype=bool)
    mask[ranked.loc[position.le(quota), 'row_number'].to_numpy(dtype=int)] = True
    return mask


def main() -> None:
    segmentation_decision, repurchase_decision = verify_source_artifacts()
    segmenter = joblib.load(SEGMENT_DIR / 'selected_model.joblib')
    repurchase = joblib.load(REPURCHASE_DIR / 'selected_model.joblib')
    if (segmenter['candidate'] != segmentation_decision['selected']
            or segmenter['features'] != segmentation_decision['features']
            or repurchase['model_name'] != repurchase_decision['selected']['model']
            or repurchase['feature_set'] != repurchase_decision['selected']['feature_set']
            or repurchase['features'] != FEATURE_SETS[repurchase['feature_set']]):
        raise ValueError('Saved model does not match selected configuration')
    bundle = {
        'version': VERSION, 'segmenter': segmenter,
        'repurchase_model': repurchase['model'],
        'repurchase_features': repurchase['features'],
        'model_name': repurchase['model_name'],
        'feature_set': repurchase['feature_set'],
        'segment_names': SEGMENT_NAMES,
        'trained_through': repurchase['trained_through'],
        'score_kind': 'uncalibrated ranking score',
    }

    snapshots, _ = load_snapshots(SNAPSHOTS)
    train = snapshots.loc[snapshots.split.eq('train')].reset_index(drop=True)
    validation = snapshots.loc[snapshots.split.eq('validation')].reset_index(drop=True)
    test = snapshots.loc[snapshots.split.eq('test')].reset_index(drop=True)
    dates = tuple(test.snapshot_date.dt.strftime('%Y-%m-%d').unique())
    if dates != TEST_DATES:
        raise ValueError('Unexpected retrospective review dates')
    invoice_dates = read_invoice_dates(TRANSACTIONS)
    # The selected classifier remains trained on the original train partition.
    validation_input = add_features(validation, invoice_dates)
    test_input = add_features(test, invoice_dates)
    verify_prior_outputs(bundle, validation_input, test_input)

    # LR is an explicit, same-snapshot learned reference from the new stage-five protocol.
    train_input = add_features(train, invoice_dates)
    reference = make_model('LR', 0.1, FEATURE_SETS['base_8'])
    reference.fit(train_input[FEATURE_SETS['base_8']], train_input[LABEL])
    test_scores = score(bundle, test_input)
    reference_scores = reference.predict_proba(test_input[FEATURE_SETS['base_8']])[:, 1]
    comparison_rows, monthly_rows = [], []
    for name, feature_set, values in (
        (bundle['model_name'], bundle['feature_set'], test_scores.score.to_numpy()),
        ('LR', 'base_8', reference_scores),
    ):
        metrics = evaluate(test_input, values)
        comparison_rows.append({
            'model': name, 'feature_set': feature_set,
            'snapshot_count': len(test_input), 'positive_count': int(test_input[LABEL].sum()),
            'average_precision': metrics.pop('ap'), **metrics,
        })
        for cutoff, month in test_input.groupby('snapshot_date', sort=True):
            month_scores = values[month.index.to_numpy(dtype=int)]
            month_metrics = evaluate(month, month_scores)
            monthly_rows.append({
                'snapshot_date': cutoff.date().isoformat(),
                'model': name, 'feature_set': feature_set,
                'snapshot_count': len(month),
                'positive_count': int(month[LABEL].sum()),
                'average_precision': month_metrics.pop('ap'),
                **month_metrics,
            })
    comparison = pd.DataFrame(comparison_rows)
    monthly = pd.DataFrame(monthly_rows)
    if comparison.snapshot_count.nunique() != 1:
        raise AssertionError('Comparison cohorts differ')

    predictions = as_dashboard_rows(test_input, test_scores, SEGMENT_NAMES)
    predictions[LABEL] = test_input[LABEL].to_numpy(dtype=int)
    predictions['top20_selected'] = monthly_top20_mask(
        test_input, test_scores.score.to_numpy())
    predictions['reference_lr_score'] = reference_scores
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT / 'model/integrated_bundle.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    prediction_path = OUTPUT / 'test_predictions.csv'
    comparison_path = OUTPUT / 'test_comparison.csv'
    monthly_path = OUTPUT / 'monthly_comparison.csv'
    predictions.to_csv(prediction_path, index=False, encoding='utf-8-sig',
                       date_format='%Y-%m-%d')
    comparison.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    monthly.to_csv(monthly_path, index=False, encoding='utf-8-sig')
    lock = {
        'version': VERSION,
        'source_scope': 'historical_replay_test_months_previously_viewed',
        'test_labels_used_for_selection': False,
        'test_labels_used_for_diagnostics': True,
        'test_dates': list(TEST_DATES),
        'model_name': bundle['model_name'], 'feature_set': bundle['feature_set'],
        'segment_candidate': segmentation_decision['candidate'],
        'segment_names': SEGMENT_NAMES,
        'score_kind': bundle['score_kind'],
        'source_hashes': {
            'snapshots': sha256(SNAPSHOTS),
            'transactions': sha256(TRANSACTIONS),
            'segment_model': sha256(SEGMENT_DIR / 'selected_model.joblib'),
            'segment_decision': sha256(SEGMENT_DIR / 'selected.json'),
            'segment_assignments': sha256(SEGMENT_DIR / 'selected_assignments.csv'),
            'repurchase_model': sha256(REPURCHASE_DIR / 'selected_model.joblib'),
            'repurchase_manifest': sha256(REPURCHASE_DIR / 'run_manifest.json'),
            'repurchase_validation_predictions': sha256(
                REPURCHASE_DIR / 'validation_predictions.csv'),
            'integration_script': sha256(Path(__file__)),
        },
        'output_hashes': {
            'model': sha256(model_path),
            'test_predictions': sha256(prediction_path),
            'test_comparison': sha256(comparison_path),
            'monthly_comparison': sha256(monthly_path),
        },
        'counts': {'train': len(train), 'validation': len(validation),
                   'test': len(test)},
        'versions': {'python': platform.python_version(),
                     'scikit_learn': sklearn.__version__,
                     'xgboost': xgboost.__version__},
    }
    (OUTPUT / 'locked_config.json').write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    selected = comparison.iloc[0]
    baseline = comparison.iloc[1]
    lines = [
        '# New grouping and repurchase model: unified historical playback', '',
        'The new K-means 4 group and the new phase five selected 14 features XGBoost have been loaded into the same model package.'
        'The models are only fitted using the original training period; the scoring and grouping results of the validation period are reproduced one by one with the upstream products.',
        '2011 years 9 – 11 months were viewed in earlier projects, the following examinations are retrospective diagnostics,'
        'It cannot be called a new independent test. Repurchase scores are not calibrated for probability and are only used for relative ranking.', '',
        '| Method | AP | Top 20% Accuracy | ROC-AUC | Brier |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in comparison.itertuples():
        lines.append(f'| {row.model}（{row.feature_set}） | '
                     f'{row.average_precision:.4f} | {row.precision_at_20pct:.2%} | '
                     f'{row.roc_auc:.4f} | {row.brier:.4f} |')
    lines += [
        '', 'Check monthly:',
        '| Month | Method | AP | Top 20% Precision |',
        '| --- | --- | ---: | ---: |',
    ]
    for row in monthly.itertuples():
        lines.append(f'| {row.snapshot_date} | {row.model} | '
                     f'{row.average_precision:.4f} | {row.precision_at_20pct:.2%} |')
    lines += [
        '', f'XGBoost relative to basic logistic regression: AP is worse than {selected.average_precision - baseline.average_precision:+.4f},'
        f'Top 20% accuracy difference {(selected.precision_at_20pct - baseline.precision_at_20pct) * 100:+.2f} percentage points.',
        'This illustrates the ranking differences on the same batch of historical records and does not infer that reaching customers increases revenue.',
        '', 'Run: `python -B -m src.integration.models`;'
        'Page: `python -m streamlit run app/dashboard.py`.'
        'Item-by-item scoring is at `test_predictions.csv`, and the dashboard export does not include post-event tags.',
    ]
    (OUTPUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Integrated {len(test_input)} retrospective customer-months; '
          f'AP={selected.average_precision:.4f}; groups={len(SEGMENT_NAMES)}', flush=True)
    print(f'Artifacts: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
