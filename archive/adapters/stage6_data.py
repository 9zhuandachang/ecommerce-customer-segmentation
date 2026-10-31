"""Verify and expose the integrated four-group/XGBoost historical replay."""

from __future__ import annotations

from src.paths import ROOT

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from src.repurchase.ranking import review_list
from src.data.validation import KEYS, LABEL, load_snapshots
from src.repurchase.comparison import (TRANSACTIONS, add_features, evaluate,
                                        read_invoice_dates)
from src.integration.models import (OUTPUT, REPURCHASE_DIR, ROOT, SEGMENT_DIR,
                                         SNAPSHOTS, VERSION, as_dashboard_rows,
                                         monthly_top20_mask, score)

SOURCE = SNAPSHOTS
FINAL = OUTPUT
MODEL = FINAL / 'model/integrated_bundle.joblib'
LOCK = FINAL / 'locked_config.json'
SAVED = FINAL / 'test_predictions.csv'
COMPARISON = FINAL / 'test_comparison.csv'
MONTHLY = FINAL / 'monthly_comparison.csv'


def sha256(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()




def export_review(frame: pd.DataFrame, version: str) -> pd.DataFrame:
    """Only historical information known at the cutoff may be exported."""
    columns = ['CustomerID', 'segment_name', 'repurchase_score', 'rank',
               'recency_days', 'frequency', 'monetary', 'snapshot_date']
    if not set(columns).issubset(frame.columns):
        raise ValueError('Review list lacks required export fields')
    out = frame[columns].copy()
    out['model_version'] = version
    return out


def verify_hashes(lock: dict) -> None:
    source_paths = {
        'snapshots': SOURCE,
        'transactions': TRANSACTIONS,
        'segment_model': SEGMENT_DIR / 'selected_model.joblib',
        'segment_decision': SEGMENT_DIR / 'selected.json',
        'segment_assignments': SEGMENT_DIR / 'selected_assignments.csv',
        'repurchase_model': REPURCHASE_DIR / 'selected_model.joblib',
        'repurchase_manifest': REPURCHASE_DIR / 'run_manifest.json',
        'repurchase_validation_predictions': REPURCHASE_DIR / 'validation_predictions.csv',
        'integration_script': ROOT / 'src/integration/models.py',
    }
    output_paths = {'model': MODEL, 'test_predictions': SAVED,
                    'test_comparison': COMPARISON,
                    'monthly_comparison': MONTHLY}
    for paths, expected in ((source_paths, lock['source_hashes']),
                            (output_paths, lock['output_hashes'])):
        if set(paths) != set(expected):
            raise ValueError('Unexpected source or output hash inventory')
        for key, path in paths.items():
            if sha256(path) != expected[key]:
                raise ValueError(f'Integrated replay file changed: {path}')


def load_stage6_replay() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Recompute every supported score, then return a label-free review table."""
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    if (lock['version'] != VERSION
            or lock['source_scope'] != 'historical_replay_test_months_previously_viewed'
            or lock['test_labels_used_for_selection'] is not False):
        raise ValueError('Unexpected integrated replay lock')
    verify_hashes(lock)
    bundle = joblib.load(MODEL)
    if (bundle['version'] != lock['version']
            or bundle['model_name'] != lock['model_name']
            or bundle['feature_set'] != lock['feature_set']
            or {str(k): v for k, v in bundle['segment_names'].items()}
            != lock['segment_names']):
        raise ValueError('Integrated model differs from frozen configuration')

    snapshots, _ = load_snapshots(SOURCE)
    test = snapshots.loc[snapshots.split.eq('test')].reset_index(drop=True)
    dates = sorted(test.snapshot_date.dt.strftime('%Y-%m-%d').unique().tolist())
    if dates != lock['test_dates'] or len(test) != lock['counts']['test']:
        raise ValueError('Unexpected dashboard cohort')
    features = add_features(test, read_invoice_dates(TRANSACTIONS))
    scores = score(bundle, features)
    saved = pd.read_csv(SAVED, dtype={'customer_id': 'string'},
                        parse_dates=['snapshot_date'])
    if (len(saved) != len(test)
            or saved.duplicated(['snapshot_date', 'customer_id']).any()
            or not saved[KEYS].reset_index(drop=True).equals(features[KEYS])):
        raise ValueError('Saved replay customer-months differ')
    np.testing.assert_allclose(saved.repurchase_score.to_numpy(),
                               scores.score.to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(saved.cluster.to_numpy(), scores.cluster.to_numpy())
    np.testing.assert_array_equal(saved.top20_selected.to_numpy(dtype=bool),
                                  monthly_top20_mask(features, scores.score.to_numpy()))
    if not saved[LABEL].reset_index(drop=True).equals(test[LABEL]):
        raise ValueError('Saved retrospective labels differ')
    expected = as_dashboard_rows(features, scores, bundle['segment_names'])
    if not saved.segment_name.equals(expected.segment_name.reset_index(drop=True)):
        raise ValueError('Saved group names differ')

    comparison = pd.read_csv(COMPARISON)
    monthly = pd.read_csv(MONTHLY)
    if (set(comparison.model) != {lock['model_name'], 'LR'}
            or comparison.snapshot_count.nunique() != 1
            or int(comparison.snapshot_count.iloc[0]) != len(test)
            or set(monthly.snapshot_date) != set(dates)
            or set(monthly.model) != set(comparison.model)
            or len(monthly) != len(dates) * len(comparison)):
        raise ValueError('Integrated comparison does not match dashboard cohort')
    selected = comparison.loc[comparison.model.eq(lock['model_name'])].iloc[0]
    actual = evaluate(test, scores.score.to_numpy())
    for column, value in (('average_precision', actual['ap']),
                          ('precision_at_20pct', actual['precision_at_20pct'])):
        if not np.isclose(selected[column], value, rtol=0, atol=1e-12):
            raise ValueError(f'Saved comparison metric differs: {column}')

    replay = expected.rename(columns={
        'snapshot_date': 'cutoff_date', 'customer_id': 'CustomerID',
        'invoice_frequency_180d': 'frequency',
        'monetary_value_180d': 'monetary',
        'average_order_value_180d': 'aov',
        'product_diversity_180d': 'product_diversity',
    })
    replay['snapshot_date'] = replay.cutoff_date.dt.strftime('%Y-%m-%d')
    if (LABEL in replay.columns or replay.CustomerID.isna().any()
            or replay.duplicated(['snapshot_date', 'CustomerID']).any()):
        raise ValueError('Invalid label-free dashboard rows')
    metadata = {
        'model_name': bundle['model_name'],
        'feature_set': bundle['feature_set'],
        'version': f"sha256:{lock['output_hashes']['model'][:12]}",
        'test_dates': dates,
        'cluster_names': bundle['segment_names'],
        'source_scope': lock['source_scope'],
        'score_kind': bundle['score_kind'],
        'monthly_comparison': monthly,
    }
    return replay, metadata, comparison
