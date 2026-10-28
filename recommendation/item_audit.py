"""Maintained full-model replay and paired feature-gain audit, without reselection."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from recommendation.data import ROOT, read_inputs, sha256
from recommendation.profiles import FEATURES
from recommendation.metrics import paired_intervals
from recommendation.experiment import write_json
from recommendation.item_features import item_context
from recommendation.item_models import score_catalog
from recommendation.item_experiment import OUTPUT as SOURCE, old_predictions
from app.item_recommendation_data import load_item_package

OUTPUT = ROOT / 'recommendation/outputs/item_v3_1_audit'


def run():
    source_hash = sha256(SOURCE / 'manifest.json')
    code_hash = sha256(Path(__file__))
    if (OUTPUT / 'manifest.json').exists():
        manifest = json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['source_manifest_sha256'] != source_hash or manifest['code_sha256'] != code_hash:
            raise ValueError('Audit source or code changed; preserve audit and use another version')
        for name, digest in manifest['output_hashes'].items():
            path = (OUTPUT / name).resolve()
            if path.parent != OUTPUT.resolve() or sha256(path) != digest:
                raise ValueError('Completed item audit output changed')
        print('Verified completed full item audit', flush=True)
        return
    package = load_item_package()
    bundle = joblib.load(SOURCE / 'model_bundle.joblib')
    old_path = ROOT / 'recommendation/outputs/profile_v2/model_bundle.joblib'
    if sha256(old_path) != bundle['prior_bundle_sha256']:
        raise ValueError('Frozen old recommendation model changed')
    old_bundle = joblib.load(old_path)
    tx, snapshots, _ = read_inputs()
    lock_hash = sha256(SOURCE / 'locked_selection.json')
    verified = 0
    for date in sorted(package['profiles'].snapshot_date.unique()):
        phase = 'validation' if date < '2011-09-01' else 'test'
        context = item_context(tx, snapshots, pd.Timestamp(date), bundle[phase]['ease'])
        expected_profiles = package['profiles'].loc[package['profiles'].snapshot_date.eq(date)].sort_values('customer_id')
        np.testing.assert_allclose(context['profiles'][FEATURES].to_numpy(), expected_profiles[FEATURES].to_numpy(),
                                   atol=1e-10, rtol=1e-10, equal_nan=True)
        ranks, values = score_catalog(bundle[phase]['models'], context)
        baseline_ranks, baseline_values = old_predictions(old_bundle, phase, context)
        ranks.update(baseline_ranks)
        values.update(baseline_values)
        catalog = context['catalog'].stock_code.to_numpy()
        idx = np.arange(len(context['customers']))[:, None]
        for method, ranking in ranks.items():
            stored = package['lists'].loc[package['lists'].snapshot_date.eq(date)
                                          & package['lists'].method.eq(method)].sort_values('customer_id')
            np.testing.assert_array_equal(stored.customer_id, context['customers'])
            np.testing.assert_array_equal(np.asarray(stored.recommended_items.map(json.loads).tolist()), catalog[ranking])
            np.testing.assert_array_equal(np.asarray(stored.purchase_counts_180d.map(json.loads).tolist()),
                                          context['counts_dense'][idx, ranking])
            np.testing.assert_allclose(np.asarray(stored.days_since_item_purchase.map(json.loads).tolist()),
                                       context['item_last'][idx, ranking], atol=1e-10)
            np.testing.assert_allclose(np.asarray(stored.ranking_scores.map(json.loads).tolist()),
                                       values[method], atol=2e-10, rtol=1e-10)
            verified += len(stored)
        print(f'Full item replay: {date}, all 15 methods matched', flush=True)
    old_months = pd.read_csv(ROOT / 'recommendation/outputs/profile_v2/monthly_comparison.csv')
    keys = ['snapshot_date', 'method', 'phase']
    controls = package['monthly'].loc[package['monthly'].method.isin(['personal_repeat', 'pure_ease', 'old_adaptive_ease'])].copy()
    controls['method'] = controls.method.replace({'pure_ease': 'ease', 'old_adaptive_ease': 'adaptive_ease'})
    paired = controls.merge(old_months, on=keys, validate='one_to_one', suffixes=('_new', '_old'))
    for metric in ('ndcg_10', 'recall_10', 'hit_10', 'new_recall_10', 'repeat_recall_10'):
        np.testing.assert_allclose(paired[metric + '_new'], paired[metric + '_old'], atol=1e-12)
    # Only validation outcomes enter gain diagnostics; the review never changes parameters or selection.
    validation = pd.read_csv(SOURCE / 'validation_customers.csv', dtype={'customer_id': 'string'})
    best = package['lock']['best_new']
    chosen = validation.loc[validation.method.eq(best)]
    draws, interval = paired_intervals(chosen, validation.loc[validation.method.eq('personal_repeat')])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    draws.to_csv(OUTPUT / 'best_vs_default_draws.csv', index=False)
    report = dict(status='completed_full_item_audit', verified_customer_lists=verified,
                  verified_product_rows=verified * 10, verified_profile_rows=len(package['profiles']),
                  old_control_months_unchanged=len(paired), best_new=best, best_vs_default=interval,
                  source_manifest_sha256=source_hash, code_sha256=code_hash)
    if best[0] in ('B', 'D'):
        group, family = best.split('_', 1)
        reference = f'{"A" if group == "B" else "C"}_{family}'
        draws, interval = paired_intervals(chosen, validation.loc[validation.method.eq(reference)])
        draws.to_csv(OUTPUT / 'best_profile_draws.csv', index=False)
        report.update(profile_reference=reference, best_profile_gain=interval)
    if sha256(SOURCE / 'locked_selection.json') != lock_hash or sha256(SOURCE / 'manifest.json') != source_hash:
        raise ValueError('Audit changed the locked model selection')
    write_json(OUTPUT / 'report.json', report)
    files = sorted(p for p in OUTPUT.iterdir() if p.is_file() and p.name != 'manifest.json')
    write_json(OUTPUT / 'manifest.json', dict(source_manifest_sha256=source_hash, code_sha256=code_hash,
               output_hashes={p.name: sha256(p) for p in files}))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
