"""Versioned purchase-habit and hybrid comparison: python -B -m recommendation.profile_experiment."""

from __future__ import annotations

from src.paths import ROOT

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd

from src.recommendation.data import (ROOT, make_context, make_training, next_basket_labels,
                                  read_inputs, sha256, source_hashes)
from pipelines.recommendation_baselines import (TUNE_DATES, VALIDATION_DATES, TEST_DATES,
                                        write_json, training_record)
from src.recommendation.algorithms import FAMILIES as BASE_FAMILIES, fit_model, scores, topk
from src.recommendation.metrics import evaluate, summarize, paired_intervals
from src.recommendation.profiles import (FEATURES, FAMILIES as PROP_FAMILIES, PROPENSITY_NAMES,
                                     purchase_profiles, attach_repeat_target, propensity_specs,
                                     fit_propensity, predict_propensity, propensity_metrics)
from archive.recommendation.hybrid import (LEARNERS, NAMES, hybrid_specs, mixing_weights,
                                   prepare_routes, fuse_routes, metric_context, fast_metrics)

OUTPUT = ROOT / 'archive/artifacts/recommendation/profile_v2'
PROTOCOL = ROOT / 'docs/protocols/recommendation_profiles.md'
METHODS = list(NAMES)
CODE_FILES = ['src/recommendation/profiles.py', 'archive/recommendation/hybrid.py', 'pipelines/recommendation_profiles.py', 'docs/protocols/recommendation_profiles.md', 'src/recommendation/data.py', 'src/recommendation/algorithms.py', 'src/recommendation/metrics.py', 'pipelines/recommendation_baselines.py', 'src/paths.py']


def save(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUTPUT / name, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d %H:%M:%S')


def check_completed() -> dict:
    manifest = json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed_profile_recommendation' or manifest['source_hashes'] != source_hashes():
        raise ValueError('Profile recommendation source/status changed')
    for name, digest in manifest['code_hashes'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT.resolve()) or sha256(path) != digest:
            raise ValueError(f'Profile implementation changed: {name}')
    for name, digest in manifest['output_hashes'].items():
        path = (OUTPUT / name).resolve()
        if path.parent != OUTPUT.resolve() or sha256(path) != digest:
            raise ValueError(f'Profile output changed: {name}')
    if sha256(ROOT / 'archive/artifacts/recommendation/v1/manifest.json') != manifest['v1_manifest_sha256']:
        raise ValueError('Original recommendation comparison changed')
    return manifest


def choose_hybrid_specs(monthly: pd.DataFrame) -> dict:
    data = monthly.loc[monthly.phase.eq('development')]
    means = data.groupby(['method', 'config_id']).ndcg_10.mean()
    choices = {}
    for algorithm in LEARNERS:
        for adaptive in (False, True):
            options = hybrid_specs(algorithm, adaptive)
            method = options[0]['method']
            for spec in options:
                part = data.loc[data.config_id.eq(spec['id'])]
                if len(part) != 5 or set(part.snapshot_date) != set(TUNE_DATES.strftime('%Y-%m-%d')):
                    raise ValueError('Incomplete forward hybrid budget')
            choices[method] = min(options, key=lambda spec: (-means.loc[(method, spec['id'])],
                                                             spec['strength'], -spec['intercept']))
    return choices


def pair_conditions(monthly: pd.DataFrame, challenger: str, reference: str) -> dict:
    data = monthly.loc[monthly.phase.eq('validation')]
    a = data.loc[data.method.eq(challenger)].set_index('snapshot_date')
    b = data.loc[data.method.eq(reference)].set_index('snapshot_date')
    expected = set(VALIDATION_DATES.strftime('%Y-%m-%d'))
    if len(a) != 3 or len(b) != 3 or set(a.index) != expected or set(b.index) != expected:
        raise ValueError('Expected the same three validation months')
    gains = a.ndcg_10 - b.ndcg_10
    return dict(ndcg_gain=float(gains.mean()), recall_gain=float((a.recall_10 - b.recall_10).mean()),
                nonworse_months=int((gains >= -1e-12).sum()),
                passes=bool(gains.mean() >= .002 - 1e-12
                            and (gains >= -1e-12).sum() >= 2
                            and (a.recall_10 - b.recall_10).mean() >= -.005 - 1e-12))


def choose_default(monthly: pd.DataFrame) -> dict:
    data = monthly.loc[monthly.phase.eq('validation')]
    if set(data.method) != set(METHODS):
        raise ValueError('Incomplete four-group comparison')
    means = data.groupby('method').ndcg_10.mean().to_dict()
    def best(options):
        return max(options, key=lambda method: (means[method], -METHODS.index(method)))
    fixed = best([f'fixed_{name}' for name in LEARNERS])
    adaptive = best([f'adaptive_{name}' for name in LEARNERS])
    pure = best(list(BASE_FAMILIES))
    vs_fixed = pair_conditions(data, adaptive, fixed)
    vs_repeat = pair_conditions(data, adaptive, 'personal_repeat')
    novel_gain = float(data.loc[data.method.eq(adaptive), 'new_recall_10'].mean()
                       - data.loc[data.method.eq('personal_repeat'), 'new_recall_10'].mean())
    profile_value = bool(vs_fixed['passes'] and vs_repeat['passes'] and novel_gain >= .005 - 1e-12)
    selected = 'personal_repeat'
    for method in (fixed, pure):
        if means[method] > means[selected] and pair_conditions(data, method, 'personal_repeat')['passes']:
            selected = method
    if profile_value and means[adaptive] > means[selected]:
        selected = adaptive
    return dict(selected_method=selected, selected_name=NAMES[selected], best_fixed=fixed,
                best_adaptive=adaptive, best_pure=pure, profile_value_demonstrated=profile_value,
                adaptive_vs_fixed=vs_fixed, adaptive_vs_repeat=vs_repeat,
                adaptive_new_recall_gain_vs_repeat=novel_gain,
                selection_phase='validation', test_labels_used_for_selection=False,
                prior_test_results_viewed=True, global_test_model_updates=False)


def fit_recommenders(tx, date, specs, audit, phase):
    training = make_training(tx, date)
    audit.append(training_record(training, phase))
    models = {method: fit_model(specs[method], training) for method in BASE_FAMILIES}
    return models


def recommend_all(models, propensity, specs, tx, snapshots, date, profiles):
    catalog = models['personal_repeat']['catalog']
    context = make_context(tx, snapshots, date, catalog)
    profile = profiles.loc[profiles.snapshot_date.eq(date)].set_index('customer_id').loc[context['customers']].reset_index()
    predictions = predict_propensity(propensity, profile)
    personal = context['counts'].toarray()
    ranks, weights = {}, {}
    for method, model in models.items():
        raw = scores(model, context)
        ranks[method] = topk(raw)
        weights[method] = np.full(len(profile), np.nan)
        if method in LEARNERS:
            routes = prepare_routes(personal, raw)
            for prefix in ('fixed', 'adaptive'):
                name = f'{prefix}_{method}'
                weights[name] = mixing_weights(specs[name], predictions, propensity['prior'])
                ranks[name] = fuse_routes(routes, weights[name])
    profile['predicted_repeat_share'] = predictions
    profile['propensity_method'] = propensity['spec']['family']
    return context, profile, ranks, weights


def phase_run(models, prop_models, selected_prop, hybrid_choices, tx, snapshots, dates, profiles, phase):
    monthly, customers, lists, displayed_profiles, prop_metrics, prop_predictions, groups = [], [], [], [], [], [], []
    catalog = models['personal_repeat']['catalog'].stock_code.to_numpy()
    for date in dates:
        context, profile, all_ranks, weights = recommend_all(
            models, prop_models[selected_prop['family']], hybrid_choices, tx, snapshots, date, profiles)
        labels = next_basket_labels(tx, snapshots, date)
        labeled_profile = attach_repeat_target(profile, labels)
        displayed_profiles.append(profile)
        for family, model in prop_models.items():
            predicted = predict_propensity(model, profile)
            prop_metrics.append(dict(phase=phase, snapshot_date=str(date.date()), family=family,
                                     config_id=model['spec']['id'], **propensity_metrics(predicted, labeled_profile)))
            prop_predictions.append(labeled_profile[['snapshot_date', 'customer_id', 'repeat_share_target']]
                                    .assign(prediction=predicted, family=family, phase=phase))
        old = context['counts'].toarray() > 0
        for method, ranks in all_ranks.items():
            spec_id = hybrid_choices[method]['id'] if method in hybrid_choices else models[method]['spec']['id']
            metric, user_rows = evaluate(ranks, context, labels, method, phase, spec_id)
            monthly.append(metric)
            customers.append(user_rows)
            product_ids = catalog[ranks]
            flags = old[np.arange(len(ranks))[:, None], ranks]
            lists.append(pd.DataFrame(dict(snapshot_date=str(date.date()), customer_id=context['customers'],
                                           method=method, recommended_items=[json.dumps(list(x)) for x in product_ids],
                                           prior_purchase_flags=[json.dumps(x.tolist()) for x in flags],
                                           repeat_weight=weights[method])))
            grouped = user_rows.merge(profile[['customer_id', 'habit_group']], on='customer_id', validate='one_to_one')
            grouped = grouped.loc[grouped.target_count.gt(0)]
            for group, part in grouped.groupby('habit_group'):
                groups.append(dict(phase=phase, snapshot_date=str(date.date()), method=method, habit_group=group,
                                   buyers=len(part), ndcg_10=float(part.ndcg_10.mean()), recall_10=float(part.recall_10.mean()),
                                   repeat_recall_10=float(part.repeat_recall_10.mean()),
                                   new_recall_10=float(part.new_recall_10.mean())))
        print(f'{phase}: {date.date()} | all 11 methods evaluated', flush=True)
    return dict(monthly=pd.DataFrame(monthly), customers=pd.concat(customers, ignore_index=True),
                lists=pd.concat(lists, ignore_index=True), profiles=pd.concat(displayed_profiles, ignore_index=True),
                propensity_metrics=pd.DataFrame(prop_metrics), propensity_predictions=pd.concat(prop_predictions, ignore_index=True),
                groups=pd.DataFrame(groups))


def run() -> None:
    started = time.perf_counter()
    if (OUTPUT / 'manifest.json').exists():
        check_completed()
        print(f'Verified completed profile recommendation: {OUTPUT}', flush=True)
        return
    from archive.adapters.recommendation_data import load_recommendations
    load_recommendations()  # Check all original hashes before reusing its locked development shortlist.
    tx, snapshots, input_audit = read_inputs()
    base_specs = json.loads((ROOT / 'archive/artifacts/recommendation/v1/shortlist.json').read_text(encoding='utf-8'))
    plan = dict(version='profile_recommendation_v2', source_hashes=input_audit['source_hashes'],
                code_hashes={name: sha256(ROOT / name) for name in CODE_FILES},
                protocol_sha256=sha256(PROTOCOL), base_specs=base_specs, propensity_specs=propensity_specs(),
                hybrid_specs=[s for a in LEARNERS for adaptive in (False, True) for s in hybrid_specs(a, adaptive)],
                v1_manifest_sha256=sha256(ROOT / 'archive/artifacts/recommendation/v1/manifest.json'),
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                time_policy='labels must end <= fitting cutoff; frozen globals per phase; pre-cutoff customer input',
                prior_test_results_viewed=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / 'locked_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Profile plan/code changed; preserve this run and use a new version')
    if (OUTPUT / 'locked_selection.json').exists():
        raise ValueError('Selection already locked; do not rerun an incomplete review automatically')
    write_json(plan_path, plan)
    write_json(OUTPUT / 'input_audit.json', input_audit)
    profile_parts = []
    for date in sorted(snapshots.snapshot_date.unique()):
        date = pd.Timestamp(date)
        customers = snapshots.loc[snapshots.snapshot_date.eq(date), 'customer_id'].sort_values()
        profile_parts.append(purchase_profiles(tx, customers, date))
    profiles = pd.concat(profile_parts, ignore_index=True)
    # Only train/development/validation labels exist here. No review labels before the lock.
    training_parts = []
    for date in sorted(snapshots.loc[snapshots.snapshot_date.lt(TEST_DATES.min()), 'snapshot_date'].unique()):
        date = pd.Timestamp(date)
        training_parts.append(attach_repeat_target(profiles.loc[profiles.snapshot_date.eq(date)],
                                                   next_basket_labels(tx, snapshots, date)))
    examples = pd.concat(training_parts, ignore_index=True)
    save('training_repeat_share_labels.csv', examples)
    prop_tuning, prop_audit = [], []
    for date in TUNE_DATES:
        hold = examples.loc[examples.snapshot_date.eq(date)]
        for spec in propensity_specs():
            model = fit_propensity(spec, examples, date)
            prediction = predict_propensity(model, hold)
            prop_tuning.append(dict(snapshot_date=str(date.date()), family=spec['family'], config_id=spec['id'],
                                    **propensity_metrics(prediction, hold)))
            prop_audit.append({key: model[key] for key in ('fit_cutoff', 'fit_rows', 'latest_label_end', 'spec')})
        print(f'Repeat-share forward comparison: {date.date()}', flush=True)
    prop_tuning = pd.DataFrame(prop_tuning)
    save('propensity_development_monthly.csv', prop_tuning)
    prop_means = prop_tuning.groupby('config_id').mae.mean().to_dict()
    prop_shortlist = {family: min([s for s in propensity_specs() if s['family'] == family],
                                  key=lambda s: (prop_means[s['id']], propensity_specs().index(s)))
                      for family in PROP_FAMILIES}
    selected_prop = min(prop_shortlist.values(), key=lambda s: (prop_means[s['id']], PROP_FAMILIES.index(s['family'])))
    write_json(OUTPUT / 'propensity_shortlist.json', dict(shortlist=prop_shortlist, selected=selected_prop))
    print(f'Forward repeat-share method selected: {selected_prop}', flush=True)
    tuning, recommender_audit = [], []
    for date in TUNE_DATES:
        prop = fit_propensity(selected_prop, examples, date)
        hold = examples.loc[examples.snapshot_date.eq(date)].sort_values('customer_id')
        predictions = predict_propensity(prop, hold)
        models = fit_recommenders(tx, date, base_specs, recommender_audit, 'development')
        context = make_context(tx, snapshots, date, models['personal_repeat']['catalog'])
        if list(hold.customer_id) != list(context['customers']):
            raise ValueError('Forward profile and recommendation cohorts differ')
        truth = metric_context(context, next_basket_labels(tx, snapshots, date))
        personal = context['counts'].toarray()
        for algorithm in LEARNERS:
            routes = prepare_routes(personal, scores(models[algorithm], context))
            for adaptive in (False, True):
                for spec in hybrid_specs(algorithm, adaptive):
                    ranks = fuse_routes(routes, mixing_weights(spec, predictions, prop['prior']))
                    tuning.append(dict(phase='development', snapshot_date=str(date.date()), method=spec['method'],
                                       config_id=spec['id'], **fast_metrics(ranks, truth)))
        print(f'Hybrid forward comparison: {date.date()} | 132 configurations', flush=True)
    tuning = pd.DataFrame(tuning)
    save('hybrid_development_monthly.csv', tuning)
    hybrid_choices = choose_hybrid_specs(tuning)
    write_json(OUTPUT / 'hybrid_shortlist.json', hybrid_choices)
    validation_models = fit_recommenders(tx, VALIDATION_DATES.min(), base_specs, recommender_audit, 'validation')
    validation_props = {family: fit_propensity(spec, examples, VALIDATION_DATES.min())
                        for family, spec in prop_shortlist.items()}
    validation = phase_run(validation_models, validation_props, selected_prop, hybrid_choices, tx, snapshots,
                           VALIDATION_DATES, profiles, 'validation')
    decision = choose_default(validation['monthly'])
    for name, frame in validation.items():
        save(f'validation_{name}.csv', frame)
    draw, interval = paired_intervals(validation['customers'].loc[validation['customers'].method.eq(decision['best_adaptive'])],
                                      validation['customers'].loc[validation['customers'].method.eq(decision['best_fixed'])])
    save('validation_paired_draws.csv', draw)
    decision['adaptive_vs_fixed_intervals'] = interval
    final_models = fit_recommenders(tx, TEST_DATES.min(), base_specs, recommender_audit, 'review_frozen')
    final_props = {family: fit_propensity(spec, examples, TEST_DATES.min()) for family, spec in prop_shortlist.items()}
    artifact = dict(validation=dict(recommenders=validation_models, propensity_models=validation_props),
                    test=dict(recommenders=final_models, propensity_models=final_props),
                    selected_propensity=selected_prop, hybrid_specs=hybrid_choices, selected_method=decision['selected_method'])
    joblib.dump(artifact, OUTPUT / 'model_bundle.joblib', compress=3)
    write_json(OUTPUT / 'training_audit.json', dict(recommenders=recommender_audit,
               propensity_forward=prop_audit, propensity_final=[{key: model[key] for key in ('fit_cutoff', 'fit_rows', 'latest_label_end', 'spec')}
               for phase in (validation_props, final_props) for model in phase.values()]))
    lock = dict(**decision, selected_propensity=selected_prop, hybrid_specs=hybrid_choices, base_specs=base_specs,
                model_sha256=sha256(OUTPUT / 'model_bundle.joblib'), plan_sha256=sha256(plan_path),
                source_hashes=input_audit['source_hashes'], features=FEATURES)
    lock_path = OUTPUT / 'locked_selection.json'
    write_json(lock_path, lock)
    lock_hash = sha256(lock_path)
    print(f'Selection locked: {lock["selected_name"]}; profile value={lock["profile_value_demonstrated"]}. Now reviewing Sep-Nov.', flush=True)
    test = phase_run(final_models, final_props, selected_prop, hybrid_choices, tx, snapshots, TEST_DATES, profiles, 'test')
    if sha256(lock_path) != lock_hash or sha256(OUTPUT / 'model_bundle.joblib') != lock['model_sha256']:
        raise ValueError('Review mutated locked selection/model')
    for name, frame in test.items():
        save(f'test_{name}.csv', frame)
    monthly = pd.concat([validation['monthly'], test['monthly']], ignore_index=True)
    comparison = summarize(monthly)
    save('monthly_comparison.csv', monthly)
    save('comparison.csv', comparison)
    safe_profiles = pd.concat([validation['profiles'], test['profiles']], ignore_index=True)
    save('profiles.csv', safe_profiles)
    for phase, models in [('validation', validation_models), ('test', final_models)]:
        save(f'{phase}_catalog.csv', models['personal_repeat']['catalog'])
    write_report(comparison, lock, pd.concat([validation['propensity_metrics'], test['propensity_metrics']]),
                 profiles, time.perf_counter() - started)
    files = sorted(p for p in OUTPUT.iterdir() if p.is_file() and p.name != 'manifest.json')
    write_json(OUTPUT / 'manifest.json', dict(status='completed_profile_recommendation', selected_method=lock['selected_method'],
               source_hashes=input_audit['source_hashes'], code_hashes=plan['code_hashes'],
               v1_manifest_sha256=plan['v1_manifest_sha256'], output_hashes={p.name: sha256(p) for p in files},
               elapsed_seconds=time.perf_counter() - started, review_after_selection_lock=True))
    print(f'Completed: {OUTPUT}', flush=True)


def write_report(comparison, lock, propensity_monthly, profiles, seconds):
    lines = ['# Purchasing Habits Profiles and Product Recommendations: Comparison Results', '', f'Default scheme: **{lock["selected_name"]}**.', '',
             'The profile is based on the statistics of orders 180 days before the cutoff point; the machine learning prediction is "if there is an order 30 days later, how much of the old products will account for it",'
             'Not the probability of repurchase. The recommendation weight is determined by the ranking effect during the development period and is not directly equivalent to proportional prediction.', '',
             f'Whether the profile value threshold is met: **{lock["profile_value_demonstrated"]}**.', '',
             '| Plan | Verify NDCG@10 | Verify Recall@10 | Verify but not bought Recall | Review NDCG@10 | Review Recall@10 | Review but not buy Recall |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        a = comparison.loc[comparison.phase.eq('validation') & comparison.method.eq(method)].iloc[0]
        b = comparison.loc[comparison.phase.eq('test') & comparison.method.eq(method)].iloc[0]
        lines.append(f'| {NAMES[method]} | {a.mean_ndcg_10:.4f} | {a.mean_recall_10:.2%} | {a.mean_new_recall_10:.2%} | '
                     f'{b.mean_ndcg_10:.4f} | {b.mean_recall_10:.2%} | {b.mean_new_recall_10:.2%} |')
    lines += ['', 'Each month, it is first calculated among customers with next orders, and then the monthly equal-weighted average is calculated. Everyone recommends 10 pieces;'
              'Unbought means no purchases were made 180 days before the cutoff, and unknown products retain the denominator of the answer. Each pure algorithm can still recommend old products; the mixed old/unbought two ways are mutually exclusive.', '',
              '## Profile Gain and Final Selection', '',
              f'- Strongest fixed mix: {NAMES[lock["best_fixed"]]}. The strongest image blend: {NAMES[lock["best_adaptive"]]}.',
              f'- The profile is relatively fixed: {json.dumps(lock["adaptive_vs_fixed"], ensure_ascii=False)}.',
              f'- The profile is relatively pure replenishment: {json.dumps(lock["adaptive_vs_repeat"], ensure_ascii=False)}.',
              f'- The image of the unbought product Recall is improved compared to pure replenishment: {lock["adaptive_new_recall_gain_vs_repeat"]:+.2%}.',
              f'- Paired customer sampling interval: `{json.dumps(lock["adaptive_vs_fixed_intervals"]["metrics"], ensure_ascii=False)}`.',
              'The range is internal diagnosis after selection; no new monthly income is promised. The imaging solution must exceed both fixed mix and pure replenishment,'
              'Simply adding a model cannot be called an improvement. See locked_selection.json for final parameters.', '',
              '## Conditional repetition ratio prediction', '',
              f'The method selected during the development period: {PROPENSITY_NAMES[lock["selected_propensity"]["family"]]},'
              f'Parameter `{json.dumps(lock["selected_propensity"], ensure_ascii=False)}`.', '',
              '| Method | Verify MAE | Verify RMSE | Verify mean deviation | Review MAE |', '|---|---:|---:|---:|---:|']
    for family in PROP_FAMILIES:
        a = propensity_monthly.loc[propensity_monthly.phase.eq('validation') & propensity_monthly.family.eq(family)]
        b = propensity_monthly.loc[propensity_monthly.phase.eq('test') & propensity_monthly.family.eq(family)]
        lines.append(f'| {PROPENSITY_NAMES[family]} | {a.mae.mean():.4f} | {a.rmse.mean():.4f} | {a.bias.mean():+.4f} | {b.mae.mean():.4f} |')
    lines += ['', 'MAE = the average absolute difference between the predicted proportion and the actual proportion, for example, 0.15 means the average difference is 15 percentage points.'
              'The first order has no previous reference and is not recorded as a new one; when there is no comparable record, the mean answer value during the fitting period is used.'
              'Recent changes require at least 4 comparable orders, the rest are missing. Both the record quantity and missing status are entered into audit.', '',
              '## Time, fair comparison and opening', '',
              '1 – 5 month forward development: Each fold fit only uses answers that have been fully observed 30 days as of that fold; 6 – 8 month validation determines the default.'
              'Models and selections are locked before calculating the 9 – 11 monthly mixed recommendation results. 9 – 11 Monthly answers have been used previously for v 1 and are a review, not a new independent test.'
              'The global model and catalog are frozen during the stage, and customer history is only updated to the cut-off point of the month.', '',
              'Fixed mixing per algorithm 11 group, profile mixing per algorithm 33 group, both report complete development results.'
              'See validation_groups.csv/test_groups.csv for profile group performance; the safe list and offline answers are saved in separate files.', '',
              '`python -B -m pipelines.recommendation_profiles`', '',
              '`python -m streamlit run app/dashboard.py` → **Product recommendations** → Select the profile recommended version.', '',
              f'All 18 Number of monthly profile lines {len(profiles):,}; run for about {seconds:.0f} seconds. Models, directories, and CSV all have hash checks.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
