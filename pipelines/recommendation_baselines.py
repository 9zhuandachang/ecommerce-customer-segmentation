"""Run the locked next-basket comparison: python -B -m recommendation.experiment."""

from __future__ import annotations

from src.paths import ROOT

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy

from src.recommendation.data import (ROOT, make_context, make_training, next_basket_labels,
                                  labels_for_export, read_inputs, sha256)
from src.recommendation.metrics import evaluate, paired_intervals, summarize
from src.recommendation.algorithms import (FAMILIES, LEARNERS, NAMES, RULES, configurations,
                                    fit_model, recommend, recommendation_table)

OUTPUT = ROOT / 'archive/artifacts/recommendation/v1'
PROTOCOL = ROOT / 'docs/protocols/recommendation_baselines.md'
TUNE_DATES = pd.date_range('2011-01-01', '2011-05-01', freq='MS')
VALIDATION_DATES = pd.date_range('2011-06-01', '2011-08-01', freq='MS')
TEST_DATES = pd.date_range('2011-09-01', '2011-11-01', freq='MS')


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(OUTPUT / name, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d %H:%M:%S')


def choose_config(summary: pd.DataFrame, method: str, specs: list[dict]) -> dict:
    part = summary.loc[summary.phase.eq('tuning') & summary.method.eq(method)]
    candidates = [spec for spec in specs if spec['method'] == method]
    if len(part) != len(candidates) or not part.months.eq(5).all():
        raise ValueError('Incomplete tuning budget or months')
    lookup = part.set_index('config_id').mean_ndcg_10.to_dict()
    # Candidate order is fixed in the plan; smaller K/factors and then larger regularization break exact ties.
    if method == 'ease':
        candidates = sorted(candidates, key=lambda s: -s['regularization'])
    return min(candidates, key=lambda s: (-lookup[s['id']], candidates.index(s)))


def choose_method(summary: pd.DataFrame, monthly: pd.DataFrame) -> dict:
    valid = summary.loc[summary.phase.eq('validation')]
    months = monthly.loc[monthly.phase.eq('validation')]
    if set(valid.method) != set(FAMILIES) or len(valid) != 5:
        raise ValueError('Incomplete validation method comparison')
    expected = set(VALIDATION_DATES.strftime('%Y-%m-%d'))
    for method in FAMILIES:
        rows = months.loc[months.method.eq(method)]
        if len(rows) != 3 or set(rows.snapshot_date) != expected:
            raise ValueError('Methods must share all three validation months')
    score = valid.set_index('method').mean_ndcg_10.to_dict()
    baseline = max(RULES, key=lambda m: (score[m], -RULES.index(m)))
    learner = max(LEARNERS, key=lambda m: (score[m], -LEARNERS.index(m)))
    a = months.loc[months.method.eq(learner)].set_index('snapshot_date')
    b = months.loc[months.method.eq(baseline)].set_index('snapshot_date')
    gains = a.ndcg_10 - b.ndcg_10
    conditions = dict(ndcg_gain_at_least_002=bool(gains.mean() >= .002 - 1e-12),
                      at_least_two_months_nonworse=bool((gains >= -1e-12).sum() >= 2),
                      mean_recall_loss_at_most_005=bool((a.recall_10 - b.recall_10).mean() >= -.005 - 1e-12))
    selected = learner if all(conditions.values()) else baseline
    return dict(selected_method=selected, selected_name=NAMES[selected], best_rule=baseline, best_learner=learner,
                mean_validation_ndcg_gain=float(gains.mean()), conditions=conditions,
                selection_phase='validation', test_labels_used_for_selection=False,
                prior_other_task_results_viewed=True)


def training_record(training: dict, phase: str) -> dict:
    return dict(phase=phase, fit_cutoff=str(training['cutoff'].date()),
                feature_start=str((training['cutoff'] - pd.Timedelta(days=180)).date()),
                latest_transaction=training['latest_transaction'], fit_customers=training['fit_customers'],
                fit_invoice_item_events=training['fit_events'], catalog_items=len(training['catalog']),
                customer_item_pairs=int(training['counts'].nnz),
                density=float(training['counts'].nnz / np.prod(training['counts'].shape)))


def phase_predictions(models: dict, tx: pd.DataFrame, snapshots: pd.DataFrame, dates: pd.DatetimeIndex,
                       phase: str, chosen: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly, users, label_parts, exports = [], [], [], []
    catalog = next(iter(models.values()))['catalog']
    for date in dates:
        context = make_context(tx, snapshots, date, catalog)
        labels = next_basket_labels(tx, snapshots, date)
        label_parts.append(labels_for_export(labels))
        for method, model in models.items():
            ranks = recommend(model, context)
            metric, rows = evaluate(ranks, context, labels, method, phase, model['spec']['id'], model['seed'])
            monthly.append(metric)
            users.append(rows)
            if method == chosen:
                exports.append(recommendation_table(model, context, ranks))
        print(f'Completed {phase} month {date.date()}', flush=True)
    return (pd.DataFrame(monthly), pd.concat(users, ignore_index=True),
            pd.concat(label_parts, ignore_index=True), pd.concat(exports, ignore_index=True) if exports else pd.DataFrame())


def run() -> None:
    started = time.perf_counter()
    tx, snapshots, input_audit = read_inputs()
    specs = configurations()
    code_files = ['src/recommendation/data.py', 'src/recommendation/algorithms.py', 'src/recommendation/metrics.py', 'pipelines/recommendation_baselines.py', 'docs/protocols/recommendation_baselines.md', 'src/paths.py']
    plan = dict(version='recommendation_v1', source_hashes=input_audit['source_hashes'], specs=specs,
                code_hashes={name: sha256(ROOT / name) for name in code_files},
                target='first purchase timestamp product set in [t,t+30d)', history_days=180, list_length=10,
                allow_repeat_recommendations=True, full_catalog_evaluation=True,
                primary_metric='mean_monthly_buyer_ndcg_at_10',
                tuning_dates=list(TUNE_DATES.strftime('%Y-%m-%d')),
                validation_dates=list(VALIDATION_DATES.strftime('%Y-%m-%d')),
                test_dates=list(TEST_DATES.strftime('%Y-%m-%d')), seed=42,
                versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                              pandas=pd.__version__, joblib=joblib.__version__))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / 'locked_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Plan/code/input changed: preserve v1 and create a new version')
        if (OUTPUT / 'manifest.json').exists():
            manifest = json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
            for name, expected in manifest['output_hashes'].items():
                if sha256(OUTPUT / name) != expected:
                    raise ValueError(f'Completed recommendation output changed: {name}')
            print(f'Verified existing complete run: {OUTPUT}', flush=True)
            return
    else:
        write_json(plan_path, plan)
    write_json(OUTPUT / 'input_audit.json', input_audit)
    tuning, audit, timings = [], [], []
    for date in TUNE_DATES:
        training = make_training(tx, date)
        audit.append(training_record(training, 'tuning'))
        context = make_context(tx, snapshots, date, training['catalog'])
        labels = next_basket_labels(tx, snapshots, date)
        for method in FAMILIES:
            for spec in [s for s in specs if s['method'] == method]:
                tick = time.perf_counter()
                model = fit_model(spec, training)
                ranks = recommend(model, context)
                metric, _ = evaluate(ranks, context, labels, method, 'tuning', spec['id'])
                tuning.append(metric)
                timings.append(dict(phase='tuning', config_id=spec['id'], date=str(date.date()),
                                    seconds=time.perf_counter() - tick))
            print(f'Tuned {method} for {date.date()}', flush=True)
        del model, training, context
    tuning_monthly = pd.DataFrame(tuning)
    tuning_summary = summarize(tuning_monthly)
    shortlist = {method: choose_config(tuning_summary, method, specs) for method in FAMILIES}
    save_csv('tuning_monthly.csv', tuning_monthly)
    save_csv('tuning_comparison.csv', tuning_summary)
    write_json(OUTPUT / 'shortlist.json', shortlist)

    training = make_training(tx, VALIDATION_DATES.min())
    audit.append(training_record(training, 'validation'))
    save_csv('validation_catalog.csv', training['catalog'])
    validation_models = {}
    for method in FAMILIES:
        tick = time.perf_counter()
        validation_models[method] = fit_model(shortlist[method], training)
        timings.append(dict(phase='validation_fit', config_id=shortlist[method]['id'],
                            seconds=time.perf_counter() - tick))
    monthly, customers, labels, _ = phase_predictions(validation_models, tx, snapshots, VALIDATION_DATES, 'validation')
    validation_summary = summarize(monthly)
    selection = choose_method(validation_summary, monthly)
    selected = selection['selected_method']
    print(f'Validation selected {selected}; best learner {selection["best_learner"]}; best rule {selection["best_rule"]}', flush=True)
    draws, intervals = paired_intervals(customers.loc[customers.method.eq(selection['best_learner'])],
                                       customers.loc[customers.method.eq(selection['best_rule'])])
    selection['validation_customer_intervals'] = intervals
    save_csv('validation_labels.csv', labels)
    save_csv('validation_customer_metrics.csv', customers)
    save_csv('bootstrap_draws.csv', draws)
    seed_months = []
    for seed in (7, 2026):
        seed_model = fit_model(shortlist['implicit_als'], training, seed)
        seed_metric, _, _, _ = phase_predictions({'implicit_als': seed_model}, tx, snapshots, VALIDATION_DATES,
                                                'seed_diagnostic')
        seed_months.append(seed_metric)
    save_csv('als_seed_monthly.csv', pd.concat(seed_months, ignore_index=True))
    # Export only historical inputs. No target fields can be passed to the table builder.
    validation_export = []
    for date in VALIDATION_DATES:
        context = make_context(tx, snapshots, date, training['catalog'])
        ranks = recommend(validation_models[selected], context)
        validation_export.append(recommendation_table(validation_models[selected], context, ranks))
    validation_selected_model = validation_models[selected]
    del validation_models, training, seed_model

    final_training = make_training(tx, TEST_DATES.min())
    audit.append(training_record(final_training, 'test_frozen_global_model'))
    final_models = {}
    for method in FAMILIES:
        tick = time.perf_counter()
        final_models[method] = fit_model(shortlist[method], final_training)
        timings.append(dict(phase='test_fit_before_labels', config_id=shortlist[method]['id'],
                            seconds=time.perf_counter() - tick))
    save_csv('test_catalog.csv', final_training['catalog'])
    model_path = OUTPUT / 'selected_model.joblib'
    joblib.dump(dict(validation=validation_selected_model, test=final_models[selected],
                     selected_method=selected, list_length=10), model_path, compress=3)
    lock = dict(**selection, shortlisted_configs=shortlist, selected_config=shortlist[selected],
                plan_sha256=sha256(plan_path), model_sha256=sha256(model_path), training_audit=audit,
                source_hashes=input_audit['source_hashes'], global_test_model_updates=False,
                customer_history_updates='pre-cutoff purchases only, with global item parameters fixed')
    lock_path = OUTPUT / 'locked_selection.json'
    write_json(lock_path, lock)
    lock_hash = sha256(lock_path)
    print('Recommendation selection and model locked. Now opening next-basket test labels.', flush=True)

    test_monthly, test_customers, test_labels, test_export = phase_predictions(
        final_models, tx, snapshots, TEST_DATES, 'test', selected)
    if sha256(lock_path) != lock_hash or sha256(model_path) != lock['model_sha256']:
        raise ValueError('Test evaluation changed the locked selection/model')
    save_csv('test_labels.csv', test_labels)
    save_csv('test_customer_metrics.csv', test_customers)
    all_monthly = pd.concat([monthly, test_monthly], ignore_index=True)
    comparison = summarize(all_monthly)
    save_csv('monthly_comparison.csv', all_monthly)
    save_csv('comparison.csv', comparison)
    export = pd.concat(validation_export + [test_export], ignore_index=True)
    validate_export(export, snapshots)
    save_csv('recommendations.csv', export)
    save_csv('training_times.csv', pd.DataFrame(timings))
    write_json(OUTPUT / 'training_audit.json', {'folds': audit})
    write_report(comparison, all_monthly, lock, input_audit, labels, test_labels)
    make_figure(all_monthly, selected)
    files = [p for p in OUTPUT.iterdir() if p.is_file() and p.name != 'manifest.json']
    write_json(OUTPUT / 'manifest.json', dict(status='completed_next_basket_comparison',
               selected_method=selected, source_hashes=input_audit['source_hashes'],
               code_hashes=plan['code_hashes'], output_hashes={p.name: sha256(p) for p in sorted(files)},
               elapsed_seconds=time.perf_counter() - started, test_evaluated_after_selection_lock=True))
    print(f'Completed next-basket recommendation: {OUTPUT}', flush=True)


def validate_export(frame: pd.DataFrame, snapshots: pd.DataFrame) -> None:
    expected = {'snapshot_date', 'customer_id', 'rank', 'stock_code', 'description',
                'purchased_in_past_180d', 'recommendation_method', 'reason'}
    if (set(frame.columns) != expected or frame.isna().any().any()
            or frame.duplicated(['snapshot_date', 'customer_id', 'rank']).any()
            or frame.duplicated(['snapshot_date', 'customer_id', 'stock_code']).any()):
        raise ValueError('Unsafe or duplicate recommendation export')
    counts = frame.groupby(['snapshot_date', 'customer_id']).size()
    dates = list(VALIDATION_DATES) + list(TEST_DATES)
    expected_cohort = snapshots.loc[snapshots.snapshot_date.isin(dates), ['snapshot_date', 'customer_id']].copy()
    expected_cohort.snapshot_date = expected_cohort.snapshot_date.dt.strftime('%Y-%m-%d')
    actual_cohort = frame[['snapshot_date', 'customer_id']].drop_duplicates()
    if (not counts.eq(10).all() or not frame['rank'].between(1, 10).all()
            or set(map(tuple, expected_cohort.to_numpy())) != set(map(tuple, actual_cohort.to_numpy()))):
        raise ValueError('Recommendation export is missing eligible customers or ten-item lists')


def make_figure(monthly: pd.DataFrame, selected: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, phase, title in zip(axes, ('validation', 'test'), ('Validation: Jun-Aug', 'Locked test: Sep-Nov')):
        rows = monthly.loc[monthly.phase.eq(phase)]
        dates = sorted(rows.snapshot_date.unique())
        for method in FAMILIES:
            values = rows.loc[rows.method.eq(method)].set_index('snapshot_date').loc[dates]
            ax.plot(range(3), values.ndcg_10, marker='o', label=method,
                    linewidth=2.5 if method == selected else 1.2)
        ax.set_xticks(range(3), [d[:7] for d in dates])
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Buyer NDCG@10')
    axes[1].legend(fontsize=8)
    fig.suptitle(f'Next-basket recommendation | selected: {selected}')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'comparison.png', dpi=160)
    plt.close(fig)


def write_report(comparison: pd.DataFrame, monthly: pd.DataFrame, lock: dict, audit: dict,
                 validation_labels: pd.DataFrame, test_labels: pd.DataFrame) -> None:
    selected = lock['selected_method']
    lines = ['# Product recommendations for next order: comparison results', '', f'**Final adoption: {NAMES[selected]}.**', '',
             'Recommend 10 products each month to customers who have purchased something in the past 180 days. The answer is an order of different items in the next 30 days,'
             'Allows recommendations for previously purchased items. The main indicator NDCG@10 simultaneously measures whether the product is hit and the ranking of the hit product.', '',
             '## Fair comparison of five methods', '',
             '| Method | Verify NDCG@10 | Verify Recall@10 | Verify Hit@10 | Test NDCG@10 | Test Recall@10 | Test Hit@10 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in FAMILIES:
        a = comparison.loc[comparison.phase.eq('validation') & comparison.method.eq(method)].iloc[0]
        b = comparison.loc[comparison.phase.eq('test') & comparison.method.eq(method)].iloc[0]
        lines.append(f'| {NAMES[method]} | {a.mean_ndcg_10:.4f} | {a.mean_recall_10:.2%} | '
                     f'{a.mean_hit_10:.2%} | {b.mean_ndcg_10:.4f} | {b.mean_recall_10:.2%} | {b.mean_hit_10:.2%} |')
    lines += ['', 'Equally weighted average for each month. The above three indicators only evaluate customers with next orders; a list is generated for all qualified customers.'
              'The complete table also reports the hit rate of all customers, @ 5, product coverage, recalls of purchased/unpurchased products, and the proportion of unknown new products.', '',
              '## Why choose it?', '',
              f'- The strongest rule during the verification period is {NAMES[lock["best_rule"]]}, and the strongest learning plan is {NAMES[lock["best_learner"]]}.',
              f'- The average NDCG difference between the strongest learning scheme and the strongest rule is: {lock["mean_validation_ndcg_gain"]:+.4f}.',
              '- Prior conditions: the average NDCG is at least 0.002 higher, at least 2 / 3 does not regress for months, and the average Recall lags no more than 0.005.',
              f'- Condition result: `{json.dumps(lock["conditions"], ensure_ascii=False)}`.',
              f'- Final parameters: `{json.dumps(lock["selected_config"], ensure_ascii=False)}`.', '',
              '## Data and time sequence', '',
              f'Clean transaction line {audit["source_lines"]:,}; after merging multiple lines of the same invoice and the same product,'
              f'Received {audit["distinct_invoice_item_events"]:,} purchase interactions. Quantity and amount do not participate in interactive weighting, reducing the impact of large wholesale orders.', '',
              '| Purpose | Customer Months | Customer Months with Next Order |', '|---|---:|---:|']
    for phase, frame in [('Verification: 6 – 8 months', validation_labels), ('Test: 9 – 11 months', test_labels)]:
        lines.append(f'| {phase} | {len(frame):,} | {int(frame.target_count.gt(0).sum()):,} |')
    lines += ['', '1 – 5 is developed monthly, and each type of learning program has 8 set of parameters. 6 – 8 monthly 6 monthly 1 global product relationship selection;'
              'After selection, use the 9 month- 1 day-ahead transactions to fit and freeze the final product relationship, and then calculate the 9 – 11 month product recommendation results at once.'
              'The customer\'s own input only uses orders that have occurred at that time on a monthly basis, and all methods follow the same rules.'
              '9 – 11 have been used for repurchase and amount tasks before this month, and the product answers in this round are not involved in the selection.', '',
              'Candidate products are determined by all sold products within the 180 days of the training period in each stage, and the methods are consistent;'
              'The same scores are sorted according to popularity and coding during the training period; new products that appeared in the test are retained in the denominator of the answer.'
              'There is no inventory data, and product listings do not represent availability at that time.', '',
              '## Month by month effects and difficulties', '',
              '| Month | Purpose | Customers with next order | NDCG@10 | Recall@10 | Hit@10 | Proportion of answers among candidates | Bought items Recall | Unbought items Recall |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in monthly.loc[monthly.method.eq(selected)].itertuples():
        lines.append(f'| {row.snapshot_date[:7]} | {"Verify" if row.phase == "validation" else "test"} |'
                     f'{row.buyers:,} | {row.ndcg_10:.4f} | {row.recall_10:.2%} | {row.hit_10:.2%} | '
                     f'{row.known_target_share:.2%} | {row.repeat_recall_10:.2%} | {row.new_recall_10:.2%} |')
    lines += ['', 'Bought/Not Bought refers to whether the item was purchased in the past 180 days. The next order may contain many items, so the recommended Recall for 10 pieces may be lower;'
              'Hit only requires at least one hit, and it cannot be used alone to claim that the recommendation is very accurate.', '',
              'Verification period 500 customer group paired sampling (the strongest learning plan minus the strongest rule):']
    for metric, values in lock['validation_customer_intervals']['metrics'].items():
        lines.append(f'- {metric}: {values["point"]:+.4f}, 95% customer sampling interval'
                     f'[{values["lower_95"]:+.4f}, {values["upper_95"]:+.4f}]。')
    lines += ['Sampling is an internal robustness check after selection and is not a guarantee of new month results. ALS Additional seed results are available in als_seed_monthly.csv.', '',
              '![Monthly recommended comparison ](comparison.png)', '', '## Open and reproduce', '',
              '`python -B -m pipelines.recommendation_baselines`', '',
              '`python -m streamlit run app/dashboard.py` → **Product recommendations** → Select date and customer.', '',
              'recommendations.csv is used for pages and exports and does not contain future orders, hits or answers.'
              'validation_labels.csv/test_labels.csv and customer-by-customer indicators are only verified offline;'
              'locked_selection.json saves the selection, parameters and model hash, and tuning_comparison.csv saves all 30 group configurations.', '',
              'Algorithm basis: [ ItemKNN ](https://www.ra.ethz.ch/CDstore/www10/papers/519/) ·'
              '[Implicit feedback ALS ](https://yifanhu.net/PUB/cf.pdf) · [ EASE ](https://arxiv.org/abs/1905.03375).', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
