"""Run the locked next-basket comparison: python -B -m recommendation.experiment."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy

from recommendation.data import (ROOT, make_context, make_training, next_basket_labels,
                                  labels_for_export, read_inputs, sha256)
from recommendation.metrics import evaluate, paired_intervals, summarize
from recommendation.models import (FAMILIES, LEARNERS, NAMES, RULES, configurations,
                                    fit_model, recommend, recommendation_table)

OUTPUT = ROOT / 'recommendation/outputs/v1'
PROTOCOL = ROOT / 'recommendation/protocol.md'
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
    code_files = ['data.py', 'models.py', 'metrics.py', 'experiment.py', 'protocol.md']
    plan = dict(version='recommendation_v1', source_hashes=input_audit['source_hashes'], specs=specs,
                code_hashes={name: sha256(ROOT / 'recommendation' / name) for name in code_files},
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
    lines = ['# 下一笔订单商品推荐：比较结果', '', f'**最终采用：{NAMES[selected]}。**', '',
             '每月为过去180天买过东西的客户推荐10种商品。答案是随后30天内下一笔订单的不同商品，'
             '允许推荐以前买过的商品。主指标NDCG@10同时衡量是否命中及命中商品的排名。', '',
             '## 五种方法的公平比较', '',
             '| 方法 | 验证NDCG@10 | 验证Recall@10 | 验证Hit@10 | 测试NDCG@10 | 测试Recall@10 | 测试Hit@10 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in FAMILIES:
        a = comparison.loc[comparison.phase.eq('validation') & comparison.method.eq(method)].iloc[0]
        b = comparison.loc[comparison.phase.eq('test') & comparison.method.eq(method)].iloc[0]
        lines.append(f'| {NAMES[method]} | {a.mean_ndcg_10:.4f} | {a.mean_recall_10:.2%} | '
                     f'{a.mean_hit_10:.2%} | {b.mean_ndcg_10:.4f} | {b.mean_recall_10:.2%} | {b.mean_hit_10:.2%} |')
    lines += ['', '各月份等权平均。上述三个指标仅评价有下一笔订单的客户；所有合格客户都会生成名单。'
              '完整表另报全体客户命中率、@5、商品覆盖、已买/未买商品的召回及未知新品占比。', '',
              '## 为什么选它', '',
              f'- 验证期最强规则为{NAMES[lock["best_rule"]]}，最强学习方案为{NAMES[lock["best_learner"]]}。',
              f'- 最强学习方案相对最强规则的验证平均NDCG差值：{lock["mean_validation_ndcg_gain"]:+.4f}。',
              '- 事先条件：平均NDCG至少高0.002、至少2/3个月不退步、平均Recall落后不超过0.005。',
              f'- 条件结果：`{json.dumps(lock["conditions"], ensure_ascii=False)}`。',
              f'- 最终参数：`{json.dumps(lock["selected_config"], ensure_ascii=False)}`。', '',
              '## 数据与时间顺序', '',
              f'清洗交易{audit["source_lines"]:,}行；合并同一发票同一商品的多行后，'
              f'得到{audit["distinct_invoice_item_events"]:,}条购买互动。数量和金额不参与互动加权，减少批发大单影响。', '',
              '| 用途 | 客户月 | 有下一笔订单的客户月 |', '|---|---:|---:|']
    for phase, frame in [('验证：6–8月', validation_labels), ('测试：9–11月', test_labels)]:
        lines.append(f'| {phase} | {len(frame):,} | {int(frame.target_count.gt(0).sum()):,} |')
    lines += ['', '1–5月向前开发，每类学习方案8组参数。6–8月用6月1日前的全局商品关系选型；'
              '选定后，用9月1日前交易拟合并冻结最终商品关系，再一次性计算9–11月商品推荐结果。'
              '客户自身输入按月只使用当时已发生的订单，全部方法遵守相同规则。'
              '9–11月此前已用于复购和金额任务，本轮商品答案未参与选型。', '',
              '候选商品由各阶段训练期180天内全部已售商品确定，各方法一致；'
              '同分统一按训练期热门程度及编码排序；测试中出现的新品保留在答案分母中。'
              '没有库存数据，商品清单不代表当时有货。', '',
              '## 逐月效果与难点', '',
              '| 月份 | 用途 | 有下一单客户 | NDCG@10 | Recall@10 | Hit@10 | 候选内答案占比 | 已买商品Recall | 未买商品Recall |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in monthly.loc[monthly.method.eq(selected)].itertuples():
        lines.append(f'| {row.snapshot_date[:7]} | {"验证" if row.phase == "validation" else "测试"} | '
                     f'{row.buyers:,} | {row.ndcg_10:.4f} | {row.recall_10:.2%} | {row.hit_10:.2%} | '
                     f'{row.known_target_share:.2%} | {row.repeat_recall_10:.2%} | {row.new_recall_10:.2%} |')
    lines += ['', '已买/未买指过去180天是否买过。下一单可能包含很多商品，因此推荐10件的Recall可能较低；'
              'Hit只要求至少命中一件，不能单独用它宣称推荐非常准确。', '',
              '验证期500次客户成组配对抽样（最强学习方案减最强规则）：']
    for metric, values in lock['validation_customer_intervals']['metrics'].items():
        lines.append(f'- {metric}：{values["point"]:+.4f}，95%客户抽样区间'
                     f'[{values["lower_95"]:+.4f}, {values["upper_95"]:+.4f}]。')
    lines += ['抽样是选型后的内部稳健性检查，不是新月份效果保证。ALS额外种子结果见als_seed_monthly.csv。', '',
              '![逐月推荐比较](comparison.png)', '', '## 打开与复现', '',
              '`python -B -m recommendation.experiment`', '',
              '`python -m streamlit run app/dashboard.py` → **Product recommendations** → 选择日期和客户。', '',
              'recommendations.csv用于页面和导出，不含未来订单、命中结果或答案。'
              'validation_labels.csv/test_labels.csv和逐客户指标仅作离线核对；'
              'locked_selection.json保存选型、参数和模型哈希，tuning_comparison.csv保存全部30组配置。', '',
              '算法依据：[ItemKNN](https://www.ra.ethz.ch/CDstore/www10/papers/519/) · '
              '[隐式反馈ALS](https://yifanhu.net/PUB/cf.pdf) · [EASE](https://arxiv.org/abs/1905.03375)。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
