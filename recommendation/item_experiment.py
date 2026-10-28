"""Locked four-feature / three-model item recommendation experiment."""

from __future__ import annotations

import json
import platform
import time

import joblib
import numpy as np
import pandas as pd

from recommendation.data import (ROOT, read_inputs, make_training, make_context,
                                  next_basket_labels, sha256, source_hashes)
from recommendation.experiment import TUNE_DATES, VALIDATION_DATES, TEST_DATES, write_json
from recommendation.models import fit_model, configurations, scores, topk
from recommendation.metrics import evaluate, summarize, paired_intervals
from recommendation.hybrid import metric_context, fast_metrics, prepare_routes, fuse_routes, mixing_weights
from recommendation.profiles import predict_propensity
from recommendation.profile_experiment import check_completed as check_v2
from recommendation.item_features import (ALL_FEATURES, FEATURE_SETS, item_context,
                                          sample_training_pairs, mature_training, sampling_weights)
from recommendation.item_models import (FAMILIES, NAMES, model_specs, fit_item_model, score_catalog)

OUTPUT = ROOT / 'recommendation/outputs/item_v3_1'
PROTOCOL = ROOT / 'recommendation/item_protocol.md'
CODE_FILES = ['item_protocol.md', 'item_features.py', 'item_models.py', 'item_experiment.py',
              'data.py', 'models.py', 'metrics.py', 'profiles.py', 'hybrid.py']
BASELINES = ('personal_repeat', 'pure_ease', 'old_adaptive_ease')
NEW_METHODS = [f'{group}_{family}' for group in FEATURE_SETS for family in FAMILIES]


def save(name, frame):
    frame.to_csv(OUTPUT / name, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d %H:%M:%S')


def check_completed():
    manifest = json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed_item_comparison' or manifest['source_hashes'] != source_hashes():
        raise ValueError('Item recommendation source/status changed')
    for name, digest in manifest['code_hashes'].items():
        path = (ROOT / 'recommendation' / name).resolve()
        if path.parent != (ROOT / 'recommendation').resolve() or sha256(path) != digest:
            raise ValueError(f'Item implementation changed: {name}')
    for name, digest in manifest['output_hashes'].items():
        path = (OUTPUT / name).resolve()
        if path.parent != OUTPUT.resolve() or sha256(path) != digest:
            raise ValueError(f'Item comparison output changed: {name}')
    if sha256(ROOT / 'recommendation/outputs/profile_v2/manifest.json') != manifest['v2_manifest_sha256']:
        raise ValueError('Prior recommendation comparison changed')
    return manifest


def choose_ease(history: list[dict], cutoff: pd.Timestamp) -> dict:
    options = [s for s in configurations() if s['method'] == 'ease']
    usable = [row for row in history if pd.Timestamp(row['label_end']) <= cutoff]
    if not usable:
        return max(options, key=lambda s: s['regularization'])
    means = pd.DataFrame(usable).groupby('config_id').ndcg_10.mean().to_dict()
    if set(means) != {spec['id'] for spec in options}:
        raise ValueError('Historical EASE budget incomplete')
    return min(options, key=lambda s: (-means[s['id']], -s['regularization']))


def common_specs(monthly: pd.DataFrame) -> dict:
    part = monthly.loc[monthly.phase.eq('development')]
    chosen = {}
    for family in FAMILIES:
        means = {}
        for spec in model_specs(family):
            rows = part.loc[part.config_id.eq(spec['id'])]
            if (len(rows) != 20 or set(rows.group) != set(FEATURE_SETS)
                    or set(rows.snapshot_date) != set(TUNE_DATES.strftime('%Y-%m-%d'))
                    or rows.duplicated(['group', 'snapshot_date']).any()):
                raise ValueError('Incomplete shared item model budget')
            means[spec['id']] = rows.ndcg_10.mean()
        options = model_specs(family)
        chosen[family] = min(options, key=lambda s: (-means[s['id']], options.index(s)))
    return chosen


def effect(monthly: pd.DataFrame, challenger: str, reference: str) -> dict:
    part = monthly.loc[monthly.phase.eq('validation')]
    a = part.loc[part.method.eq(challenger)].set_index('snapshot_date')
    b = part.loc[part.method.eq(reference)].set_index('snapshot_date')
    expected = set(VALIDATION_DATES.strftime('%Y-%m-%d'))
    if len(a) != 3 or len(b) != 3 or set(a.index) != expected or set(b.index) != expected:
        raise ValueError('Effect comparison must use the same three validation months')
    return dict(challenger=challenger, reference=reference,
                ndcg_gain=float((a.ndcg_10 - b.ndcg_10).mean()),
                recall_gain=float((a.recall_10 - b.recall_10).mean()),
                new_recall_gain=float((a.new_recall_10 - b.new_recall_10).mean()),
                nonworse_months=int(((a.ndcg_10 - b.ndcg_10) >= -1e-12).sum()))


def choose_default(monthly: pd.DataFrame) -> dict:
    data = monthly.loc[monthly.phase.eq('validation')]
    if set(data.method) != set(NAMES):
        raise ValueError('Incomplete item and old method comparison')
    means = data.groupby('method').ndcg_10.mean().to_dict()
    improvements = {name: effect(data, name, 'personal_repeat') for name in NEW_METHODS}
    eligible = [name for name, gain in improvements.items()
                if gain['ndcg_gain'] >= .01 - 1e-12 and gain['recall_gain'] >= .01 - 1e-12
                and gain['new_recall_gain'] >= .01 - 1e-12 and gain['nonworse_months'] >= 2]
    best = max(NEW_METHODS, key=lambda name: (means[name], -NEW_METHODS.index(name)))
    selected = max(eligible, key=lambda name: (means[name], -NEW_METHODS.index(name))) if eligible else 'personal_repeat'
    ablations = []
    for family in FAMILIES:
        for a, b, kind in [('B', 'A', 'profile_without_ease'), ('D', 'C', 'profile_with_ease'),
                            ('C', 'A', 'ease_without_profile'), ('D', 'B', 'ease_with_profile')]:
            result = effect(data, f'{a}_{family}', f'{b}_{family}')
            result['kind'] = kind
            result['family'] = family
            result['demonstrated'] = bool(result['ndcg_gain'] >= .002 - 1e-12
                                           and result['recall_gain'] >= -.005 - 1e-12
                                           and result['nonworse_months'] >= 2)
            ablations.append(result)
    selected_group = selected.split('_')[0]
    profile_reference = ('A_' + selected.split('_')[1] if selected_group == 'B' else
                         'C_' + selected.split('_')[1] if selected_group == 'D' else None)
    profile_effect = next((row for row in ablations if row['challenger'] == selected
                           and row['reference'] == profile_reference), None)
    return dict(selected_method=selected, selected_name=NAMES[selected], best_new=best,
                deployment_gate_passed=bool(eligible), eligible=eligible,
                improvement_vs_default=improvements[best],
                selected_profile_reference=profile_reference,
                selected_profile_value_demonstrated=bool(profile_effect and profile_effect['demonstrated']),
                ablations=ablations, selection_phase='validation',
                test_labels_used_for_selection=False, prior_validation_and_test_results_viewed=True,
                global_test_model_updates=False)


def old_predictions(old_bundle, phase, context):
    models = old_bundle[phase]['recommenders']
    personal = context['counts_dense']
    raw = scores(models['ease'], context)
    prop = old_bundle[phase]['propensity_models'][old_bundle['selected_propensity']['family']]
    predictions = predict_propensity(prop, context['profiles'])
    weight = mixing_weights(old_bundle['hybrid_specs']['adaptive_ease'], predictions, prop['prior'])
    ranks = dict(personal_repeat=topk(personal), pure_ease=topk(raw),
                 old_adaptive_ease=fuse_routes(prepare_routes(personal, raw), weight))
    values = {'personal_repeat': personal, 'pure_ease': raw}
    top_values = {name: np.take_along_axis(value, ranks[name], axis=1) for name, value in values.items()}
    # Channel scores for the old adaptive comparator, not probabilities.
    adaptive = np.zeros_like(personal)
    for row in range(len(personal)):
        old = np.flatnonzero(personal[row] > 0)
        old = old[np.argsort(-personal[row, old], kind='stable')]
        new = np.flatnonzero(personal[row] == 0)
        new = new[np.argsort(-raw[row, new], kind='stable')]
        adaptive[row, old] = weight[row] / np.arange(1, len(old) + 1)
        adaptive[row, new] = (1 - weight[row]) / np.arange(1, len(new) + 1)
    top_values['old_adaptive_ease'] = np.take_along_axis(adaptive, ranks['old_adaptive_ease'], axis=1)
    return ranks, top_values


def phase_run(models, ease, old_bundle, phase, dates, tx, snapshots):
    monthly, customer_rows, lists, profiles, groups = [], [], [], [], []
    for date in dates:
        context = item_context(tx, snapshots, date, ease)
        labels = next_basket_labels(tx, snapshots, date)
        rankings, values = score_catalog(models, context)
        old_ranks, old_values = old_predictions(old_bundle, phase, context)
        rankings.update(old_ranks)
        values.update(old_values)
        profile = context['profiles'].copy()
        profiles.append(profile)
        catalog = context['catalog'].stock_code.to_numpy()
        for method, ranks in rankings.items():
            config = models[method]['spec']['id'] if method in models else 'old_frozen'
            metric, customers = evaluate(ranks, context, labels, method, phase, config)
            monthly.append(metric)
            customer_rows.append(customers)
            idx = np.arange(len(ranks))[:, None]
            counts = context['counts_dense'][idx, ranks]
            last = context['item_last'][idx, ranks]
            lists.append(pd.DataFrame(dict(snapshot_date=str(date.date()), customer_id=context['customers'], method=method,
                       recommended_items=[json.dumps(list(x)) for x in catalog[ranks]],
                       purchase_counts_180d=[json.dumps(x.tolist()) for x in counts],
                       days_since_item_purchase=[json.dumps(x.tolist()) for x in last],
                       ranking_scores=[json.dumps(x.tolist()) for x in values[method]])))
            merged = customers.merge(profile[['customer_id', 'habit_group']], on='customer_id', validate='one_to_one')
            for group, part in merged.loc[merged.target_count.gt(0)].groupby('habit_group'):
                groups.append(dict(phase=phase, snapshot_date=str(date.date()), method=method, habit_group=group,
                                   buyers=len(part), ndcg_10=part.ndcg_10.mean(), recall_10=part.recall_10.mean(),
                                   new_recall_10=part.new_recall_10.mean()))
        print(f'{phase}: {date.date()} | all 15 full-catalog lists completed', flush=True)
    return dict(monthly=pd.DataFrame(monthly), customers=pd.concat(customer_rows, ignore_index=True),
                lists=pd.concat(lists, ignore_index=True), profiles=pd.concat(profiles, ignore_index=True), groups=pd.DataFrame(groups))


def run():
    started = time.perf_counter()
    if (OUTPUT / 'manifest.json').exists():
        check_completed()
        print(f'Verified completed item comparison: {OUTPUT}', flush=True)
        return
    check_v2()
    tx, snapshots, audit = read_inputs()
    previous_samples = ROOT / 'recommendation/outputs/item_v3/training_samples.joblib'
    plan = dict(version='item_v3_1', source_hashes=audit['source_hashes'],
                code_hashes={name: sha256(ROOT / 'recommendation' / name) for name in CODE_FILES},
                features=FEATURE_SETS, model_specs={family: model_specs(family) for family in FAMILIES},
                v2_manifest_sha256=sha256(ROOT / 'recommendation/outputs/profile_v2/manifest.json'),
                versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                reused_sample_sha256=sha256(previous_samples) if previous_samples.exists() else None,
                full_catalog_evaluation=True, prior_validation_and_test_results_viewed=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / 'locked_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Item plan/code changed; preserve outputs and create a new version')
    if (OUTPUT / 'locked_selection.json').exists():
        raise ValueError('Item selection locked; do not automatically re-open review')
    write_json(plan_path, plan)
    write_json(OUTPUT / 'input_audit.json', audit)
    old_bundle = joblib.load(ROOT / 'recommendation/outputs/profile_v2/model_bundle.joblib')
    # Fully reproducible training-data preparation can resume before any model selection.
    preparation_path = OUTPUT / 'training_samples.joblib'
    if preparation_path.exists():
        prepared = joblib.load(preparation_path)
        if prepared['plan_sha256'] != sha256(plan_path):
            raise ValueError('Prepared item samples do not match the plan')
        parts, ease_history, ease_audit = prepared['parts'], prepared['ease_history'], prepared['ease_audit']
    elif previous_samples.exists():
        previous_plan = json.loads((previous_samples.parent / 'locked_plan.json').read_text(encoding='utf-8'))
        if previous_plan['source_hashes'] != audit['source_hashes'] or previous_plan['features'] != FEATURE_SETS:
            raise ValueError('Original sampled feature matrix does not match current data/features')
        prepared = joblib.load(previous_samples)
        parts, ease_history, ease_audit = prepared['parts'], prepared['ease_history'], prepared['ease_audit']
        for part in parts:
            part['sampling_weights'] = sampling_weights(part['queries'], len(part['catalog_codes']))
        joblib.dump(dict(parts=parts, ease_history=ease_history, ease_audit=ease_audit,
                         plan_sha256=sha256(plan_path)), preparation_path, compress=3)
        print('Reused audited past-only feature rows; corrected inclusion weights before validation.', flush=True)
    else:
        parts, ease_history, ease_audit = [], [], []
        for date in pd.date_range(snapshots.snapshot_date.min(), TUNE_DATES.max(), freq='MS'):
            spec = choose_ease(ease_history, date)
            training = make_training(tx, date)
            buyers = snapshots.loc[snapshots.repurchase_30d.eq(1)]
            context = make_context(tx, buyers, date, training['catalog'])
            labels = next_basket_labels(tx, buyers, date)
            truth = metric_context(context, labels)
            selected_ease = None
            for candidate in [s for s in configurations() if s['method'] == 'ease']:
                model = fit_model(candidate, training)
                ranks = topk(scores(model, context))
                ease_history.append(dict(snapshot_date=str(date.date()), label_end=str((date + pd.Timedelta(days=30)).date()),
                                         config_id=candidate['id'], **fast_metrics(ranks, truth)))
                if candidate['id'] == spec['id']:
                    selected_ease = model
            feature_context = item_context(tx, buyers, date, selected_ease)
            part = sample_training_pairs(feature_context, labels)
            parts.append(part)
            eligible_dates = sorted({r['snapshot_date'] for r in ease_history
                                     if pd.Timestamp(r['label_end']) <= date})
            ease_audit.append(dict(feature_date=str(date.date()), fit_cutoff=str(date.date()),
                                   latest_transaction=training['latest_transaction'], selected_spec=spec,
                                   selection_label_dates=eligible_dates))
            print(f'Past-only item sample: {date.date()} | {len(part["y"]):,} pairs | EASE lambda={spec["regularization"]}', flush=True)
        joblib.dump(dict(parts=parts, ease_history=ease_history, ease_audit=ease_audit,
                         plan_sha256=sha256(plan_path)), preparation_path, compress=3)
    save('historical_ease_comparison.csv', pd.DataFrame(ease_history))
    ease_spec = choose_ease(ease_history, VALIDATION_DATES.min())
    validation_training = make_training(tx, VALIDATION_DATES.min())
    validation_ease = fit_model(ease_spec, validation_training)
    # Cache each development month only in process memory; no temporary repository files.
    tuning = []
    training_audit = []
    for date in TUNE_DATES:
        fit = mature_training(parts, date)
        training_audit.append(dict(phase='development', fit_cutoff=str(date.date()), rows=len(fit['y']),
                                   queries=len(fit['groups']), latest_label_end=fit['latest_label_end']))
        historical_spec = next(row['selected_spec'] for row in ease_audit if row['feature_date'] == str(date.date()))
        ease = fit_model(historical_spec, make_training(tx, date))
        buyers = snapshots.loc[snapshots.repurchase_30d.eq(1)]
        context = item_context(tx, buyers, date, ease)
        labels = next_basket_labels(tx, buyers, date)
        truth = metric_context(context, labels)
        for family in FAMILIES:
            models = {f'{group}_{spec["id"]}': fit_item_model(spec, group, fit)
                      for group in FEATURE_SETS for spec in model_specs(family)}
            rankings, _ = score_catalog(models, context)
            for name, ranks in rankings.items():
                model = models[name]
                tuning.append(dict(phase='development', snapshot_date=str(date.date()), group=model['group'],
                                   family=family, config_id=model['spec']['id'], **fast_metrics(ranks, truth)))
            print(f'Item forward fold: {date.date()} | {family}, four features x four configs', flush=True)
            save('development_monthly.csv', pd.DataFrame(tuning))
        del models, context, fit
    shortlist = common_specs(pd.DataFrame(tuning))
    write_json(OUTPUT / 'shared_shortlist.json', shortlist)
    fit = mature_training(parts, VALIDATION_DATES.min())
    validation_models = {f'{group}_{family}': fit_item_model(spec, group, fit)
                          for family, spec in shortlist.items() for group in FEATURE_SETS}
    training_audit.append(dict(phase='validation', fit_cutoff=fit['fit_cutoff'], rows=len(fit['y']),
                               queries=len(fit['groups']), latest_label_end=fit['latest_label_end']))
    validation = phase_run(validation_models, validation_ease, old_bundle, 'validation', VALIDATION_DATES, tx, snapshots)
    for name, frame in validation.items():
        save(f'validation_{name}.csv', frame)
    decision = choose_default(validation['monthly'])
    save('validation_ablations.csv', pd.DataFrame(decision['ablations']))
    best = decision['best_new']
    draws, intervals = paired_intervals(validation['customers'].loc[validation['customers'].method.eq(best)],
                                        validation['customers'].loc[validation['customers'].method.eq('personal_repeat')])
    save('validation_best_paired_draws.csv', draws)
    decision['best_vs_default_intervals'] = intervals
    reference = decision['selected_profile_reference']
    if reference:
        draw, interval = paired_intervals(validation['customers'].loc[validation['customers'].method.eq(decision['selected_method'])],
                                          validation['customers'].loc[validation['customers'].method.eq(reference)])
        save('validation_profile_paired_draws.csv', draw)
        decision['selected_profile_intervals'] = interval
    # Refit with matured validation examples, preserving the June global signal for those input months.
    for date in VALIDATION_DATES:
        buyers = snapshots.loc[snapshots.repurchase_30d.eq(1)]
        context = item_context(tx, buyers, date, validation_ease)
        parts.append(sample_training_pairs(context, next_basket_labels(tx, buyers, date)))
    fit = mature_training(parts, TEST_DATES.min())
    final_models = {f'{group}_{family}': fit_item_model(spec, group, fit)
                    for family, spec in shortlist.items() for group in FEATURE_SETS}
    training_audit.append(dict(phase='review_frozen', fit_cutoff=fit['fit_cutoff'], rows=len(fit['y']),
                               queries=len(fit['groups']), latest_label_end=fit['latest_label_end']))
    final_training = make_training(tx, TEST_DATES.min())
    final_ease = fit_model(ease_spec, final_training)
    artifact = dict(validation=dict(models=validation_models, ease=validation_ease),
                    test=dict(models=final_models, ease=final_ease), ease_spec=ease_spec,
                    selected_method=decision['selected_method'], shortlist=shortlist,
                    prior_bundle_sha256=sha256(ROOT / 'recommendation/outputs/profile_v2/model_bundle.joblib'))
    joblib.dump(artifact, OUTPUT / 'model_bundle.joblib', compress=3)
    write_json(OUTPUT / 'training_audit.json', dict(item_models=training_audit, ease_features=ease_audit,
               validation_ease_spec=ease_spec, final_ease_spec=ease_spec,
               sampling=[dict(date=str(p['date'].date()), queries=len(p['groups']), pairs=len(p['y']),
                              positives=int(p['y'].sum()), unknown_only_buyers=p['unknown_only_buyers']) for p in parts]))
    write_json(OUTPUT / 'feature_manifest.json', dict(features=FEATURE_SETS, all_columns=ALL_FEATURES,
               input_feature_count={key: len(value) for key, value in FEATURE_SETS.items()},
               no_id_features=True, full_catalog_evaluation=True))
    lock = dict(**decision, source_hashes=audit['source_hashes'], shared_shortlist=shortlist,
                ease_spec=ease_spec, model_sha256=sha256(OUTPUT / 'model_bundle.joblib'),
                plan_sha256=sha256(plan_path))
    lock_path = OUTPUT / 'locked_selection.json'
    write_json(lock_path, lock)
    lock_hash = sha256(lock_path)
    print(f'Selection locked: {lock["selected_name"]}; deployment gate={lock["deployment_gate_passed"]}. Starting fixed review.', flush=True)
    test = phase_run(final_models, final_ease, old_bundle, 'test', TEST_DATES, tx, snapshots)
    if sha256(lock_path) != lock_hash or sha256(OUTPUT / 'model_bundle.joblib') != lock['model_sha256']:
        raise ValueError('Review mutated selection or model')
    for name, frame in test.items():
        save(f'test_{name}.csv', frame)
    monthly = pd.concat([validation['monthly'], test['monthly']], ignore_index=True)
    comparison = summarize(monthly)
    save('monthly_comparison.csv', monthly)
    save('comparison.csv', comparison)
    save('profiles.csv', pd.concat([validation['profiles'], test['profiles']], ignore_index=True))
    save('validation_catalog.csv', validation_ease['catalog'])
    save('test_catalog.csv', final_ease['catalog'])
    write_report(comparison, lock, pd.DataFrame(decision['ablations']), time.perf_counter() - started)
    files = sorted(p for p in OUTPUT.iterdir() if p.is_file() and p.name != 'manifest.json')
    write_json(OUTPUT / 'manifest.json', dict(status='completed_item_comparison', selected_method=lock['selected_method'],
               source_hashes=audit['source_hashes'], code_hashes=plan['code_hashes'],
               v2_manifest_sha256=plan['v2_manifest_sha256'], output_hashes={p.name: sha256(p) for p in files},
               elapsed_seconds=time.perf_counter() - started, review_after_selection_lock=True))
    print(f'Completed item-level comparison: {OUTPUT}', flush=True)


def write_report(comparison, lock, ablations, elapsed):
    lines = ['# 商品级推荐：四组特征与三类模型', '', f'默认选择：**{lock["selected_name"]}**。', '',
             f'整体替换门槛满足：{lock["deployment_gate_passed"]}；所选方案的画像额外价值：'
             f'{lock["selected_profile_value_demonstrated"]}。', '',
             '所有方案每人10件，对完整相同商品目录打分；主指标为有下一单客户的月均NDCG@10，'
             '月份等权。目录外真实商品保留分母。没有下一单仍生成名单，不当作不喜欢所有商品。', '',
             '| 方案 | 验证NDCG | 验证Recall | 验证未买Recall | 复核NDCG | 复核Recall | 复核未买Recall |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in NAMES:
        a = comparison.loc[comparison.phase.eq('validation') & comparison.method.eq(method)].iloc[0]
        b = comparison.loc[comparison.phase.eq('test') & comparison.method.eq(method)].iloc[0]
        lines.append(f'| {NAMES[method]} | {a.mean_ndcg_10:.4f} | {a.mean_recall_10:.2%} | {a.mean_new_recall_10:.2%} | '
                     f'{b.mean_ndcg_10:.4f} | {b.mean_recall_10:.2%} | {b.mean_new_recall_10:.2%} |')
    lines += ['', '## 同模型同参数的特征归因', '',
              '| 对照 | 模型 | NDCG差值 | Recall差值 | 不退步月份 | 达到额外价值门槛 |', '|---|---|---:|---:|---:|---|']
    for row in ablations.itertuples():
        lines.append(f'| {row.challenger} − {row.reference} | {row.family} | {row.ndcg_gain:+.4f} | '
                     f'{row.recall_gain:+.2%} | {row.nonworse_months}/3 | {row.demonstrated} |')
    lines += ['', 'A基础；B基础+画像及交互；C基础+EASE；D完整。同类模型四组使用相同参数，'
              '参数仅由五个开发月、四组平均NDCG确定。B-A及D-C检验画像，C-A及D-B检验EASE信号。', '',
              '## 选择、参数和边界', '',
              f'- 最佳新方案：{NAMES[lock["best_new"]]}。相对纯补货：`{json.dumps(lock["improvement_vs_default"], ensure_ascii=False)}`。',
              '- 默认替换门槛：NDCG+0.01、Recall+1个百分点、未买Recall+1个百分点，至少2/3月NDCG不退步。',
              '- 画像额外价值门槛：同模型同参数NDCG+0.002、至少2/3月不退步、Recall下降不超过0.5个百分点。',
              f'- 各类共同参数：`{json.dumps(lock["shared_shortlist"], ensure_ascii=False)}`。',
              f'- 阶段固定EASE参数：`{json.dumps(lock["ease_spec"], ensure_ascii=False)}`。',
              f'- 最佳新方案减纯补货客户抽样：`{json.dumps(lock["best_vs_default_intervals"]["metrics"], ensure_ascii=False)}`。',
              '抽样为多次选型后的内部诊断，不是新的独立显著性证明。', '',
              '历史训练EASE逐月只使用此前交易，其正则也只由更早已成熟标签决定。元模型拟合只用30天答案已成熟的客户月。'
              '训练负例含难例及均匀样本，所有特征组共用；分类训练校正入样概率，并按客户月等权。'
              '排序在相同抽样查询组上优化LambdaRank，不声称无偏还原全目录损失；验证和复核对完整目录打分。'
              '分数只用于同一客户同一方法的推荐排序，不是校准购买概率。', '',
              '1–5月五轮向前开发共240次模型拟合；6–8月确定默认并锁定最终模型；9–11月结果此前已查看，'
              '这里只作固定复核，不据此改参数。全局模型/目录阶段内冻结，客户历史与可观察商品统计按月更新。', '',
              '## 打开与复现', '', '`python -B -m recommendation.item_experiment`', '',
              '`python -m streamlit run app/dashboard.py` → Product recommendations → 商品级打分对比。', '',
              'development_monthly.csv为240次开发结果；validation_ablations.csv为完整同参数对照；'
              'training_audit.json为时间/采样审计；training_samples.joblib为正式可复现训练资料，含标签，仅供离线。'
              '页面只读profiles.csv及*_lists.csv，不读取*_customers.csv中的未来答案。', '',
              f'运行耗时约{elapsed:.0f}秒；全部模型、代码、来源及产物有哈希核对。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
