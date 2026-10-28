"""Fixed-split, ranking-first comparison. Run with python -B -m value_prediction.ranking_comparison."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from stage4.kmeans import FEATURES as BASE_FEATURES
from value_prediction.experiments import TARGET, attach_labels, top20_mask
from value_prediction.labels import CLEAN, SCHEDULE, SNAPSHOTS, sha256
from value_prediction.optimization import fit_rows_for_date
from value_prediction.temporal_features import FEATURE_PATH, MANIFEST_PATH, FEATURES as EXTRA_FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/ranking_comparison_v1'
PROTOCOL = ROOT / 'value_prediction/ranking_comparison_protocol.md'
LABELS = ROOT / 'value_prediction/outputs/spend_labels.csv'
FEATURES = BASE_FEATURES + EXTRA_FEATURES
KEYS = ['snapshot_date', 'customer_id']
TUNE_DATES = pd.date_range('2011-01-01', '2011-05-01', freq='MS')
VALIDATION_DATES = pd.date_range('2011-06-01', '2011-08-01', freq='MS')
REVIEW_DATES = pd.date_range('2011-09-01', '2011-11-01', freq='MS')
FAMILIES = ('lgbm_tweedie', 'xgb_tweedie', 'lgbm_two_stage')
RULES = ('spend_180d', 'spend_90d', 'spend_30d')
METHODS = RULES + FAMILIES
NAMES = {'spend_180d': '过去180天消费排序', 'spend_90d': '过去90天消费排序',
         'spend_30d': '过去30天消费排序', 'lgbm_tweedie': 'LightGBM Tweedie',
         'xgb_tweedie': 'XGBoost Tweedie', 'lgbm_two_stage': '两步LightGBM'}
EXPORT_COLUMNS = KEYS + ['priority_rank', 'priority_percentile', 'priority_band',
                         'baseline_rank', 'recency_days', 'monetary_value_180d', 'ranking_method']
SEED = 42


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path: Path, data: pd.DataFrame) -> None:
    data.to_csv(path, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')


def configurations() -> list[dict]:
    return [dict(id=f'{family}_{trees}_{leaves}_{profile}', family=family,
                 n_estimators=trees, num_leaves=leaves, profile=profile)
            for family in FAMILIES for trees in (120, 240)
            for leaves in (15, 31) for profile in ('standard', 'conservative')]


def model_parameters(spec: dict, seed: int = SEED) -> dict:
    conservative = spec['profile'] == 'conservative'
    common = dict(n_estimators=spec['n_estimators'], learning_rate=.05,
                  random_state=seed, n_jobs=1, reg_lambda=5. if conservative else 0.,
                  subsample=.8 if conservative else 1., colsample_bytree=.9 if conservative else 1.)
    if spec['family'] == 'xgb_tweedie':
        return dict(**common, objective='reg:tweedie', tweedie_variance_power=1.5,
                    tree_method='hist', grow_policy='lossguide', max_depth=0,
                    max_leaves=spec['num_leaves'], min_child_weight=10 if conservative else 1,
                    max_bin=255, verbosity=0)
    return dict(**common, num_leaves=spec['num_leaves'],
                min_child_samples=100 if conservative else 20,
                subsample_freq=1 if conservative else 0, verbose=-1)


def source_hashes() -> dict:
    return {name: sha256(path) for name, path in {
        'snapshots': SNAPSHOTS, 'labels': LABELS, 'temporal_features': FEATURE_PATH,
        'clean_transactions': CLEAN, 'schedule': SCHEDULE,
    }.items()}


def verify_sources() -> dict:
    hashes = source_hashes()
    temporal = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    labels = json.loads((LABELS.parent / 'label_manifest.json').read_text(encoding='utf-8'))
    mapping = {'snapshots': 'snapshot_sha256', 'labels': 'labels_sha256',
               'temporal_features': 'features_sha256', 'clean_transactions': 'clean_sha256',
               'schedule': 'schedule_sha256'}
    for name, key in mapping.items():
        manifest = labels if name == 'labels' else temporal
        if hashes[name] != manifest[key]:
            raise ValueError(f'Ranking input hash mismatch: {name}')
    if not temporal['all_180d_and_lifetime_amounts_reconciled'] or not labels['all_reconciled']:
        raise ValueError('Upstream amount reconciliation failed')
    return hashes


def load_data(splits: tuple[str, ...]) -> pd.DataFrame:
    """Keep selection and review labels in separate frames and call sites."""
    snapshots = pd.read_csv(SNAPSHOTS, dtype={'customer_id': 'string'}, parse_dates=['snapshot_date'])
    snapshots = snapshots.loc[snapshots.split.isin(splits)].copy()
    snapshots.snapshot_date = snapshots.snapshot_date.dt.normalize()
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'}, parse_dates=['snapshot_date'])
    labels = labels.loc[labels.split.isin(splits)].copy()
    data = attach_labels(snapshots, labels)
    temporal = pd.read_csv(FEATURE_PATH, dtype={'customer_id': 'string'}, parse_dates=['snapshot_date'])
    if temporal.duplicated(KEYS).any():
        raise ValueError('Duplicate temporal features')
    data = data.merge(temporal, on=KEYS, how='left', validate='one_to_one')
    values = data[FEATURES + [TARGET]].to_numpy(dtype=float)
    if (data.empty or data.duplicated(KEYS).any() or not np.isfinite(values).all()
            or (values < 0).any()):
        raise ValueError('Invalid ranking cohort or feature/label values')
    if len(FEATURES) != 20 or len(set(FEATURES)) != 20:
        raise ValueError('Expected exactly 20 historical inputs')
    return data.sort_values(KEYS, kind='stable').reset_index(drop=True)


def training_rows(data: pd.DataFrame, stage: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    allowed = ('train', 'validation') if stage == 'review' else ('train',)
    if stage not in ('tuning', 'validation', 'review'):
        raise ValueError('Unknown training stage')
    fit = fit_rows_for_date(data.loc[data.split.isin(allowed)], pd.Timestamp(cutoff))
    if fit.split.eq('test').any() or not (fit.snapshot_date < cutoff).all():
        raise ValueError('Evaluation rows entered model training')
    return fit


def fit_model(spec: dict, fit: pd.DataFrame, seed: int = SEED) -> dict:
    params = model_parameters(spec, seed)
    x, y = fit[FEATURES], fit[TARGET]
    bundle = dict(spec=spec, features=FEATURES, seed=seed)
    if spec['family'] == 'xgb_tweedie':
        model = xgb.XGBRegressor(**params)
        model.fit(x, y)
        bundle['regressor'] = model
    elif spec['family'] == 'lgbm_tweedie':
        model = lgb.LGBMRegressor(**params, objective='tweedie', tweedie_variance_power=1.5)
        model.fit(x, y)
        bundle['regressor'] = model
    elif spec['family'] == 'lgbm_two_stage':
        classifier_params = dict(params, n_estimators=max(80, spec['n_estimators'] * 2 // 3))
        classifier = lgb.LGBMClassifier(**classifier_params, objective='binary')
        classifier.fit(x, y.gt(0).astype(int))
        positive = y.gt(0)
        if positive.sum() < 100:
            raise ValueError('Too few purchasing examples')
        amount = lgb.LGBMRegressor(**params, objective='gamma')
        amount.fit(x.loc[positive], y.loc[positive])
        bundle.update(classifier=classifier, positive_regressor=amount)
    else:
        raise ValueError('Unknown model family')
    return bundle


def predict(bundle: dict, data: pd.DataFrame) -> np.ndarray:
    """Only named historical inputs are accessed, never a target or customer ID."""
    x = data[bundle['features']]
    if 'classifier' in bundle:
        score = bundle['classifier'].predict_proba(x)[:, 1] * bundle['positive_regressor'].predict(x)
    else:
        score = bundle['regressor'].predict(x)
    score = np.maximum(np.asarray(score, dtype=float), 0)
    if len(score) != len(data) or not np.isfinite(score).all():
        raise ValueError('Invalid model output')
    return score


def rule_score(method: str, data: pd.DataFrame) -> np.ndarray:
    column, divisor = {'spend_30d': ('monetary_value_30d', 1),
                       'spend_90d': ('monetary_value_90d', 3),
                       'spend_180d': ('monetary_value_180d', 6)}[method]
    return data[column].to_numpy(dtype=float) / divisor


def sorted_order(data: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    ranked = data[['customer_id']].reset_index(drop=True).assign(score=score, position=np.arange(len(data)))
    return ranked.sort_values(['score', 'customer_id'], ascending=[False, True], kind='stable').position.to_numpy()


def one_month_metrics(data: pd.DataFrame, score: np.ndarray) -> dict:
    score = np.asarray(score, dtype=float)
    if (len(score) != len(data) or data.empty or data.snapshot_date.nunique() != 1
            or data.duplicated(KEYS).any() or not np.isfinite(score).all() or (score < 0).any()):
        raise ValueError('Metrics require one unique month and valid scores')
    y = data[TARGET].to_numpy(dtype=float)
    actual = float(y.sum())
    if not np.isfinite(y).all() or (y < 0).any() or actual <= 0:
        raise ValueError('Invalid evaluation targets')
    order = sorted_order(data, score)
    quota = int(np.ceil(.2 * len(data)))
    chosen = order[:quota]
    result = dict(customers=len(data), actual_spend=actual, top20_count=quota,
                  top20_actual_spend=float(y[chosen].sum()),
                  mae=float(np.abs(score - y).mean()), top20_buyer_share=float((y[chosen] > 0).mean()))
    for percent in (10, 20, 30):
        n = int(np.ceil(percent / 100 * len(data)))
        result[f'capture_{percent}'] = float(y[order[:n]].sum() / actual)
    oracle = np.sort(y)[::-1][:quota].sum()
    result['oracle_capture_20'] = float(oracle / actual)
    return result


def evaluate_predictions(data: pd.DataFrame, scores: np.ndarray, method: str, phase: str,
                         config_id: str = '', seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for date, part in data.groupby('snapshot_date', sort=True):
        positions = data.index.get_indexer(part.index)
        rows.append(dict(phase=phase, method=method, config_id=config_id, seed=seed,
                         snapshot_date=str(date.date()), **one_month_metrics(part, scores[positions])))
    pred = data[KEYS + ['split', TARGET]].copy()
    pred['method'], pred['config_id'], pred['seed'], pred['score'] = method, config_id, seed, scores
    pred['phase'] = phase
    return pd.DataFrame(rows), pred


def summarize(monthly: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key, part in monthly.groupby(['phase', 'method', 'config_id', 'seed'], dropna=False, sort=False):
        phase, method, config_id, seed = key
        records.append(dict(phase=phase, method=method, config_id=config_id, seed=int(seed),
                            months=len(part), customers=int(part.customers.sum()),
                            mean_capture_20=float(part.capture_20.mean()),
                            pooled_capture_20=float(part.top20_actual_spend.sum() / part.actual_spend.sum()),
                            worst_capture_20=float(part.capture_20.min()),
                            mean_capture_10=float(part.capture_10.mean()), mean_capture_30=float(part.capture_30.mean()),
                            mae=float(np.average(part.mae, weights=part.customers))))
    return pd.DataFrame(records)


def choose_tuning_config(summary: pd.DataFrame, specs: list[dict], family: str) -> dict:
    lookup = {s['id']: s for s in specs}
    part = summary.loc[summary.phase.eq('tuning') & summary.method.eq(family)].copy()
    if len(part) != 8 or not part.months.eq(5).all():
        raise ValueError('Expected all eight configurations and five development folds')
    part['trees'] = part.config_id.map(lambda key: lookup[key]['n_estimators'])
    part['leaves'] = part.config_id.map(lambda key: lookup[key]['num_leaves'])
    part['conservative_first'] = part.config_id.map(lambda key: lookup[key]['profile'] != 'conservative')
    best = part.sort_values(['mean_capture_20', 'trees', 'leaves', 'conservative_first', 'config_id'],
                            ascending=[False, True, True, True, True], kind='stable').iloc[0]
    return lookup[best.config_id]


def validation_leaders(summary: pd.DataFrame, monthly: pd.DataFrame) -> tuple[str, str]:
    """Explicitly discard review rows, even if a caller supplies them."""
    valid = summary.loc[summary.phase.eq('validation')].copy()
    months = monthly.loc[monthly.phase.eq('validation')]
    expected = set(VALIDATION_DATES.strftime('%Y-%m-%d'))
    if set(valid.method) != set(METHODS) or len(valid) != len(METHODS):
        raise ValueError('Validation comparison is incomplete')
    for method in METHODS:
        part = months.loc[months.method.eq(method)]
        if len(part) != 3 or set(part.snapshot_date) != expected:
            raise ValueError('Validation must contain the same three months for every method')
    def best(methods):
        return sorted(methods, key=lambda m: (-float(valid.loc[valid.method.eq(m), 'mean_capture_20'].iloc[0]), methods.index(m)))[0]
    return best(RULES), best(FAMILIES)


def weighted_capture(y: np.ndarray, weights: np.ndarray, order: np.ndarray) -> float:
    """Equivalent to repeating sampled customer rows, including a partial boundary tie."""
    w = weights[order]
    quota = int(np.ceil(.2 * weights.sum()))
    selected_weights = np.minimum(w, np.maximum(0, quota - (np.cumsum(w) - w)))
    return float(np.dot(y[order], selected_weights) / np.dot(y, weights))


def paired_bootstrap(data: pd.DataFrame, model_scores: np.ndarray, baseline_scores: np.ndarray,
                     resamples: int = 500, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    if data.duplicated(KEYS).any() or resamples < 2:
        raise ValueError('Invalid paired customer bootstrap')
    ids = np.sort(data.customer_id.unique())
    codes = pd.Index(ids).get_indexer(data.customer_id)
    groups = []
    point = []
    for _, part in data.groupby('snapshot_date', sort=True):
        pos = data.index.get_indexer(part.index)
        y = part[TARGET].to_numpy(dtype=float)
        a, b = sorted_order(part, model_scores[pos]), sorted_order(part, baseline_scores[pos])
        groups.append((y, codes[pos], a, b))
        point.append(100 * (weighted_capture(y, np.ones(len(part), dtype=int), a)
                            - weighted_capture(y, np.ones(len(part), dtype=int), b)))
    rng = np.random.default_rng(seed)
    values = []
    for i in range(resamples):
        counts = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        gains = [100 * (weighted_capture(y, counts[code], a) - weighted_capture(y, counts[code], b))
                 for y, code, a, b in groups]
        values.append(dict(resample=i + 1, mean_gain_pp=float(np.mean(gains))))
    draws = pd.DataFrame(values)
    lo, hi = np.quantile(draws.mean_gain_pp, [.025, .975])
    return draws, dict(point_gain_pp=float(np.mean(point)), lower_95_pp=float(lo),
                       upper_95_pp=float(hi), resamples=resamples, seed=seed,
                       customer_clusters=len(ids), scope='validation_customer_resampling_after_selection')


def decide(monthly: pd.DataFrame, baseline: str, model: str, interval: dict) -> dict:
    valid = monthly.loc[monthly.phase.eq('validation')]
    a = valid.loc[valid.method.eq(model)].set_index('snapshot_date').capture_20
    b = valid.loc[valid.method.eq(baseline)].set_index('snapshot_date').capture_20
    if len(a) != 3 or not a.index.equals(b.index):
        raise ValueError('Invalid validation month pairing')
    gain = 100 * (a - b)
    conditions = dict(mean_gain_at_least_1pp=bool(gain.mean() >= 1 - 1e-10),
                      at_least_two_months_nonworse=bool((gain >= -1e-10).sum() >= 2),
                      worst_month_loss_at_most_1pp=bool(gain.min() >= -1 - 1e-10),
                      customer_interval_lower_positive=bool(interval['lower_95_pp'] > 0))
    winner = model if all(conditions.values()) else baseline
    return dict(selected_method=winner, selected_name=NAMES[winner], best_rule=baseline,
                best_model=model, validation_gain_pp=float(gain.mean()),
                monthly_gain_pp={k: float(v) for k, v in gain.items()},
                conditions=conditions, customer_interval=interval, selection_phase='validation',
                independent_test_available=False)


def make_priority(data: pd.DataFrame, score: np.ndarray, baseline_score: np.ndarray, method: str) -> pd.DataFrame:
    """Ranking/export intentionally accepts a feature-only table with no future labels."""
    if data.empty or data.duplicated(KEYS).any():
        raise ValueError('Invalid priority cohort')
    for values in (score, baseline_score):
        if len(values) != len(data) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid priority scores')
    parts = []
    for _, part in data.groupby('snapshot_date', sort=True):
        pos = data.index.get_indexer(part.index)
        order = sorted_order(part, score[pos])
        base_order = sorted_order(part, baseline_score[pos])
        base_ranks = np.empty(len(part), dtype=int)
        base_ranks[base_order] = np.arange(1, len(part) + 1)
        out = part[KEYS + ['recency_days', 'monetary_value_180d']].iloc[order].copy()
        out['priority_rank'] = np.arange(1, len(out) + 1)
        out['priority_percentile'] = 100 * (len(out) - out.priority_rank + 1) / len(out)
        out['priority_band'] = np.where(out.priority_rank <= np.ceil(.2 * len(out)), 'Top 20%', 'Other')
        out['baseline_rank'] = base_ranks[order]
        out['ranking_method'] = method
        parts.append(out[EXPORT_COLUMNS])
    return pd.concat(parts, ignore_index=True)


def fit_audit(fit: pd.DataFrame, stage: str, cutoff: pd.Timestamp) -> dict:
    return dict(stage=stage, prediction_start=str(cutoff.date()), rows=len(fit),
                splits=sorted(fit.split.unique()), latest_snapshot=str(fit.snapshot_date.max().date()),
                latest_label_end=str((fit.snapshot_date.max() + pd.Timedelta(days=30)).date()),
                test_rows=int(fit.split.eq('test').sum()))


def locked_plan(hashes: dict) -> dict:
    specs = configurations()
    return dict(version='ranking_comparison_v1', features=FEATURES,
                primary_metric='mean_monthly_capture_at_20pct', source_hashes=hashes,
                protocol_sha256=sha256(PROTOCOL), script_sha256=sha256(Path(__file__)),
                tuning_dates=list(TUNE_DATES.strftime('%Y-%m-%d')),
                validation_dates=list(VALIDATION_DATES.strftime('%Y-%m-%d')),
                review_dates=list(REVIEW_DATES.strftime('%Y-%m-%d')), review_previously_viewed=True,
                specifications=specs, parameters={s['id']: model_parameters(s) for s in specs},
                seed=SEED, seed_diagnostics=[7, 2026], customer_bootstrap_resamples=500,
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              pandas=pd.__version__, lightgbm=lgb.__version__, xgboost=xgb.__version__))


def run() -> None:
    started = time.perf_counter()
    hashes = verify_sources()
    plan = locked_plan(hashes)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT / 'locked_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Plan/code/source changed; preserve this run and use a new version directory')
        manifest_path = OUTPUT / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for name, expected in manifest['outputs_sha256'].items():
                if sha256(OUTPUT / name) != expected:
                    raise ValueError(f'Completed comparison changed: {name}')
            print(f'Verified existing complete run: {OUTPUT}', flush=True)
            return
    else:
        write_json(plan_path, plan)
    selection_data = load_data(('train', 'validation'))
    specs = configurations()
    audits, tuning_rows, timings = [], [], []
    for date in TUNE_DATES:
        fit = training_rows(selection_data, 'tuning', date)
        holdout = selection_data.loc[selection_data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        audits.append(fit_audit(fit, 'tuning', date))
        for spec in specs:
            tick = time.perf_counter()
            bundle = fit_model(spec, fit)
            scores = predict(bundle, holdout)
            metric, _ = evaluate_predictions(holdout, scores, spec['family'], 'tuning', spec['id'])
            tuning_rows.append(metric)
            timings.append(dict(phase='tuning', config_id=spec['id'], snapshot_date=str(date.date()),
                                seconds=time.perf_counter() - tick))
        print(f'Tuned all 24 configurations for {date.date()}', flush=True)
    tuning_monthly = pd.concat(tuning_rows, ignore_index=True)
    tuning_summary = summarize(tuning_monthly)
    shortlist = {family: choose_tuning_config(tuning_summary, specs, family) for family in FAMILIES}
    save_csv(OUTPUT / 'tuning_monthly.csv', tuning_monthly)
    save_csv(OUTPUT / 'tuning_comparison.csv', tuning_summary)
    write_json(OUTPUT / 'shortlist.json', shortlist)

    fit = training_rows(selection_data, 'validation', VALIDATION_DATES.min())
    audits.append(fit_audit(fit, 'validation', VALIDATION_DATES.min()))
    validation = selection_data.loc[selection_data.split.eq('validation')].copy().reset_index(drop=True)
    if set(validation.snapshot_date) != set(VALIDATION_DATES):
        raise ValueError('Unexpected validation dates')
    bundles, score_map, month_parts, prediction_parts = {}, {}, [], []
    for method in METHODS:
        tick = time.perf_counter()
        if method in RULES:
            score_map[method] = rule_score(method, validation)
            config_id = 'rule'
        else:
            bundles[method] = fit_model(shortlist[method], fit)
            score_map[method] = predict(bundles[method], validation)
            config_id = shortlist[method]['id']
        metrics, preds = evaluate_predictions(validation, score_map[method], method, 'validation', config_id)
        month_parts.append(metrics)
        prediction_parts.append(preds)
        timings.append(dict(phase='validation', config_id=config_id, seconds=time.perf_counter() - tick))
    validation_monthly = pd.concat(month_parts, ignore_index=True)
    validation_summary = summarize(validation_monthly)
    best_rule, best_model = validation_leaders(validation_summary, validation_monthly)
    draws, interval = paired_bootstrap(validation, score_map[best_model], score_map[best_rule])
    decision = decide(validation_monthly, best_rule, best_model, interval)
    selected = decision['selected_method']
    print(f'Validation choice: {selected}; best model {best_model}; rule {best_rule}', flush=True)
    save_csv(OUTPUT / 'validation_predictions.csv', pd.concat(prediction_parts, ignore_index=True))
    save_csv(OUTPUT / 'bootstrap_draws.csv', draws)

    seed_rows = []
    for family in FAMILIES:
        for seed in (7, 2026):
            candidate = fit_model(shortlist[family], fit, seed)
            metrics, _ = evaluate_predictions(validation, predict(candidate, validation), family,
                                               'seed_diagnostic', shortlist[family]['id'], seed)
            seed_rows.append(metrics)
    save_csv(OUTPUT / 'seed_monthly.csv', pd.concat(seed_rows, ignore_index=True))

    final_fit = training_rows(selection_data, 'review', REVIEW_DATES.min())
    audits.append(fit_audit(final_fit, 'review_frozen_model', REVIEW_DATES.min()))
    final_bundles = {family: fit_model(shortlist[family], final_fit) for family in FAMILIES}
    model_path = OUTPUT / 'ranking_bundle.joblib'
    joblib.dump(dict(features=FEATURES, validation=bundles, review=final_bundles,
                     selected_method=selected, baseline_method=best_rule), model_path, compress=3)
    lock = dict(**decision, shortlisted_configs=shortlist,
                selected_parameters=(model_parameters(shortlist[selected]) if selected in FAMILIES else {}),
                source_hashes=hashes, plan_sha256=sha256(plan_path), model_sha256=sha256(model_path),
                training_audit=audits, review_labels_used_for_training_or_selection=False,
                review_model_updates=False)
    lock_path = OUTPUT / 'locked_selection.json'
    write_json(lock_path, lock)
    lock_hash = sha256(lock_path)
    print('Selection and all model artifacts locked; now computing September-November review.', flush=True)

    # This is the first review-label loading/evaluation call in this run.
    review = load_data(('test',))
    if set(review.snapshot_date) != set(REVIEW_DATES):
        raise ValueError('Unexpected review dates')
    review_scores = {}
    review_predictions = []
    for method in METHODS:
        scores = rule_score(method, review) if method in RULES else predict(final_bundles[method], review)
        review_scores[method] = scores
        metrics, preds = evaluate_predictions(review, scores, method, 'review',
                                               'rule' if method in RULES else shortlist[method]['id'])
        month_parts.append(metrics)
        review_predictions.append(preds)
    if sha256(lock_path) != lock_hash or sha256(model_path) != lock['model_sha256']:
        raise ValueError('Review modified the selected method/model')
    save_csv(OUTPUT / 'review_predictions.csv', pd.concat(review_predictions, ignore_index=True))
    monthly = pd.concat(month_parts, ignore_index=True)
    comparison = summarize(monthly)
    save_csv(OUTPUT / 'monthly_comparison.csv', monthly)
    save_csv(OUTPUT / 'comparison.csv', comparison)
    save_csv(OUTPUT / 'training_times.csv', pd.DataFrame(timings))
    write_json(OUTPUT / 'training_audit.json', {'folds': audits})
    priority = pd.concat([
        make_priority(validation.drop(columns=TARGET), score_map[selected], score_map[best_rule], selected),
        make_priority(review.drop(columns=TARGET), review_scores[selected], review_scores[best_rule], selected),
    ], ignore_index=True)
    save_csv(OUTPUT / 'customer_priority.csv', priority)
    paired = []
    for phase in ('validation', 'review'):
        a = monthly.loc[monthly.phase.eq(phase) & monthly.method.eq(selected)].set_index('snapshot_date')
        b = monthly.loc[monthly.phase.eq(phase) & monthly.method.eq(best_rule)].set_index('snapshot_date')
        for date, row in a.iterrows():
            paired.append(dict(snapshot_date=date, phase=phase, method=selected, baseline_method=best_rule,
                               customers=int(row.customers), top20_count=int(row.top20_count),
                               model_capture=float(row.capture_20), baseline_capture=float(b.loc[date, 'capture_20']),
                               gain_pp=float(100 * (row.capture_20 - b.loc[date, 'capture_20']))))
    save_csv(OUTPUT / 'priority_monthly.csv', pd.DataFrame(paired))
    make_figure(monthly, selected)
    report(comparison, monthly, lock, tuning_summary)
    outputs = [p for p in OUTPUT.iterdir() if p.is_file() and p.name != 'manifest.json']
    write_json(OUTPUT / 'manifest.json', dict(status='completed_internal_ranking_comparison',
               selected_method=selected, baseline_method=best_rule, source_hashes=hashes,
               independent_test_available=False, selection_phase='validation',
               outputs_sha256={p.name: sha256(p) for p in sorted(outputs)},
               elapsed_seconds=time.perf_counter() - started))
    print(f'Completed comparison and verified exports: {OUTPUT}', flush=True)


def make_figure(monthly: pd.DataFrame, selected: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, phase, title in zip(axes, ('validation', 'review'), ('Validation: Jun-Aug', 'Review: Sep-Nov')):
        part = monthly.loc[monthly.phase.eq(phase)]
        dates = sorted(part.snapshot_date.unique())
        for method in METHODS:
            rows = part.loc[part.method.eq(method)].set_index('snapshot_date').loc[dates]
            ax.plot(np.arange(3), 100 * rows.capture_20, marker='o', label=method,
                    linewidth=2.8 if method == selected else 1.2, alpha=1 if method == selected else .65)
        ax.set_xticks(range(3), [d[:7] for d in dates])
        ax.set_title(title)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Actual spend captured by Top 20% (%)')
    axes[1].legend(fontsize=8, loc='best')
    fig.suptitle(f'Fixed-split comparison | selected: {selected}')
    fig.tight_layout()
    fig.savefig(OUTPUT / 'comparison.png', dpi=160)
    plt.close(fig)


def report(comparison: pd.DataFrame, monthly: pd.DataFrame, lock: dict, tuning: pd.DataFrame) -> None:
    selected, baseline = lock['selected_method'], lock['best_rule']
    lines = ['# 高消费潜力排序：统一比较结果', '',
             f'**最终采用：{NAMES[selected]}。** 只按6–8月验证结果选型，9–11月为锁定后的复核结果。', '',
             '目标：每月挑选20%客户，覆盖更多后续30天合格消费额。'
             '所有学习模型共用20项已知购买特征，每类8组参数、5个向前开发折。', '',
             '| 方法 | 验证：月均覆盖 | 验证：金额加总覆盖 | 复核：月均覆盖 | 复核：金额加总覆盖 |',
             '|---|---:|---:|---:|---:|']
    for method in METHODS:
        a = comparison.loc[comparison.phase.eq('validation') & comparison.method.eq(method)].iloc[0]
        b = comparison.loc[comparison.phase.eq('review') & comparison.method.eq(method)].iloc[0]
        lines.append(f'| {NAMES[method]} | {a.mean_capture_20:.2%} | {a.pooled_capture_20:.2%} | '
                     f'{b.mean_capture_20:.2%} | {b.pooled_capture_20:.2%} |')
    lines += ['', '## 为什么选它', '',
              f'- 最强规则：{NAMES[baseline]}；最强学习方案：{NAMES[lock["best_model"]]}。',
              f'- 最强学习方案验证期月均覆盖相对最强规则：{lock["validation_gain_pp"]:+.2f}个百分点。',
              '- 固定条件：平均提升≥1个百分点、至少2/3个月不退步、最差月落后≤1个百分点，'
              '配对客户重抽样区间下界>0。',
              f'- 条件结果：`{json.dumps(lock["conditions"], ensure_ascii=False)}`。',
              ('- 四项满足，采用验证期胜出的学习方案。' if selected in FAMILIES
               else '- 学习方案未满足全部内部选型条件，采用验证期最强消费排序规则。'), '',
              '## 最终参数与公平比较', '',
              '所有模型均不使用测试标签拟合。验证模型只训练到2011年5月，'
              '最终模型只训练到2011年8月；后者固定评分9、10、11月，中途不更新。', '',
              '| 学习方案 | 开发选定配置 | 开发月均覆盖 |', '|---|---|---:|']
    for family, spec in lock['shortlisted_configs'].items():
        metric = tuning.loc[tuning.config_id.eq(spec['id'])].iloc[0]
        lines.append(f'| {NAMES[family]} | {spec["n_estimators"]}棵回归树，{spec["num_leaves"]}叶，'
                     f'{spec["profile"]} | {metric.mean_capture_20:.2%} |')
    lines += ['', '学习率均为0.05、种子42；Tweedie方差幂1.5。两步模型分类树数为回归树数的2/3，'
              '用购买概率乘购买后的Gamma金额均值排序。全部具体参数见locked_plan.json及locked_selection.json。', '',
              '## 逐月名单效果', '', '| 月份 | 用途 | 客户数 | 20%名额 | 最终方法覆盖 | 最强规则覆盖 | 差值 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for phase in ('validation', 'review'):
        a = monthly.loc[monthly.phase.eq(phase) & monthly.method.eq(selected)].set_index('snapshot_date')
        b = monthly.loc[monthly.phase.eq(phase) & monthly.method.eq(baseline)].set_index('snapshot_date')
        for date, row in a.iterrows():
            lines.append(f'| {date[:7]} | {"验证" if phase == "validation" else "复核"} | '
                         f'{int(row.customers):,} | {int(row.top20_count):,} | {row.capture_20:.2%} | '
                         f'{b.loc[date, "capture_20"]:.2%} | {100 * (row.capture_20 - b.loc[date, "capture_20"]):+.2f}个百分点 |')
    ci = lock['customer_interval']
    lines += ['', '## 稳健性与使用', '',
              f'验证期500次客户成组配对重抽样：最强学习方案对最强规则的月均提升为'
              f'{ci["point_gain_pp"]:+.2f}个百分点，95%客户抽样区间'
              f'[{ci["lower_95_pp"]:+.2f}, {ci["upper_95_pp"]:+.2f}]个百分点。'
              '这是选型后的内部诊断，不包含月份变化带来的不确定性。另两个种子的逐月结果见seed_monthly.csv。', '',
              '此前这些月份的金额结果已经用于研究，本轮不能提供新独立测试。'
              '模型比较和20%名单效用与个人金额预测准确度是不同指标；名单百分位只表示当月相对名次。', '',
              '![逐月比较](comparison.png)', '',
              '复现：`python -B -m value_prediction.ranking_comparison`；'
              '打开：`python -m streamlit run app/dashboard.py` → High-value customers。', '',
              'comparison.csv为总体对比，monthly_comparison.csv为逐月对比，tuning_comparison.csv保留全部24组参数；'
              'validation_predictions.csv/review_predictions.csv保留内部逐条评分与真实答案，'
              'customer_priority.csv为可导出的安全名单，不含未来答案或个人预计金额。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
