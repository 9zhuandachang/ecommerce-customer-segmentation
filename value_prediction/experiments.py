"""Compare 30-day spend prediction with historical-spend baselines.

Run ``python -m value_prediction.experiments`` after ``value_prediction.labels``.
Selection uses June-August 2011 only. September-November is read after the
decision and model artifact are frozen, and is reported with the caveat that
these calendar months were already examined for the separate repurchase task.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from stage4.kmeans import FEATURES, LABEL, load_snapshots
from value_prediction.labels import OUTPUT as LABEL_OUTPUT
from value_prediction.labels import SNAPSHOTS, CLEAN, SCHEDULE, sha256

TARGET = 'spend_30d'
LABELS = LABEL_OUTPUT / 'spend_labels.csv'
MANIFEST = LABEL_OUTPUT / 'label_manifest.json'
OUTPUT = ROOT / 'value_prediction/outputs/experiment'
MODEL = OUTPUT / 'model/spend_bundle.joblib'
LOCK = OUTPUT / 'locked_config.json'
METHODS = ('historic_30d', 'zero', 'direct_tweedie', 'two_stage', 'direct_l2')
MODEL_METHODS = ('direct_tweedie', 'two_stage', 'direct_l2')
TOP_FRACTION = .2
AMOUNT_BIAS_TOLERANCE = .10
OOF_DATES = pd.date_range('2011-03-01', '2011-05-01', freq='MS')


def attach_labels(snapshots: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Require a one-to-one match with the existing snapshot cohort."""
    keys = ['snapshot_date', 'customer_id']
    labels = labels.copy()
    labels['snapshot_date'] = pd.to_datetime(labels['snapshot_date'], errors='raise')
    labels['customer_id'] = labels['customer_id'].astype('string')
    if labels.duplicated(keys).any() or snapshots.duplicated(keys).any():
        raise ValueError('Duplicate customer-month key')
    merged = snapshots.merge(labels[keys + ['split', LABEL, TARGET]],
                             on=keys, how='left', validate='one_to_one',
                             suffixes=('', '_label'), indicator=True)
    if not merged._merge.eq('both').all() or len(merged) != len(snapshots):
        raise ValueError('Spend labels do not cover the modeling snapshots')
    if not merged.split.eq(merged.split_label).all() or not merged[LABEL].eq(
            merged[f'{LABEL}_label']).all():
        raise ValueError('Spend label split or repurchase label differs')
    if merged[TARGET].isna().any() or (merged[TARGET] < 0).any():
        raise ValueError('Invalid 30-day spend label')
    return merged.drop(columns=['split_label', f'{LABEL}_label', '_merge'])


def historical_score(frame: pd.DataFrame) -> np.ndarray:
    """Past 180-day eligible spend converted to a 30-day rate."""
    return (frame.monetary_value_180d.to_numpy(dtype=float) / 6).clip(min=0)


def train_model(method: str, fit: pd.DataFrame) -> dict:
    """Fit one of three fixed alternatives using pre-cutoff features."""
    x = fit[FEATURES]
    y = fit[TARGET].to_numpy(dtype=float)
    if method == 'direct_tweedie':
        model = lgb.LGBMRegressor(
            objective='tweedie', tweedie_variance_power=1.5,
            n_estimators=120, learning_rate=.05, num_leaves=15,
            random_state=42, n_jobs=1, verbose=-1)
        model.fit(x, y)
        return {'method': method, 'features': FEATURES, 'regressor': model}
    if method == 'direct_l2':
        model = lgb.LGBMRegressor(
            objective='regression_l2', n_estimators=120, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        model.fit(x, y)
        return {'method': method, 'features': FEATURES, 'regressor': model}
    if method == 'two_stage':
        classifier = lgb.LGBMClassifier(
            objective='binary', n_estimators=80, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        classifier.fit(x, (y > 0).astype(int))
        positive = y > 0
        if positive.sum() < 100:
            raise ValueError('Too few positive-spend snapshots')
        positive_regressor = lgb.LGBMRegressor(
            objective='gamma', n_estimators=120, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        positive_regressor.fit(x.loc[positive], y[positive])
        return {'method': method, 'features': FEATURES,
                'classifier': classifier, 'positive_regressor': positive_regressor}
    raise ValueError(f'Unknown method: {method}')


def predict_spend(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    x = frame[bundle['features']]
    if bundle['method'] in ('direct_tweedie', 'direct_l2'):
        scores = np.maximum(bundle['regressor'].predict(x), 0)
    elif bundle['method'] == 'two_stage':
        probability = bundle['classifier'].predict_proba(x)[:, 1]
        conditional_spend = bundle['positive_regressor'].predict(x)
        scores = probability * conditional_spend
    else:
        raise ValueError('Unknown saved spend model')
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError('Spend predictions must be finite and nonnegative')
    return scores


def top20_mask(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    ranked = frame[['snapshot_date', 'customer_id']].copy()
    ranked['score'] = scores
    ranked['row_number'] = np.arange(len(frame))
    ranked = ranked.sort_values(['snapshot_date', 'score', 'customer_id'],
                                ascending=[True, False, True], kind='stable')
    position = ranked.groupby('snapshot_date', sort=False).cumcount() + 1
    quota = np.ceil(ranked.groupby('snapshot_date').customer_id.transform('size')
                    * TOP_FRACTION)
    mask = np.zeros(len(frame), dtype=bool)
    mask[ranked.loc[position <= quota, 'row_number'].to_numpy(dtype=int)] = True
    return mask


def evaluate(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    y = frame[TARGET].to_numpy(dtype=float)
    selected = top20_mask(frame, scores)
    total = float(y.sum())
    selected_spend = float(y[selected].sum())
    return {
        'snapshot_count': len(frame), 'buyer_count': int((y > 0).sum()),
        'actual_spend': total, 'predicted_spend': float(scores.sum()),
        'aggregate_bias': float(scores.sum() / total - 1) if total > 0 else np.nan,
        'mae': float(mean_absolute_error(y, scores)),
        'wape': float(np.abs(y - scores).sum() / total) if total > 0 else np.nan,
        'positive_mae': float(mean_absolute_error(y[y > 0], scores[y > 0]))
        if (y > 0).any() else np.nan,
        'top20_count': int(selected.sum()),
        'top20_actual_spend': selected_spend,
        'top20_spend_capture': selected_spend / total if total > 0 else np.nan,
        'top20_spend_lift': (selected_spend / total) / (selected.mean())
        if total > 0 else np.nan,
    }


def compare_months(validation: pd.DataFrame, predictions: dict[str, np.ndarray],
                   period: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    month_rows = []
    prediction_rows = []
    for method, scores in predictions.items():
        scores = np.asarray(scores, dtype=float)
        summary = evaluate(validation, scores)
        if method == 'zero':
            summary.update(top20_count=0, top20_actual_spend=np.nan,
                           top20_spend_capture=np.nan, top20_spend_lift=np.nan)
        metric_rows.append({'period': period, 'method': method, **summary})
        output = validation[['snapshot_date', 'customer_id', 'split', LABEL, TARGET]].copy()
        output['method'] = method
        output['predicted_spend_30d'] = scores
        output['top20_selected'] = (top20_mask(validation, scores) if method != 'zero'
                                    else np.zeros(len(validation), dtype=bool))
        prediction_rows.append(output)
        for date, part in validation.groupby('snapshot_date', sort=True):
            local = scores[validation.index.get_indexer(part.index)]
            month_summary = evaluate(part, local)
            if method == 'zero':
                month_summary.update(top20_count=0, top20_actual_spend=np.nan,
                                     top20_spend_capture=np.nan, top20_spend_lift=np.nan)
            month_rows.append({'period': period, 'method': method,
                               'snapshot_date': date.strftime('%Y-%m-%d'),
                               **month_summary})
    return pd.DataFrame(metric_rows), pd.DataFrame(month_rows), pd.concat(
        prediction_rows, ignore_index=True)


def candidate_passes(metrics: pd.DataFrame, monthly: pd.DataFrame,
                     method: str) -> bool:
    """Check error, customer-value capture, and aggregate amount bias."""
    baseline = metrics.set_index('method').loc['historic_30d']
    baseline_months = monthly.loc[monthly.method.eq('historic_30d')].set_index('snapshot_date')
    row = metrics.set_index('method').loc[method]
    candidate_months = monthly.loc[monthly.method.eq(method)].set_index('snapshot_date')
    if (len(baseline_months) != 3 or len(candidate_months) != 3
            or not baseline_months.index.is_unique
            or not candidate_months.index.is_unique
            or not baseline_months.index.equals(candidate_months.index)):
        return False
    stable = int(((candidate_months.mae <= baseline_months.mae)
                  & (candidate_months.top20_spend_capture
                     >= baseline_months.top20_spend_capture)).sum())
    return bool(row.mae < baseline.mae
                and row.top20_spend_capture > baseline.top20_spend_capture
                and abs(row.aggregate_bias) <= AMOUNT_BIAS_TOLERANCE
                and stable >= 2)


def choose_model(metrics: pd.DataFrame, monthly: pd.DataFrame) -> str | None:
    """Among qualifying models, prefer lower MAE and then greater capture."""
    eligible = []
    for method in MODEL_METHODS:
        if candidate_passes(metrics, monthly, method):
            row = metrics.set_index('method').loc[method]
            eligible.append((float(row.mae), -float(row.top20_spend_capture), method))
    return min(eligible)[2] if eligible else None


def training_only_scale_check(train: pd.DataFrame, validation: pd.DataFrame,
                              validation_scores: np.ndarray) -> tuple[pd.DataFrame, float, dict]:
    """Test one global amount multiplier derived solely from earlier train folds."""
    rows = []
    for date in OOF_DATES:
        fit = train.loc[(train.snapshot_date + pd.Timedelta(days=30)).le(date)
                        & train.snapshot_date.lt(date)]
        heldout = train.loc[train.snapshot_date.eq(date)]
        if fit.empty or heldout.empty:
            raise ValueError(f'Empty training-only spend fold: {date.date()}')
        if fit.snapshot_date.max() + pd.Timedelta(days=30) > date:
            raise AssertionError('Training-only fold uses unavailable labels')
        scores = predict_spend(train_model('direct_tweedie', fit), heldout)
        rows.append({'target_date': date.strftime('%Y-%m-%d'),
                     'fit_count': len(fit), 'holdout_count': len(heldout),
                     'actual_spend': float(heldout[TARGET].sum()),
                     'predicted_spend': float(scores.sum())})
    table = pd.DataFrame(rows)
    factor = float(table.actual_spend.sum() / table.predicted_spend.sum())
    checked = evaluate(validation, validation_scores * factor)
    return table, factor, checked


def check_sources() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for path, key in [(SNAPSHOTS, 'snapshot_sha256'), (CLEAN, 'clean_sha256'),
                      (SCHEDULE, 'schedule_sha256'), (LABELS, 'labels_sha256'),
                      (LABEL_OUTPUT / 'spend_label_audit.csv', 'audit_sha256')]:
        if sha256(path) != manifest[key]:
            raise ValueError(f'Spend label source changed: {path}')
    if not manifest['all_reconciled']:
        raise ValueError('Spend labels did not reconcile')
    return manifest


def main() -> None:
    label_manifest = check_sources()
    data, _ = load_snapshots(SNAPSHOTS)
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    development = attach_labels(
        data.loc[data.split.isin(['train', 'validation'])].copy(),
        labels.loc[labels.split.isin(['train', 'validation'])].copy())
    train = development.loc[development.split.eq('train')].copy().reset_index(drop=True)
    validation = development.loc[development.split.eq('validation')].copy().reset_index(drop=True)
    if train.snapshot_date.max() + pd.Timedelta(days=30) > validation.snapshot_date.min():
        raise ValueError('Training spend labels cross into validation')
    validation_predictions = {
        'historic_30d': historical_score(validation),
        'zero': np.zeros(len(validation)),
    }
    for method in MODEL_METHODS:
        model = train_model(method, train)
        validation_predictions[method] = predict_spend(model, validation)
    oof_rows, oof_factor, scaled_validation = training_only_scale_check(
        train, validation, validation_predictions['direct_tweedie'])
    val_metrics, val_monthly, val_rows = compare_months(
        validation, validation_predictions, 'validation')
    selected = choose_model(val_metrics, val_monthly)
    validation_passed = selected is not None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    val_metrics.to_csv(OUTPUT / 'validation_comparison.csv', index=False,
                       encoding='utf-8-sig')
    val_monthly.to_csv(OUTPUT / 'monthly_validation.csv', index=False,
                       encoding='utf-8-sig')
    val_rows.to_csv(OUTPUT / 'validation_predictions.csv', index=False,
                    encoding='utf-8-sig', date_format='%Y-%m-%d')
    oof_rows.to_csv(OUTPUT / 'training_only_scale_check.csv', index=False,
                    encoding='utf-8-sig')
    lock = {
        'selected_on_validation': selected,
        'validation_acceptance_passed': validation_passed,
        'amount_bias_tolerance': AMOUNT_BIAS_TOLERANCE,
        'acceptance_rule': 'MAE below historic_30d and Top20 spend capture above it on pooled period; both no worse in at least 2 of 3 months; total predicted-spend bias within +/-10%',
        'target': TARGET, 'features': FEATURES,
        'train_count': len(train), 'validation_count': len(validation),
        'snapshot_sha256': sha256(SNAPSHOTS),
        'labels_sha256': sha256(LABELS),
        'label_manifest_sha256': sha256(MANIFEST),
        'script_sha256': sha256(Path(__file__)),
        'test_dates_previously_examined_for_repurchase': True,
        'test_spend_labels_used_for_selection': False,
        'training_only_scale_factor': oof_factor,
        'training_only_scaled_validation_mae': scaled_validation['mae'],
        'training_only_scaled_validation_bias': scaled_validation['aggregate_bias'],
        'training_only_scale_adopted': False,
    }
    if selected is not None:
        fit = pd.concat([train, validation], ignore_index=True)
        test_start = data.loc[data.split.eq('test'), 'snapshot_date'].min()
        if fit.snapshot_date.max() + pd.Timedelta(days=30) > test_start:
            raise ValueError('Model fit label crosses into test')
        bundle = train_model(selected, fit)
        bundle['snapshot_sha256'] = lock['snapshot_sha256']
        bundle['labels_sha256'] = lock['labels_sha256']
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL, compress=3)
        lock['model_sha256'] = sha256(MODEL)
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
    # Only now attach test spend targets and score the fixed selection.
    test = attach_labels(
        data.loc[data.split.eq('test')].copy(),
        labels.loc[labels.split.eq('test')].copy()).reset_index(drop=True)
    test_predictions = {'historic_30d': historical_score(test),
                        'zero': np.zeros(len(test))}
    if selected is not None:
        if sha256(MODEL) != lock['model_sha256']:
            raise ValueError('Frozen spend model changed')
        loaded = joblib.load(MODEL)
        test_predictions[selected] = predict_spend(loaded, test)
        np.testing.assert_array_equal(test_predictions[selected],
                                      predict_spend(joblib.load(MODEL), test))
    test_metrics, test_monthly, test_rows = compare_months(test, test_predictions, 'test')
    test_metrics.to_csv(OUTPUT / 'test_comparison.csv', index=False,
                        encoding='utf-8-sig')
    test_monthly.to_csv(OUTPUT / 'monthly_test.csv', index=False,
                        encoding='utf-8-sig')
    test_rows.to_csv(OUTPUT / 'test_predictions.csv', index=False,
                     encoding='utf-8-sig', date_format='%Y-%m-%d')
    test_passed = bool(selected is not None
                       and candidate_passes(test_metrics, test_monthly, selected))
    test_baseline = test_metrics.set_index('method').loc['historic_30d']
    test_selected = (test_metrics.set_index('method').loc[selected]
                     if selected is not None else None)
    test_mae_wins = (int((test_monthly.loc[test_monthly.method.eq(selected)]
                          .set_index('snapshot_date').mae
                          < test_monthly.loc[test_monthly.method.eq('historic_30d')]
                          .set_index('snapshot_date').mae).sum())
                     if selected is not None else 0)
    decision = {
        'selected_on_validation': selected,
        'validation_acceptance_passed': validation_passed,
        'retrospective_test_gate_passed': test_passed,
        'offline_model_accepted': bool(validation_passed and test_passed),
        'integrated_in_dashboard': False,
        'test_period_previously_seen_for_repurchase': True,
        'note': 'September-November spend results are a retrospective check; do not tune and reselect on them.',
        'lock_sha256': sha256(LOCK),
    }
    (OUTPUT / 'decision.json').write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    baseline = val_metrics.set_index('method').loc['historic_30d']
    lines = [
        '# 未来 30 天客户消费额预测', '',
        f'标签基于合格交易行金额，覆盖 {label_manifest["snapshot_count"]:,} 条客户月快照；'
        '零购买记为零。原始交易、清洗结果和原复购模型未改动。',
        '金额代表合格交易的消费额，不是净利润、CLV 或营销增量。', '',
        '模型只使用快照日前 180 天的 8 项特征。训练期至 2011 年 5 月；'
        '2011 年 6–8 月选模型；模型选择写入锁定配置后，才报告 9–11 月结果。'
        '这些测试月份在复购任务中曾被查看，不能称为整个项目完全未查看的时间段。', '',
        '验收：整体 MAE 低于历史 180 天金额折算的 30 天基线，'
        'Top 20% 实际消费额覆盖率高于该基线，至少 2/3 个月两项均不退步，'
        '预测总额偏差不超过 ±10%。验证期选模，测试期只检查已固定的方案。', '',
        'Top 20% 覆盖率按每月评分选前 20% 客户月快照，再用入选客户的实际消费额'
        '除以当期全部客户的实际消费额。', '',
        '| 期间 | 方法 | MAE | Top 20% 实际消费额覆盖 | WAPE | 总额偏差 | Top 20% 金额提升倍数 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in pd.concat([val_metrics, test_metrics]).itertuples(index=False):
        capture = f'{row.top20_spend_capture:.2%}' if np.isfinite(
            row.top20_spend_capture) else '—'
        lift = f'{row.top20_spend_lift:.2f}' if np.isfinite(
            row.top20_spend_lift) else '—'
        lines.append(f'| {row.period} | {row.method} | {row.mae:.2f} | '
                     f'{capture} | {row.wape:.2%} | {row.aggregate_bias:+.2%} | {lift} |')
    lines += ['', f'验证期历史金额基线：MAE {baseline.mae:.2f}，'
              f'Top 20% 消费额覆盖 {baseline.top20_spend_capture:.2%}。',
              f'选定方案：**{selected if selected else "无（未达到验收门槛）"}**。',
              f'验证期验收：**{"通过" if validation_passed else "未通过"}**；'
              f'已查看时间段的事后测试门槛：**{"通过" if test_passed else "未通过"}**。'
              f'离线模型结论：**{"接受" if validation_passed and test_passed else "暂不接受"}**。',
              '零预测只用于检验 MAE，不提供可用的客户排序；因此其 Top 20% 指标不计算。'
              '验证期零预测 MAE 低于候选模型，表明只看 MAE 会偏向全部预测不购买。',
              f'针对 direct_tweedie，训练期内部 3–5 月前向预测得到金额缩放系数 {oof_factor:.4f}；'
              f'用于验证期后 MAE {scaled_validation["mae"]:.2f}、'
              f'总额偏差 {scaled_validation["aggregate_bias"]:+.2%}，'
              '仍未通过预设门槛，因此没有采用该缩放。',
              (f'固定方案在 9–11 月的 MAE 为 {test_selected.mae:.2f}，'
               f'历史金额基线为 {test_baseline.mae:.2f}；'
               f'三个测试月中只有 {test_mae_wins} 个月的 MAE 更低。'
               f'Top 20% 消费额覆盖率为 {test_selected.top20_spend_capture:.2%}，'
               f'基线为 {test_baseline.top20_spend_capture:.2%}。'
               '总额偏差只衡量群体总量，不能证明单个客户金额准确。'
               if test_selected is not None else '验证期没有达标候选方案。'),
              (f'验证期实际购买占比 {baseline.buyer_count / baseline.snapshot_count:.2%}，'
               f'测试期 {test_baseline.buyer_count / test_baseline.snapshot_count:.2%}；'
               f'每条客户月快照的实际平均消费额从 '
               f'{baseline.actual_spend / baseline.snapshot_count:.2f} 增至 '
               f'{test_baseline.actual_spend / test_baseline.snapshot_count:.2f}。'
               '这是期间分布变化的观察，尚不能单凭汇总指标确定误差原因。'),
              '逐月指标、逐条预测、标签审计、冻结配置和采用决定见输出目录。'
              '当前未接入仪表板；测试月份已在复购任务中被查看，不能作为全项目全新独立验证。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'validation selected={selected}; baseline MAE={baseline.mae:.2f} '
          f'capture={baseline.top20_spend_capture:.2%}; outputs={OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
