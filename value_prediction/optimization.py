"""Reproducible development-only, forward-chaining spend comparison.

The previously viewed September-November period is excluded from all comparisons.
Run ``python -m value_prediction.temporal_features`` before this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from stage4.kmeans import FEATURES as BASE_FEATURES, load_snapshots
from value_prediction.calibration import lagged_amount_factor
from value_prediction.experiments import LABELS, TARGET, attach_labels, evaluate, top20_mask
from value_prediction.labels import CLEAN, SNAPSHOTS, SCHEDULE, sha256
from value_prediction.seasonal_features import (
    FEATURE_PATH as SEASONAL_PATH, MANIFEST_PATH as SEASONAL_MANIFEST,
)
from value_prediction.temporal_features import (
    FEATURE_PATH, MANIFEST_PATH, FEATURES as EXTRA_FEATURES,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/optimization'
DEVELOPMENT_DATES = pd.date_range('2011-01-01', '2011-08-01', freq='MS')
METHODS = ('historic_180d', 'recent_90d', 'base_tweedie',
           'extended_tweedie', 'extended_tweedie_lag_calibrated',
           'extended_tweedie_history_available',
           'extended_tweedie_previous_year',
           'extended_tweedie_previous_year_recent_weighted',
           'extended_poisson', 'extended_l2',
           'extended_two_stage', 'extended_two_stage_previous_year')


def check_sources() -> None:
    feature_manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    for path, key in [(SNAPSHOTS, 'snapshot_sha256'), (CLEAN, 'clean_sha256'),
                      (SCHEDULE, 'schedule_sha256'), (FEATURE_PATH, 'features_sha256')]:
        if sha256(path) != feature_manifest[key]:
            raise ValueError(f'Temporal feature source changed: {path}')
    if not feature_manifest['all_180d_and_lifetime_amounts_reconciled']:
        raise ValueError('Temporal feature reconciliation failed')
    label_manifest_path = LABELS.parent / 'label_manifest.json'
    label_manifest = json.loads(label_manifest_path.read_text(encoding='utf-8'))
    if sha256(LABELS) != label_manifest['labels_sha256']:
        raise ValueError('Spend labels changed')
    seasonal_manifest = json.loads(SEASONAL_MANIFEST.read_text(encoding='utf-8'))
    for path, key in [(SNAPSHOTS, 'snapshot_sha256'), (CLEAN, 'clean_sha256'),
                      (SEASONAL_PATH, 'features_sha256')]:
        if sha256(path) != seasonal_manifest[key]:
            raise ValueError(f'Seasonal feature source changed: {path}')


def fit_predict(method: str, fit: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    if method == 'historic_180d':
        return target.monetary_value_180d.to_numpy(dtype=float) / 6
    if method == 'recent_90d':
        return target.monetary_value_90d.to_numpy(dtype=float) / 3
    if method == 'base_tweedie':
        features = BASE_FEATURES
    elif method == 'extended_tweedie_history_available':
        features = BASE_FEATURES + EXTRA_FEATURES + ['prior_year_available']
    elif method in ('extended_tweedie_previous_year',
                    'extended_tweedie_previous_year_recent_weighted',
                    'extended_two_stage_previous_year'):
        features = BASE_FEATURES + EXTRA_FEATURES + ['previous_year_30d_spend']
    else:
        features = BASE_FEATURES + EXTRA_FEATURES
    if method in ('extended_two_stage', 'extended_two_stage_previous_year'):
        classifier = lgb.LGBMClassifier(
            objective='binary', n_estimators=80, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        classifier.fit(fit[features], fit[TARGET].gt(0).astype(int))
        buyers = fit[TARGET].gt(0)
        amount_model = lgb.LGBMRegressor(
            objective='gamma', n_estimators=120, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        amount_model.fit(fit.loc[buyers, features], fit.loc[buyers, TARGET])
        return (classifier.predict_proba(target[features])[:, 1]
                * amount_model.predict(target[features]))
    objectives = {
        'base_tweedie': 'tweedie', 'extended_tweedie': 'tweedie',
        'extended_tweedie_history_available': 'tweedie',
        'extended_tweedie_previous_year': 'tweedie',
        'extended_tweedie_previous_year_recent_weighted': 'tweedie',
        'extended_poisson': 'poisson', 'extended_l2': 'regression_l2',
    }
    objective = objectives[method]
    parameters = dict(objective=objective, n_estimators=120,
                      learning_rate=.05, num_leaves=15, random_state=42,
                      n_jobs=1, verbose=-1)
    if objective == 'tweedie':
        parameters['tweedie_variance_power'] = 1.5
    model = lgb.LGBMRegressor(**parameters)
    sample_weight = None
    if method == 'extended_tweedie_previous_year_recent_weighted':
        target_date = pd.Timestamp(target.snapshot_date.iloc[0])
        age_days = (target_date - fit.snapshot_date).dt.days.to_numpy(dtype=float)
        if (age_days < 30).any():
            raise ValueError('Recent-weighted fit includes immature labels')
        sample_weight = np.power(0.5, age_days / 180)
    model.fit(fit[features], fit[TARGET], sample_weight=sample_weight)
    return np.maximum(model.predict(target[features]), 0)


def fit_rows_for_date(data: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """A training label is available only when its full 30-day window ended."""
    fit = data.loc[data.snapshot_date.add(pd.Timedelta(days=30)).le(date)].copy()
    if fit.empty or (fit.snapshot_date + pd.Timedelta(days=30) > date).any():
        raise ValueError(f'No leakage-safe training rows for {date.date()}')
    return fit


def score_with_group_bias(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    summary = evaluate(frame, scores)
    top = top20_mask(frame, scores)
    actual = frame.loc[top, TARGET].sum()
    summary['top20_group_bias'] = (float(scores[top].sum() / actual - 1)
                                   if actual > 0 else np.nan)
    return summary


def main() -> None:
    check_sources()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    temporal = pd.read_csv(FEATURE_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    development = attach_labels(
        snapshots.loc[snapshots.split.isin(['train', 'validation'])].copy(),
        labels.loc[labels.split.isin(['train', 'validation'])].copy())
    if temporal.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate temporal feature key')
    development = development.merge(temporal, on=['snapshot_date', 'customer_id'],
                                    how='left', validate='one_to_one', indicator=True)
    if not development._merge.eq('both').all() or development[EXTRA_FEATURES].isna().any().any():
        raise ValueError('Temporal features do not cover development cohort')
    development = development.drop(columns='_merge')
    seasonal = pd.read_csv(SEASONAL_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    if seasonal.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate seasonal feature key')
    development = development.merge(seasonal, on=['snapshot_date', 'customer_id'],
                                    how='left', validate='one_to_one', indicator=True)
    if not development._merge.eq('both').all():
        raise ValueError('Seasonal features do not cover development cohort')
    development = development.drop(columns='_merge')
    development['prior_year_available'] = (
        development.previous_year_30d_spend.notna().astype('int8'))

    rows, months, folds = [], [], []
    raw_history: list[pd.DataFrame] = []
    for date in DEVELOPMENT_DATES:
        fit = fit_rows_for_date(development, date)
        holdout = development.loc[development.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if holdout.empty:
            raise ValueError(f'No holdout rows: {date.date()}')
        folds.append({'target_date': date.strftime('%Y-%m-%d'),
                      'fit_count': len(fit), 'holdout_count': len(holdout),
                      'latest_fit_cutoff': fit.snapshot_date.max().strftime('%Y-%m-%d'),
                      'latest_fit_label_end': (
                          fit.snapshot_date.max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d')})
        raw_tweedie = None
        for method in METHODS:
            if method == 'extended_tweedie_lag_calibrated':
                if raw_tweedie is None:
                    raise AssertionError('Raw Tweedie forecast must be evaluated first')
                prior = (pd.concat(raw_history, ignore_index=True)
                         if raw_history else pd.DataFrame())
                if not prior.empty:
                    prior = prior.loc[
                        prior.snapshot_date.add(pd.Timedelta(days=30)).le(date)]
                scores = raw_tweedie * lagged_amount_factor(prior, date)
            else:
                scores = fit_predict(method, fit, holdout)
                if method == 'extended_tweedie':
                    raw_tweedie = scores.copy()
            if len(scores) != len(holdout) or not np.isfinite(scores).all() or (scores < 0).any():
                raise ValueError(f'Invalid {method} predictions for {date.date()}')
            metric = score_with_group_bias(holdout, scores)
            months.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                           'method': method, **metric})
            item = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
            item['method'] = method
            item['predicted_spend_30d'] = scores
            rows.append(item)
        raw_history.append(pd.DataFrame({
            'snapshot_date': holdout.snapshot_date,
            'actual_spend_30d': holdout[TARGET].to_numpy(dtype=float),
            'raw_predicted_spend_30d': raw_tweedie,
        }))
        print(f'completed development cutoff {date.date()}', flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    monthly = pd.DataFrame(months)
    totals = []
    for period, dates in [('rolling_jan_may', DEVELOPMENT_DATES[:5]),
                          ('known_jun_aug', DEVELOPMENT_DATES[5:]),
                          ('rolling_jan_aug', DEVELOPMENT_DATES)]:
        part = development.loc[development.snapshot_date.isin(dates)].copy()
        for method in METHODS:
            pred = predictions.loc[predictions.method.eq(method)
                                   & predictions.snapshot_date.isin(dates)]
            joined = part.merge(pred[['snapshot_date', 'customer_id', 'predicted_spend_30d']],
                                on=['snapshot_date', 'customer_id'], validate='one_to_one')
            if len(joined) != len(part):
                raise ValueError('Development prediction coverage failed')
            scores = joined.predicted_spend_30d.to_numpy(dtype=float)
            totals.append({'period': period, 'method': method,
                           **score_with_group_bias(joined, scores)})
    totals = pd.DataFrame(totals)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / 'development_predictions.csv', index=False,
                       encoding='utf-8-sig', date_format='%Y-%m-%d')
    monthly.to_csv(OUTPUT / 'monthly_development.csv', index=False,
                   encoding='utf-8-sig')
    totals.to_csv(OUTPUT / 'development_comparison.csv', index=False,
                  encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(OUTPUT / 'folds.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'status': 'development_only_no_final_selection',
        'protocol_sha256': sha256(ROOT / 'value_prediction/optimization_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'snapshot_sha256': sha256(SNAPSHOTS),
        'labels_sha256': sha256(LABELS),
        'temporal_features_sha256': sha256(FEATURE_PATH),
        'seasonal_features_sha256': sha256(SEASONAL_PATH),
        'development_predictions_sha256': sha256(
            OUTPUT / 'development_predictions.csv'),
        'test_rows_used': False,
        'folds': folds,
        'candidate_methods': METHODS,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    baseline = totals.loc[totals.method.eq('historic_180d')].set_index('period')
    lines = ['# 消费额预测开发期向前验证', '',
             '本报告只使用 2011 年 1–8 月开发期，9–11 月测试行未参与训练或评估。',
             '每月重新训练时，仅使用未来 30 天标签窗口已经结束的更早客户快照。',
             '1–5 月用于滚动开发比较；6–8 月是此前已看过的开发确认期。', '',
             '| 期间 | 方法 | MAE | 购买者 MAE | Top20 消费额覆盖 | 总额偏差 | Top20 群体偏差 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in totals.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | {row.top20_group_bias:+.2%} |')
    lines += ['', '与同期间历史金额基线相比：', '']
    for row in totals.loc[totals.method.ne('historic_180d')].itertuples(index=False):
        b = baseline.loc[row.period]
        lines.append(f'- {row.period} / {row.method}：MAE '
                     f'{row.mae / b.mae - 1:+.2%}；Top20 覆盖 '
                     f'{(row.top20_spend_capture - b.top20_spend_capture) * 100:+.2f} 个百分点。')
    lines += ['', '逐月结果、每条客户预测、训练截止证明和输入哈希见同目录。'
              '这里的结果仅供下一轮设计，不能作为最终独立测试。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote development-only comparison to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
