"""Audit whether two-stage spend error comes from purchase or buyer amount.

Only January-August 2011 development cohorts are evaluated. Future buyer
status is used in the post-hoc error decomposition, never as a model input.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from stage4.kmeans import FEATURES as BASE_FEATURES, load_snapshots
from value_prediction.calibration import (
    lagged_amount_factor, monotone_buyer_amount,
    segmented_buyer_amount_factors,
)
from value_prediction.experiments import (
    LABELS, TARGET, attach_labels, evaluate, top20_mask,
)
from value_prediction.labels import SNAPSHOTS, sha256
from value_prediction.optimization import (
    DEVELOPMENT_DATES, check_sources, fit_rows_for_date,
)
from value_prediction.seasonal_features import FEATURE_PATH as SEASONAL_PATH
from value_prediction.temporal_features import FEATURE_PATH, FEATURES as EXTRA_FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/hurdle_diagnostics'
FEATURES = BASE_FEATURES + EXTRA_FEATURES
SEASONAL_FEATURES = FEATURES + ['previous_year_30d_spend']
VARIANTS = {
    'base_20': ('mean_if_purchase', 'median_if_purchase'),
    'amount_plus_prior_year': ('mean_if_purchase_prior_year',
                               'median_if_purchase_prior_year'),
    'base_20_buyer_lag_calibrated': (
        'mean_if_purchase_lag_calibrated', 'median_if_purchase'),
    'prior_year_buyer_lag_calibrated': (
        'mean_if_purchase_prior_year_lag_calibrated',
        'median_if_purchase_prior_year'),
    'base_20_recent_weighted': (
        'mean_if_purchase_recent_weighted', 'median_if_purchase'),
    'base_20_isotonic_buyer_calibrated': (
        'mean_if_purchase_isotonic', 'median_if_purchase'),
    'base_20_l2_mean': ('mean_if_purchase_l2', 'median_if_purchase'),
    'base_20_segmented_buyer_lag_calibrated': (
        'mean_if_purchase_segmented', 'median_if_purchase'),
}


def high_amount_mask(frame: pd.DataFrame, mean: np.ndarray) -> np.ndarray:
    """Select exactly the top 10% by pre-outcome conditional mean."""
    scores = np.asarray(mean, dtype=float)
    if (len(scores) != len(frame) or not np.isfinite(scores).all()
            or (scores < 0).any() or frame.customer_id.isna().any()):
        raise ValueError('Invalid high-amount rank inputs')
    ranked = frame[['customer_id']].copy().reset_index(drop=True)
    ranked['score'] = scores
    ranked['row_number'] = np.arange(len(frame))
    count = int(np.ceil(.1 * len(frame)))
    selected = ranked.sort_values(
        ['score', 'customer_id'], ascending=[False, True], kind='stable'
    ).head(count).row_number.to_numpy(dtype=int)
    mask = np.zeros(len(frame), dtype=bool)
    mask[selected] = True
    return mask


def recent_amount_weights(dates: pd.Series,
                          target_date: pd.Timestamp) -> np.ndarray:
    """Give more influence to mature, recent buyer labels."""
    age_days = (pd.Timestamp(target_date)
                - pd.to_datetime(dates, errors='raise')).dt.days.to_numpy(dtype=float)
    if not np.isfinite(age_days).all() or (age_days < 30).any():
        raise ValueError('Recent amount fit includes an immature label')
    return np.power(0.5, age_days / 180)


def decompose_spend(actual: np.ndarray, probability: np.ndarray,
                    mean_if_purchase: np.ndarray,
                    median_if_purchase: np.ndarray,
                    prior_average_order: np.ndarray) -> dict:
    """Exactly split total bias into purchase-incidence and buyer-amount parts."""
    actual, probability, mean_if_purchase, median_if_purchase, prior_average_order = (
        np.asarray(value, dtype=float) for value in (
            actual, probability, mean_if_purchase, median_if_purchase,
            prior_average_order))
    if (actual.ndim != 1 or any(value.shape != actual.shape for value in (
            probability, mean_if_purchase, median_if_purchase,
            prior_average_order)) or any(not np.isfinite(value).all() for value in (
            actual, probability, mean_if_purchase, median_if_purchase,
            prior_average_order)) or (actual < 0).any()
            or (probability < 0).any() or (probability > 1).any()
            or (mean_if_purchase < 0).any() or (median_if_purchase < 0).any()
            or (prior_average_order < 0).any()):
        raise ValueError('Invalid hurdle diagnostic inputs')
    buyer = actual > 0
    if not buyer.any() or actual.sum() <= 0:
        raise ValueError('Hurdle diagnostic needs positive buyer spend')
    expected = probability * mean_if_purchase
    incidence_error = float(expected.sum() - mean_if_purchase[buyer].sum())
    amount_error = float(mean_if_purchase[buyer].sum() - actual.sum())
    total_error = float(expected.sum() - actual.sum())
    if not np.isclose(incidence_error + amount_error, total_error, atol=1e-7):
        raise AssertionError('Hurdle spend decomposition does not reconcile')
    total = float(actual.sum())
    return {
        'buyer_count': int(buyer.sum()),
        'predicted_buyer_count': float(probability.sum()),
        'buyer_count_bias': float(probability.sum() / buyer.sum() - 1),
        'buyer_rate': float(buyer.mean()),
        'average_precision': float(average_precision_score(buyer, probability)),
        'brier_score': float(brier_score_loss(buyer, probability)),
        'actual_spend': total, 'predicted_spend': float(expected.sum()),
        'aggregate_bias': total_error / total,
        'incidence_component_bias': incidence_error / total,
        'buyer_amount_component_bias': amount_error / total,
        'actual_buyer_mean_spend': total / buyer.sum(),
        'predicted_buyer_mean_if_purchase': float(mean_if_purchase[buyer].mean()),
        'conditional_median_buyer_mae': float(np.abs(
            median_if_purchase[buyer] - actual[buyer]).mean()),
        'prior_average_order_buyer_mae': float(np.abs(
            prior_average_order[buyer] - actual[buyer]).mean()),
    }


def train_predict(fit: pd.DataFrame,
                  holdout: pd.DataFrame) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    buyer = fit[TARGET].gt(0)
    if buyer.sum() < 100 or (~buyer).sum() < 100:
        raise ValueError('Too few buyer or nonbuyer training labels')
    classifier = lgb.LGBMClassifier(
        objective='binary', n_estimators=80, learning_rate=.05,
        num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
    classifier.fit(fit[FEATURES], buyer.astype(int))
    probability = classifier.predict_proba(holdout[FEATURES])[:, 1]
    amounts = {}
    for variant, features in [('base_20', FEATURES),
                              ('amount_plus_prior_year', SEASONAL_FEATURES)]:
        mean_model = lgb.LGBMRegressor(
            objective='gamma', n_estimators=120, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        mean_model.fit(fit.loc[buyer, features], fit.loc[buyer, TARGET])
        median_model = lgb.LGBMRegressor(
            objective='regression_l1', n_estimators=120, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        median_model.fit(fit.loc[buyer, features], fit.loc[buyer, TARGET])
        conditional_mean = np.maximum(mean_model.predict(holdout[features]), 0)
        conditional_median = np.maximum(median_model.predict(holdout[features]), 0)
        amounts[variant] = conditional_mean, conditional_median
    return probability, amounts


def score(frame: pd.DataFrame, variant: str) -> dict:
    actual = frame[TARGET].to_numpy(dtype=float)
    probability = frame.purchase_probability.to_numpy(dtype=float)
    mean_column, median_column = VARIANTS[variant]
    mean = frame[mean_column].to_numpy(dtype=float)
    median = frame[median_column].to_numpy(dtype=float)
    prior_order = frame.average_order_value_180d.to_numpy(dtype=float)
    diagnostic = decompose_spend(actual, probability, mean, median, prior_order)
    expected = probability * mean
    rank_and_error = evaluate(frame, expected)
    if not np.isclose(diagnostic['aggregate_bias'],
                      rank_and_error['aggregate_bias'], atol=1e-9):
        raise AssertionError('Hurdle evaluation amount mismatch')
    return {**rank_and_error, **diagnostic}


def top_buyer_tail(frame: pd.DataFrame) -> dict:
    """Audit whether the highest actual buyers were ranked and sized well."""
    frame = frame.reset_index(drop=True)
    buyer = frame.loc[frame[TARGET].gt(0)]
    if buyer.empty:
        raise ValueError('No buyers for high-spend tail audit')
    count = max(1, int(np.ceil(len(buyer) * .01)))
    tail = buyer.nlargest(count, TARGET)
    expected = frame.expected_spend_30d.to_numpy(dtype=float)
    top = top20_mask(frame, expected)
    actual_tail = float(tail[TARGET].sum())
    predicted_tail = float(frame.loc[tail.index, 'expected_spend_30d'].sum())
    return {
        'buyer_count': len(buyer), 'top_one_percent_buyer_count': count,
        'top_one_percent_actual_spend': actual_tail,
        'top_one_percent_actual_spend_share': float(actual_tail / frame[TARGET].sum()),
        'top_one_percent_in_predicted_top20': int(top[tail.index].sum()),
        'top_one_percent_predicted_spend': predicted_tail,
        'top_one_percent_predicted_to_actual': predicted_tail / actual_tail,
    }


def main() -> None:
    check_sources()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    snapshots = snapshots.loc[snapshots.split.isin(['train', 'validation'])].copy()
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    labels = labels.loc[labels.split.isin(['train', 'validation'])].copy()
    temporal = pd.read_csv(FEATURE_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    data = attach_labels(snapshots, labels).merge(
        temporal, on=['snapshot_date', 'customer_id'], how='left',
        validate='one_to_one', indicator=True)
    if (not data._merge.eq('both').all() or data[EXTRA_FEATURES].isna().any().any()):
        raise ValueError('Missing hurdle diagnostic feature rows')
    data = data.drop(columns='_merge')
    seasonal = pd.read_csv(SEASONAL_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    if seasonal.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate seasonal feature keys')
    data = data.merge(seasonal, on=['snapshot_date', 'customer_id'], how='left',
                      validate='one_to_one', indicator=True)
    if not data._merge.eq('both').all():
        raise ValueError('Missing seasonal cohort rows')
    data = data.drop(columns='_merge')

    predictions, monthly, folds = [], [], []
    raw_buyer_history: dict[str, list[pd.DataFrame]] = {
        'base_20': [], 'amount_plus_prior_year': [],
    }
    for date in DEVELOPMENT_DATES:
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if holdout.empty:
            raise ValueError(f'No hurdle diagnostic cohort at {date.date()}')
        probability, amounts = train_predict(fit, holdout)
        holdout['purchase_probability'] = probability
        holdout['high_amount_segment'] = high_amount_mask(
            holdout, amounts['base_20'][0])
        fold = {
            'target_date': date.strftime('%Y-%m-%d'), 'fit_count': len(fit),
            'holdout_count': len(holdout),
            'latest_fit_label_end': (
                fit.snapshot_date.max() + pd.Timedelta(days=30)
            ).strftime('%Y-%m-%d'),
        }
        for variant, (mean, median) in amounts.items():
            mean_column, median_column = VARIANTS[variant]
            holdout[mean_column] = mean
            holdout[median_column] = median
            monthly.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                            'variant': variant, **score(holdout, variant)})
            history = (pd.concat(raw_buyer_history[variant], ignore_index=True)
                       if raw_buyer_history[variant] else pd.DataFrame())
            if not history.empty:
                history = history.loc[
                    history.snapshot_date.add(pd.Timedelta(days=30)).le(date)]
            factor = lagged_amount_factor(history, date)
            calibrated_variant = ('base_20_buyer_lag_calibrated'
                                  if variant == 'base_20'
                                  else 'prior_year_buyer_lag_calibrated')
            calibrated_mean_column, _ = VARIANTS[calibrated_variant]
            holdout[calibrated_mean_column] = mean * factor
            monthly.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                            'variant': calibrated_variant,
                            **score(holdout, calibrated_variant)})
            if variant == 'base_20':
                low_factor, high_factor = segmented_buyer_amount_factors(
                    history, date)
                holdout['mean_if_purchase_segmented'] = mean * np.where(
                    holdout.high_amount_segment.to_numpy(dtype=bool),
                    high_factor, low_factor)
                monthly.append({
                    'snapshot_date': date.strftime('%Y-%m-%d'),
                    'variant': 'base_20_segmented_buyer_lag_calibrated',
                    **score(holdout,
                            'base_20_segmented_buyer_lag_calibrated'),
                })
                fold['ordinary_buyer_amount_factor'] = low_factor
                fold['high_buyer_amount_factor'] = high_factor
                holdout['mean_if_purchase_isotonic'] = monotone_buyer_amount(
                    history, date, mean)
                monthly.append({
                    'snapshot_date': date.strftime('%Y-%m-%d'),
                    'variant': 'base_20_isotonic_buyer_calibrated',
                    **score(holdout, 'base_20_isotonic_buyer_calibrated'),
                })
            fold[f'{variant}_buyer_amount_factor'] = factor
            buyer_rows = holdout.loc[holdout[TARGET].gt(0)]
            past_buyers = pd.DataFrame({
                'snapshot_date': buyer_rows.snapshot_date,
                'actual_spend_30d': buyer_rows[TARGET].to_numpy(dtype=float),
                'raw_predicted_spend_30d': mean[holdout[TARGET].gt(0).to_numpy()],
            })
            if variant == 'base_20':
                past_buyers['high_amount_segment'] = (
                    buyer_rows.high_amount_segment.to_numpy(dtype=bool))
            raw_buyer_history[variant].append(past_buyers)
        buyer = fit[TARGET].gt(0)
        recent_model = lgb.LGBMRegressor(
            objective='gamma', n_estimators=120, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        recent_model.fit(
            fit.loc[buyer, FEATURES], fit.loc[buyer, TARGET],
            sample_weight=recent_amount_weights(
                fit.loc[buyer, 'snapshot_date'], date))
        holdout['mean_if_purchase_recent_weighted'] = np.maximum(
            recent_model.predict(holdout[FEATURES]), 0)
        monthly.append({
            'snapshot_date': date.strftime('%Y-%m-%d'),
            'variant': 'base_20_recent_weighted',
            **score(holdout, 'base_20_recent_weighted'),
        })
        l2_model = lgb.LGBMRegressor(
            objective='regression_l2', n_estimators=120, learning_rate=.05,
            num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
        l2_model.fit(fit.loc[buyer, FEATURES], fit.loc[buyer, TARGET])
        holdout['mean_if_purchase_l2'] = np.maximum(
            l2_model.predict(holdout[FEATURES]), 0)
        monthly.append({
            'snapshot_date': date.strftime('%Y-%m-%d'),
            'variant': 'base_20_l2_mean',
            **score(holdout, 'base_20_l2_mean'),
        })
        holdout['expected_spend_30d'] = (
            probability * holdout['mean_if_purchase'])
        holdout['expected_spend_prior_year'] = (
            probability * holdout['mean_if_purchase_prior_year'])
        holdout['expected_spend_lag_calibrated'] = (
            probability * holdout['mean_if_purchase_lag_calibrated'])
        holdout['expected_spend_prior_year_lag_calibrated'] = (
            probability * holdout['mean_if_purchase_prior_year_lag_calibrated'])
        holdout['expected_spend_recent_weighted'] = (
            probability * holdout['mean_if_purchase_recent_weighted'])
        holdout['expected_spend_isotonic'] = (
            probability * holdout['mean_if_purchase_isotonic'])
        holdout['expected_spend_l2_mean'] = (
            probability * holdout['mean_if_purchase_l2'])
        holdout['expected_spend_segmented'] = (
            probability * holdout['mean_if_purchase_segmented'])
        predictions.append(holdout[[
            'snapshot_date', 'customer_id', TARGET, 'average_order_value_180d',
            'purchase_probability', 'mean_if_purchase', 'median_if_purchase',
            'expected_spend_30d', 'mean_if_purchase_prior_year',
            'median_if_purchase_prior_year', 'expected_spend_prior_year',
            'mean_if_purchase_lag_calibrated',
            'mean_if_purchase_prior_year_lag_calibrated',
            'expected_spend_lag_calibrated',
            'expected_spend_prior_year_lag_calibrated',
            'mean_if_purchase_recent_weighted',
            'expected_spend_recent_weighted',
            'mean_if_purchase_isotonic', 'expected_spend_isotonic',
            'mean_if_purchase_l2', 'expected_spend_l2_mean',
            'high_amount_segment', 'mean_if_purchase_segmented',
            'expected_spend_segmented',
        ]].copy())
        folds.append(fold)
        print(f'completed hurdle diagnostic {date.date()}', flush=True)

    predictions = pd.concat(predictions, ignore_index=True)
    tail = pd.DataFrame([
        {'snapshot_date': date.strftime('%Y-%m-%d'),
         **top_buyer_tail(predictions.loc[
             predictions.snapshot_date.eq(date)].copy())}
        for date in DEVELOPMENT_DATES
    ])
    pooled = pd.DataFrame([
        {'period': name, 'variant': variant,
         **score(predictions.loc[
             predictions.snapshot_date.isin(dates)].copy().reset_index(drop=True),
             variant)}
        for name, dates in (
            ('rolling_jan_may', DEVELOPMENT_DATES[:5]),
            ('known_jun_aug', DEVELOPMENT_DATES[5:]),
            ('rolling_jan_aug', DEVELOPMENT_DATES),
        )
        for variant in VARIANTS
    ])
    monthly = pd.DataFrame(monthly)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / 'customer_predictions.csv', index=False,
                       encoding='utf-8-sig', date_format='%Y-%m-%d')
    monthly.to_csv(OUTPUT / 'monthly_diagnostics.csv', index=False,
                   encoding='utf-8-sig')
    tail.to_csv(OUTPUT / 'top_buyer_tail.csv', index=False,
                encoding='utf-8-sig')
    pooled.to_csv(OUTPUT / 'pooled_diagnostics.csv', index=False,
                  encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(OUTPUT / 'folds.csv', index=False,
                               encoding='utf-8-sig')
    manifest = {
        'status': 'development_only_no_independent_acceptance',
        'protocol_sha256': sha256(ROOT / 'value_prediction/optimization_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'snapshot_sha256': sha256(SNAPSHOTS),
        'labels_sha256': sha256(LABELS),
        'temporal_features_sha256': sha256(FEATURE_PATH),
        'seasonal_features_sha256': sha256(SEASONAL_PATH),
        'predictions_sha256': sha256(OUTPUT / 'customer_predictions.csv'),
        'top_buyer_tail_sha256': sha256(OUTPUT / 'top_buyer_tail.csv'),
        'test_rows_used': False, 'folds': folds,
        'probability_features': FEATURES,
        'amount_feature_sets': {
            'base_20': FEATURES,
            'amount_plus_prior_year': SEASONAL_FEATURES,
            'base_20_recent_weighted': FEATURES,
            'base_20_isotonic_buyer_calibrated': FEATURES,
            'base_20_l2_mean': FEATURES,
            'base_20_segmented_buyer_lag_calibrated': FEATURES,
        },
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')
    lines = [
        '# 两阶段金额模型误差来源审计', '',
        '本报告使用已有两阶段模型的固定参数；2011 年 1–8 月按时间向前评估，'
        '9–11 月行未参与。',
        '购买概率、购买条件均值及中位数都只由截点前特征预测。'
        '实际购买状态仅在事后评估时使用。', '',
        '`预测总额−实际总额 = [Σ(p×m)−Σ(I×m)] + [Σ(I×m)−Σ实际金额]`。'
        '前一项为购买发生部分，后一项为购买者金额部分；两项相加精确等于总额误差。',
        '', '| 期间 | 金额特征 | AP | 预测购买人数偏差 | 购买发生部分 | 购买者金额部分 | 总额偏差 | 购买者中位数 MAE / 历史单均价 MAE |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in pooled.itertuples(index=False):
        lines.append(f'| {row.period} | {row.variant} | '
                     f'{row.average_precision:.4f} | '
                     f'{row.buyer_count_bias:+.2%} | '
                     f'{row.incidence_component_bias:+.2%} | '
                     f'{row.buyer_amount_component_bias:+.2%} | '
                     f'{row.aggregate_bias:+.2%} | '
                     f'{row.conditional_median_buyer_mae:.2f} / '
                     f'{row.prior_average_order_buyer_mae:.2f} |')
    lines += ['', '## 月度诊断', '',
              '| 截点 | 金额特征 | 实际 / 预测购买数 | AP | 购买发生部分 | 购买者金额部分 | 总额偏差 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in monthly.itertuples(index=False):
        lines.append(f'| {row.snapshot_date} | {row.variant} | '
                     f'{row.buyer_count} / '
                     f'{row.predicted_buyer_count:.0f} | '
                     f'{row.average_precision:.4f} | '
                     f'{row.incidence_component_bias:+.2%} | '
                     f'{row.buyer_amount_component_bias:+.2%} | '
                     f'{row.aggregate_bias:+.2%} |')
    lines += ['', '## 高额购买者尾部审计（原两阶段模型）', '',
              '每月在实际购买者中取消费额最高的约 1%；实际金额只用于事后审计。',
              '', '| 截点 | 尾部人数 | 占当月实际金额 | 进入预测 Top20 | 尾部预测金额 / 实际金额 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in tail.itertuples(index=False):
        lines.append(f'| {row.snapshot_date} | {row.top_one_percent_buyer_count} | '
                     f'{row.top_one_percent_actual_spend_share:.2%} | '
                     f'{row.top_one_percent_in_predicted_top20} / '
                     f'{row.top_one_percent_buyer_count} | '
                     f'{row.top_one_percent_predicted_to_actual:.2%} |')
    lines += ['', '该分解用于定位误差来源，不等同于任何强化目标已经通过。'
              '原目标仍按 `optimization_protocol.md` 与未来可信数据判断。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote hurdle diagnostic to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
