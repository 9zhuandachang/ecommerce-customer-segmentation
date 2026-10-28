"""Fixed buyer-weighted Tweedie comparison on previously viewed development dates."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from stage4.kmeans import FEATURES as BASE_FEATURES, load_snapshots
from value_prediction.experiments import LABELS, TARGET, attach_labels
from value_prediction.labels import SNAPSHOTS, sha256
from value_prediction.optimization import (
    check_sources, fit_rows_for_date, score_with_group_bias,
)
from value_prediction.seasonal_features import FEATURE_PATH as SEASONAL_PATH
from value_prediction.temporal_features import FEATURE_PATH, FEATURES as EXTRA_FEATURES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUT = HERE / 'outputs/buyer_weight'
PROTOCOL = HERE / 'buyer_weight_protocol.md'
REFERENCE = HERE / 'outputs/optimization/development_predictions.csv'
REFERENCE_MANIFEST = REFERENCE.parent / 'manifest.json'
DATES = pd.date_range('2011-01-01', '2011-08-01', freq='MS')
METHODS = ('historic_180d', 'extended_tweedie', 'buyer_weight_2x_tweedie')
FEATURES = BASE_FEATURES + EXTRA_FEATURES


def buyer_weights(amounts: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(amounts, dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Buyer weights require nonnegative mature labels')
    return np.where(values > 0, 2.0, 1.0)


def fit_predict_buyer_weighted(fit: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    if fit.empty or target.empty or not fit.snapshot_date.lt(target.snapshot_date.min()).all():
        raise ValueError('Training must precede the forecast month')
    if (fit.snapshot_date + pd.Timedelta(days=30)
            > target.snapshot_date.min()).any():
        raise ValueError('Training labels are not fully mature')
    model = lgb.LGBMRegressor(
        objective='tweedie', n_estimators=120, learning_rate=.05,
        num_leaves=15, tweedie_variance_power=1.5,
        random_state=42, n_jobs=1, verbose=-1)
    model.fit(fit[FEATURES], fit[TARGET], sample_weight=buyer_weights(fit[TARGET]))
    scores = np.maximum(model.predict(target[FEATURES]), 0)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite weighted Tweedie predictions')
    return scores


def load_development() -> pd.DataFrame:
    check_sources()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    cohort = attach_labels(
        snapshots.loc[snapshots.split.isin(['train', 'validation'])].copy(),
        labels.loc[labels.split.isin(['train', 'validation'])].copy())
    for path in (FEATURE_PATH, SEASONAL_PATH):
        feature = pd.read_csv(path, dtype={'customer_id': 'string'},
                              parse_dates=['snapshot_date'])
        if feature.duplicated(['snapshot_date', 'customer_id']).any():
            raise ValueError(f'Duplicate feature key in {path}')
        cohort = cohort.merge(feature, on=['snapshot_date', 'customer_id'],
                              how='left', validate='one_to_one', indicator=True)
        if not cohort._merge.eq('both').all():
            raise ValueError(f'Feature source does not cover cohort: {path}')
        cohort = cohort.drop(columns='_merge')
    if cohort[FEATURES].isna().any().any():
        raise ValueError('Missing 20-feature training values')
    return cohort


def load_reference() -> pd.DataFrame:
    manifest = json.loads(REFERENCE_MANIFEST.read_text(encoding='utf-8'))
    if (manifest.get('development_predictions_sha256') != sha256(REFERENCE)
            or manifest.get('snapshot_sha256') != sha256(SNAPSHOTS)
            or manifest.get('labels_sha256') != sha256(LABELS)
            or manifest.get('test_rows_used') is not False):
        raise ValueError('Original development predictions changed')
    reference = pd.read_csv(REFERENCE, dtype={'customer_id': 'string'},
                            parse_dates=['snapshot_date'])
    reference = reference.loc[reference.method.isin(METHODS[:2])].copy()
    if (reference.empty
            or reference.duplicated(['snapshot_date', 'customer_id', 'method']).any()
            or set(reference.snapshot_date) != set(DATES)):
        raise ValueError('Original development reference incomplete')
    return reference


def reference_for_holdout(reference: pd.DataFrame,
                          holdout: pd.DataFrame) -> pd.DataFrame:
    date = pd.Timestamp(holdout.snapshot_date.iloc[0])
    part = reference.loc[reference.snapshot_date.eq(date)].copy()
    if part.groupby(['snapshot_date', 'customer_id']).method.nunique().ne(2).any():
        raise ValueError('Reference methods do not share a customer cohort')
    wide = part.pivot(index=['snapshot_date', 'customer_id', TARGET],
                      columns='method', values='predicted_spend_30d').reset_index()
    wide.columns.name = None
    merged = holdout[['snapshot_date', 'customer_id', TARGET]].merge(
        wide, on=['snapshot_date', 'customer_id', TARGET],
        how='left', validate='one_to_one', indicator=True)
    if (len(merged) != len(holdout) or not merged._merge.eq('both').all()
            or merged[list(METHODS[:2])].isna().any().any()):
        raise ValueError('Reference and current cohort or labels differ')
    return merged.drop(columns='_merge')


def advancement_checks(summary: pd.DataFrame) -> dict:
    period = summary.loc[summary.period.eq('rolling_jan_may')].set_index('method')
    raw = period.loc['extended_tweedie']
    weighted = period.loc['buyer_weight_2x_tweedie']
    checks = {
        'buyer_mae_reduction_at_least_5pct':
        weighted.positive_mae <= .95 * raw.positive_mae,
        'overall_mae_increase_at_most_2pct': weighted.mae <= 1.02 * raw.mae,
        'top20_capture_not_lower':
        weighted.top20_spend_capture >= raw.top20_spend_capture,
        'aggregate_bias_abs_not_larger':
        abs(weighted.aggregate_bias) <= abs(raw.aggregate_bias),
    }
    return {key: bool(value) for key, value in checks.items()}


def make_report(summary: pd.DataFrame, checks: dict,
                predictions: pd.DataFrame) -> str:
    lines = [
        '# 已查看开发期：购买者双倍训练权重的金额模型', '',
        '事前方案见 `buyer_weight_protocol.md`。只给已成熟、未来 30 天确实购买的历史训练行 2 倍权重；当前预测时不知道未来购买状态。',
        '这不是新的独立测试，也不改变原强化目标。', '',
        '| 期间 | 方法 | 全体 MAE | 购买者 MAE | Top20 消费覆盖 | 总额偏差 | Top20 群体偏差 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | {row.top20_group_bias:+.2%} |')
    lines += ['', '## 运行前锁定的开发期推进门槛', '']
    for key, passed in checks.items():
        lines.append(f'- `{key}`：{"满足" if passed else "未满足"}')
    lines += ['', '## 事后误差分解（解释取舍，不用于调权重）', '',
              '| 期间 | 方法 | 零消费客户占比 | 零消费客户 MAE | 购买者 MAE |',
              '| --- | --- | ---: | ---: | ---: |']
    for period, dates in [('rolling_jan_may', DATES[:5]),
                          ('known_jun_aug', DATES[5:])]:
        for method in METHODS:
            part = predictions.loc[predictions.snapshot_date.isin(dates)
                                   & predictions.method.eq(method)]
            zero = part.loc[part[TARGET].eq(0)]
            buyer = part.loc[part[TARGET].gt(0)]
            lines.append(f'| {period} | {method} | {len(zero) / len(part):.2%} | '
                         f'{zero.predicted_spend_30d.mean():.2f} | '
                         f'{np.abs(buyer.predicted_spend_30d - buyer[TARGET]).mean():.2f} |')
    lines += ['', '零消费客户的实际金额为 0，因此该组 MAE 就是平均预测金额。'
              '双倍权重减少了购买者误差，却显著抬高零消费客户预测，说明两项目标目前存在冲突。',
              '只有运行前四项推进条件全部满足才建议进入新数据的固定测试；'
              '任何结果均不能作为原商家的独立验收。',
              '逐月指标、客户级预测、训练标签截止和源哈希见同目录。', '']
    return '\n'.join(lines)


def main() -> None:
    data = load_development()
    reference = load_reference()
    rows, months, folds = [], [], []
    for date in DATES:
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if holdout.empty:
            raise ValueError(f'No holdout cohort on {date.date()}')
        same_cohort = reference_for_holdout(reference, holdout)
        weighted_scores = fit_predict_buyer_weighted(fit, holdout)
        period = 'rolling_jan_may' if date <= DATES[4] else 'known_jun_aug'
        folds.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                      'fit_count': len(fit), 'holdout_count': len(holdout),
                      'fit_buyers': int(fit[TARGET].gt(0).sum()),
                      'latest_fit_label_end': (
                          fit.snapshot_date.max() + pd.Timedelta(days=30)
                      ).strftime('%Y-%m-%d')})
        for method in METHODS:
            scores = (weighted_scores if method == METHODS[2]
                      else same_cohort[method].to_numpy(dtype=float))
            months.append({'period': period, 'snapshot_date': date.strftime('%Y-%m-%d'),
                           'method': method, 'zero_spend_share':
                           float(holdout[TARGET].eq(0).mean()),
                           **score_with_group_bias(holdout, scores)})
            part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
            part['method'] = method
            part['predicted_spend_30d'] = scores
            rows.append(part)
        print(f'completed buyer-weighted cutoff {date.date()}', flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    period_rows = []
    for period, dates in [('rolling_jan_may', DATES[:5]),
                          ('known_jun_aug', DATES[5:])]:
        for method in METHODS:
            subset = predictions.loc[predictions.snapshot_date.isin(dates)
                                     & predictions.method.eq(method)]
            scores = subset.predicted_spend_30d.to_numpy(dtype=float)
            period_rows.append({'period': period, 'method': method,
                                'zero_spend_share': float(subset[TARGET].eq(0).mean()),
                                **score_with_group_bias(subset, scores)})
    summary = pd.DataFrame(period_rows)
    checks = advancement_checks(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {'customer_predictions': OUTPUT / 'customer_predictions.csv',
             'monthly_metrics': OUTPUT / 'monthly_metrics.csv',
             'period_metrics': OUTPUT / 'period_metrics.csv',
             'folds': OUTPUT / 'folds.csv'}
    for name, frame in [('customer_predictions', predictions),
                        ('monthly_metrics', pd.DataFrame(months)),
                        ('period_metrics', summary), ('folds', pd.DataFrame(folds))]:
        frame.to_csv(paths[name], index=False, encoding='utf-8-sig',
                     date_format='%Y-%m-%d')
    report = OUTPUT / 'report.md'
    report.write_text(make_report(summary, checks, predictions), encoding='utf-8')
    manifest = {
        'status': 'development_only_not_independent_acceptance',
        'candidate': METHODS[2], 'buyer_training_weight': 2.0,
        'comparison_methods': METHODS[:2], 'development_gate': checks,
        'gate_passed': all(checks.values()),
        'source_snapshot_sha256': sha256(SNAPSHOTS),
        'source_labels_sha256': sha256(LABELS),
        'source_temporal_features_sha256': sha256(FEATURE_PATH),
        'source_seasonal_features_sha256': sha256(SEASONAL_PATH),
        'reference_predictions_sha256': sha256(REFERENCE),
        'protocol_sha256': sha256(PROTOCOL), 'script_sha256': sha256(Path(__file__)),
        'outputs_sha256': {name: sha256(path) for name, path in paths.items()},
        'report_sha256': sha256(report),
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                    indent=2) + '\n', encoding='utf-8')
    print(f'wrote {report}', flush=True)


if __name__ == '__main__':
    main()
