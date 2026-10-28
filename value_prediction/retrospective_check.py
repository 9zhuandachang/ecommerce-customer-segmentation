"""One fixed-method, forward-in-time check on previously viewed test months.

This period is not an untouched holdout. Never use these results to change
features, model settings, or acceptance thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage4.kmeans import load_snapshots
from value_prediction.calibration import lagged_amount_factor
from value_prediction.experiments import LABELS, TARGET, attach_labels
from value_prediction.labels import SNAPSHOTS, sha256
from value_prediction.optimization import (
    check_sources, fit_predict, fit_rows_for_date, score_with_group_bias,
)
from value_prediction.temporal_features import FEATURE_PATH, FEATURES as EXTRA_FEATURES
from value_prediction.seasonal_features import FEATURE_PATH as SEASONAL_PATH

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/retrospective_check'
METHODS = ('historic_180d', 'extended_tweedie',
           'extended_tweedie_lag_calibrated',
           'extended_tweedie_history_available',
           'extended_tweedie_previous_year')


def main() -> None:
    check_sources()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    temporal = pd.read_csv(FEATURE_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    data = attach_labels(snapshots, labels).merge(
        temporal, on=['snapshot_date', 'customer_id'], how='left',
        validate='one_to_one', indicator=True)
    if (not data._merge.eq('both').all()
            or data[EXTRA_FEATURES].isna().any().any()):
        raise ValueError('Temporal features do not cover retrospective cohort')
    data = data.drop(columns='_merge')
    seasonal = pd.read_csv(SEASONAL_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    data = data.merge(seasonal, on=['snapshot_date', 'customer_id'],
                      how='left', validate='one_to_one', indicator=True)
    if not data._merge.eq('both').all():
        raise ValueError('Seasonal features do not cover retrospective cohort')
    data = data.drop(columns='_merge')
    data['prior_year_available'] = (
        data.previous_year_30d_spend.notna().astype('int8'))
    test_dates = sorted(data.loc[data.split.eq('test'), 'snapshot_date'].unique())
    if test_dates != list(pd.date_range('2011-09-01', '2011-11-01', freq='MS')):
        raise ValueError('Unexpected retrospective dates')
    development_prediction_path = (ROOT / 'value_prediction/outputs/optimization'
                                   / 'development_predictions.csv')
    development_manifest = json.loads((development_prediction_path.parent
                                       / 'manifest.json').read_text(encoding='utf-8'))
    if (development_manifest.get('development_predictions_sha256')
            != sha256(development_prediction_path)
            or development_manifest.get('temporal_features_sha256')
            != sha256(FEATURE_PATH)
            or development_manifest.get('seasonal_features_sha256')
            != sha256(SEASONAL_PATH)
            or development_manifest.get('snapshot_sha256') != sha256(SNAPSHOTS)
            or development_manifest.get('labels_sha256') != sha256(LABELS)):
        raise ValueError('Development forecasts or sources changed')
    development = pd.read_csv(development_prediction_path,
                              parse_dates=['snapshot_date'])
    prior = development.loc[development.method.eq('extended_tweedie'),
                            ['snapshot_date', 'customer_id', TARGET,
                             'predicted_spend_30d']].copy()
    prior = prior.rename(columns={TARGET: 'actual_spend_30d',
                                  'predicted_spend_30d': 'raw_predicted_spend_30d'})
    if (prior.empty or prior.snapshot_date.max() != pd.Timestamp('2011-08-01')
            or prior.snapshot_date.min() != pd.Timestamp('2011-01-01')
            or prior.duplicated(['snapshot_date', 'customer_id']).any()):
        raise ValueError('Expected complete prior raw development forecasts')
    rows, monthly, fits = [], [], []
    for date in test_dates:
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if not fit.snapshot_date.lt(date).all():
            raise ValueError('Current or future snapshots entered training')
        fits.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                     'fit_count': len(fit),
                     'latest_fit_label_end': (
                         fit.snapshot_date.max() + pd.Timedelta(days=30)
                     ).strftime('%Y-%m-%d')})
        raw_tweedie = None
        for method in METHODS:
            if method == 'extended_tweedie_lag_calibrated':
                history = prior.loc[prior.snapshot_date.add(pd.Timedelta(days=30))
                                    .le(date)]
                score = raw_tweedie * lagged_amount_factor(history, date)
            else:
                score = fit_predict(method, fit, holdout)
                if method == 'extended_tweedie':
                    raw_tweedie = score.copy()
            if not np.isfinite(score).all() or (score < 0).any():
                raise ValueError(f'Invalid retrospective score: {method}')
            monthly.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                            'method': method,
                            **score_with_group_bias(holdout, score)})
            part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
            part['method'] = method
            part['predicted_spend_30d'] = score
            rows.append(part)
        prior = pd.concat([prior, pd.DataFrame({
            'snapshot_date': holdout.snapshot_date,
            'customer_id': holdout.customer_id,
            'actual_spend_30d': holdout[TARGET].to_numpy(dtype=float),
            'raw_predicted_spend_30d': raw_tweedie,
        })], ignore_index=True)
        print(f'completed retrospective cutoff {date.date()}', flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    metrics = []
    for method in METHODS:
        part = predictions.loc[predictions.method.eq(method)].copy()
        metrics.append({'method': method, **score_with_group_bias(
            part, part.predicted_spend_30d.to_numpy(dtype=float))})
    overall = pd.DataFrame(metrics)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / 'customer_predictions.csv', index=False,
                       encoding='utf-8-sig', date_format='%Y-%m-%d')
    pd.DataFrame(monthly).to_csv(OUTPUT / 'monthly_metrics.csv', index=False,
                                 encoding='utf-8-sig')
    overall.to_csv(OUTPUT / 'overall_metrics.csv', index=False, encoding='utf-8-sig')
    (OUTPUT / 'manifest.json').write_text(json.dumps({
        'status': 'retrospective_previously_viewed_not_independent',
        'methods_fixed_before_this_run': METHODS,
        'script_sha256': sha256(Path(__file__)),
        'snapshot_sha256': sha256(SNAPSHOTS),
        'labels_sha256': sha256(LABELS),
        'temporal_features_sha256': sha256(FEATURE_PATH),
        'seasonal_features_sha256': sha256(SEASONAL_PATH),
        'prior_development_predictions_sha256': sha256(development_prediction_path),
        'fit_cutoffs': fits,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# 2011 年 9–11 月固定方法回顾检查', '',
        '这三个月此前已经被查看，结果不能被称为全新独立测试，也不用于再次调参。',
        '滞后校准公式在原项目 1–8 月开发期固定；这里仅按同一公式回看。',
        '去年同期金额和“历史是否完整”也仅依开发期确定，在这里作回顾诊断。',
        '每月只用完整标签已结束的过去快照训练；9 月标签可以在 10 月训练中使用。', '',
        '| 期间 | 方法 | MAE | 购买者 MAE | Top20 覆盖 | 总额偏差 |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in pd.DataFrame(monthly).itertuples(index=False):
        lines.append(f'| {row.snapshot_date} | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} |')
    for row in overall.itertuples(index=False):
        lines.append(f'| 合计 | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} |')
    lines += ['', '客户级预测、月度指标和训练截止审计见同目录。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote retrospective report to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
