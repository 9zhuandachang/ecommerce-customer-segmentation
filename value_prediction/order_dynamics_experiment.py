"""Forward-only conditional amount experiment with invoice-value dynamics."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from stage4.kmeans import FEATURES as BASE_FEATURES, load_snapshots
from value_prediction.conditional_amount import conditional_metrics
from value_prediction.experiments import LABELS, TARGET, attach_labels
from value_prediction.labels import CLEAN, SCHEDULE, SNAPSHOTS, sha256
from value_prediction.optimization import (
    DEVELOPMENT_DATES, check_sources, fit_rows_for_date,
)
from value_prediction.order_dynamics_features import (
    FEATURE_PATH as DYNAMICS_PATH, FEATURES as DYNAMICS_FEATURES,
    OUTPUT as FEATURE_OUTPUT,
)
from value_prediction.temporal_features import (
    FEATURE_PATH as TEMPORAL_PATH, FEATURES as TEMPORAL_FEATURES,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'outputs/order_dynamics_experiment'
PROTOCOL = HERE / 'order_dynamics_protocol.md'
CONDITIONAL = HERE / 'outputs/conditional_amount'
METHODS = ('historic_monthly_rate', 'average_prior_invoice',
           'conditional_l1_median', 'conditional_l1_order_dynamics')
KEYS = ['snapshot_date', 'customer_id']
FEATURES = BASE_FEATURES + TEMPORAL_FEATURES + DYNAMICS_FEATURES


def _check_inputs() -> Path:
    check_sources()
    feature_manifest = json.loads((FEATURE_OUTPUT / 'feature_manifest.json')
                                  .read_text(encoding='utf-8'))
    for path, key in [(SNAPSHOTS, 'snapshot_sha256'), (CLEAN, 'clean_sha256'),
                      (SCHEDULE, 'schedule_sha256'),
                      (DYNAMICS_PATH, 'features_sha256')]:
        if sha256(path) != feature_manifest[key]:
            raise ValueError(f'Order dynamics feature source changed: {path}')
    if (feature_manifest['protocol_sha256'] != sha256(PROTOCOL)
            or feature_manifest['script_sha256'] != sha256(
                HERE / 'order_dynamics_features.py')
            or not feature_manifest['all_amounts_and_counts_reconciled']):
        raise ValueError('Order dynamics feature audit changed')
    original_manifest = json.loads((CONDITIONAL / 'manifest.json')
                                   .read_text(encoding='utf-8'))
    if (original_manifest['test_rows_used'] is not False
            or original_manifest['script_sha256'] != sha256(
                HERE / 'conditional_amount.py')
            or original_manifest['protocol_sha256'] != sha256(
                HERE / 'optimization_protocol.md')
            or original_manifest['snapshot_sha256'] != sha256(SNAPSHOTS)
            or original_manifest['labels_sha256'] != sha256(LABELS)):
        raise ValueError('Original conditional forecast source changed')
    return CONDITIONAL / 'development_predictions.csv'


def load_development_data() -> pd.DataFrame:
    _check_inputs()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    snapshots = snapshots.loc[snapshots.split.isin(['train', 'validation'])].copy()
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    labels = labels.loc[labels.split.isin(['train', 'validation'])].copy()
    temporal = pd.read_csv(TEMPORAL_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    dynamics = pd.read_csv(DYNAMICS_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    data = attach_labels(snapshots, labels)
    for source, columns in [(temporal, TEMPORAL_FEATURES),
                            (dynamics, DYNAMICS_FEATURES)]:
        data = data.merge(source[KEYS + columns], on=KEYS, how='left',
                          validate='one_to_one', indicator=True)
        if (not data._merge.eq('both').all()
                or data[columns].isna().any().any()):
            raise ValueError('Missing pre-cutoff amount features')
        data = data.drop(columns='_merge')
    if (len(data) != len(snapshots)
            or data.snapshot_date.max() > DEVELOPMENT_DATES[-1]):
        raise ValueError('Development cohort includes unexpected test rows')
    return data


def fit_predict_dynamics(fit: pd.DataFrame,
                         holdout: pd.DataFrame) -> np.ndarray:
    buyers = fit[TARGET].gt(0)
    if buyers.sum() < 100:
        raise ValueError('Too few mature buyer labels')
    model = lgb.LGBMRegressor(
        objective='regression_l1', n_estimators=120, learning_rate=.05,
        num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
    model.fit(fit.loc[buyers, FEATURES], fit.loc[buyers, TARGET])
    return np.maximum(model.predict(holdout[FEATURES]), 0)


def advancement_checks(summary: pd.DataFrame,
                       monthly: pd.DataFrame) -> dict:
    period = summary.loc[summary.period.eq('rolling_jan_may')].set_index('method')
    old = period.loc['conditional_l1_median']
    new = period.loc['conditional_l1_order_dynamics']
    early = monthly.loc[monthly.snapshot_date.isin(
        DEVELOPMENT_DATES[:5].strftime('%Y-%m-%d'))]
    pivot = early.pivot(index='snapshot_date', columns='method', values='buyer_mae')
    return {
        'buyer_mae_reduction_at_least_5pct': bool(new.buyer_mae <= .95 * old.buyer_mae),
        'at_least_four_of_five_months_not_worse': bool((
            pivot['conditional_l1_order_dynamics']
            <= pivot['conditional_l1_median']).sum() >= 4),
        'absolute_buyer_amount_bias_not_larger': bool(
            abs(new.buyer_amount_bias) <= abs(old.buyer_amount_bias)),
    }


def main() -> None:
    data = load_development_data()
    original_path = _check_inputs()
    original = pd.read_csv(original_path, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    original = original.loc[original.method.isin(METHODS[:-1])].copy()
    if (original.duplicated(KEYS + ['method']).any()
            or set(original.snapshot_date) != set(DEVELOPMENT_DATES)):
        raise ValueError('Original conditional comparison is incomplete')
    rows, folds, monthly = [], [], []
    for date in DEVELOPMENT_DATES:
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy()
        if holdout.empty:
            raise ValueError(f'No holdout rows: {date.date()}')
        new_scores = fit_predict_dynamics(fit, holdout)
        saved = original.loc[original.snapshot_date.eq(date)].copy()
        if (len(saved) != len(holdout) * 3
                or set(saved.customer_id) != set(holdout.customer_id)):
            raise ValueError(f'Original conditional cohort differs: {date.date()}')
        for method in METHODS:
            if method == 'conditional_l1_order_dynamics':
                part = holdout[KEYS + [TARGET]].copy()
                part['predicted_amount_if_purchase_30d'] = new_scores
            else:
                part = saved.loc[saved.method.eq(method),
                                 KEYS + [TARGET,
                                         'predicted_amount_if_purchase_30d']].copy()
                audit = holdout[KEYS + [TARGET]].merge(
                    part[KEYS + [TARGET]], on=KEYS, how='outer',
                    validate='one_to_one', indicator=True, suffixes=('_current', '_old'))
                if (len(audit) != len(holdout) or not audit._merge.eq('both').all()
                        or not np.allclose(audit[f'{TARGET}_current'],
                                           audit[f'{TARGET}_old'], atol=1e-9, rtol=0)):
                    raise ValueError(f'Original labels differ: {date.date()} / {method}')
            part['method'] = method
            metric = conditional_metrics(
                part[TARGET].to_numpy(dtype=float),
                part.predicted_amount_if_purchase_30d.to_numpy(dtype=float))
            monthly.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                            'method': method, **metric})
            rows.append(part)
        folds.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                      'train_rows': len(fit),
                      'train_buyer_rows': int(fit[TARGET].gt(0).sum()),
                      'last_train_label_end': (
                          fit.snapshot_date.max() + pd.Timedelta(days=30)
                      ).strftime('%Y-%m-%d'),
                      'target_rows': len(holdout)})
        print(f'completed order dynamics cutoff {date.date()}', flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    monthly = pd.DataFrame(monthly)
    summary = []
    for period, dates in [('rolling_jan_may', DEVELOPMENT_DATES[:5]),
                          ('known_jun_aug', DEVELOPMENT_DATES[5:])]:
        for method in METHODS:
            part = predictions.loc[predictions.snapshot_date.isin(dates)
                                   & predictions.method.eq(method)]
            summary.append({'period': period, 'method': method,
                            **conditional_metrics(
                                part[TARGET].to_numpy(dtype=float),
                                part.predicted_amount_if_purchase_30d.to_numpy(dtype=float))})
    summary = pd.DataFrame(summary)
    checks = advancement_checks(summary, monthly)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / 'development_predictions.csv', index=False,
                       encoding='utf-8-sig', date_format='%Y-%m-%d')
    monthly.to_csv(OUTPUT / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUTPUT / 'period_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(OUTPUT / 'folds.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'status': 'development_only_conditional_amount_not_unconditional_forecast',
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'original_predictions_sha256': sha256(original_path),
        'dynamics_features_sha256': sha256(DYNAMICS_PATH),
        'development_predictions_sha256': sha256(OUTPUT / 'development_predictions.csv'),
        'test_rows_used': False,
        'advancement_checks': checks,
        'advance_to_new_test': all(checks.values()),
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 订单金额变化特征：购买后金额研发结果', '',
             '本实验只检验“若未来购买，会花多少”。1–5 月用于开发；已查看的 6–8 月'
             '仅作方向核对；9–11 月未读取。不能把条件金额当成全体客户的预计消费额。', '',
             '| 期间 | 方法 | 购买者 MAE | 购买者金额偏差 | 购买者数 |',
             '| --- | --- | ---: | ---: | ---: |']
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.buyer_mae:.2f} | '
                     f'{row.buyer_amount_bias:+.2%} | {row.buyer_count:,} |')
    lines += ['', '## 事前固定的研发推进条件', '']
    for name, passed in checks.items():
        lines.append(f'- `{name}`：{"满足" if passed else "未满足"}')
    lines += ['', f'值得进入新的独立检验：**{"是" if all(checks.values()) else "否"}**。'
              '这不改变原无条件金额强化目标，也不能作为系统接入依据。',
              '逐月、逐客户、训练标签截止、特征逐分核对及哈希见同目录。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote order dynamics results to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
