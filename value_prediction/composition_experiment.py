"""Fixed transaction-composition feature candidate on development-only cutoffs."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from value_prediction.buyer_weight import (
    DATES, FEATURES as ORIGINAL_FEATURES, REFERENCE, load_development, load_reference,
    reference_for_holdout,
)
from value_prediction.composition_features import (
    FEATURE_PATH, FEATURES as COMPOSITION_FEATURES, MANIFEST_PATH,
    PROTOCOL,
)
from value_prediction.experiments import TARGET
from value_prediction.labels import CLEAN, SCHEDULE, SNAPSHOTS, sha256
from value_prediction.optimization import fit_rows_for_date, score_with_group_bias

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'outputs/composition_experiment'
METHODS = ('historic_180d', 'extended_tweedie', 'composition_tweedie')
FEATURES = ORIGINAL_FEATURES + COMPOSITION_FEATURES


def load_candidate_data() -> pd.DataFrame:
    data = load_development()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    expected = {
        'features_sha256': sha256(FEATURE_PATH),
        'source_snapshot_sha256': sha256(SNAPSHOTS),
        'source_clean_sha256': sha256(CLEAN),
        'source_schedule_sha256': sha256(SCHEDULE),
        'protocol_sha256': sha256(PROTOCOL),
    }
    if any(manifest.get(key) != value for key, value in expected.items()) or (
            manifest.get('status') != 'pre_cutoff_features_all_amounts_reconciled'):
        raise ValueError('Composition features or sources changed')
    candidate = pd.read_csv(FEATURE_PATH, dtype={'customer_id': 'string'},
                            parse_dates=['snapshot_date'])
    if candidate.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError('Duplicate composition feature key')
    data = data.merge(candidate, on=['snapshot_date', 'customer_id'],
                      how='left', validate='one_to_one', indicator=True)
    if (not data._merge.eq('both').all()
            or data[COMPOSITION_FEATURES].isna().any().any()):
        raise ValueError('Composition features do not cover development cohort')
    return data.drop(columns='_merge')


def fit_predict_composition(fit: pd.DataFrame,
                            holdout: pd.DataFrame) -> np.ndarray:
    if fit.empty or holdout.empty or not fit.snapshot_date.lt(holdout.snapshot_date.min()).all():
        raise ValueError('Training must precede the holdout')
    if (fit.snapshot_date + pd.Timedelta(days=30)
            > holdout.snapshot_date.min()).any():
        raise ValueError('Immature training labels')
    model = lgb.LGBMRegressor(
        objective='tweedie', n_estimators=120, learning_rate=.05,
        num_leaves=15, tweedie_variance_power=1.5,
        random_state=42, n_jobs=1, verbose=-1)
    model.fit(fit[FEATURES], fit[TARGET])
    scores = np.maximum(model.predict(holdout[FEATURES]), 0)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite composition model scores')
    return scores


def advancement_checks(summary: pd.DataFrame) -> dict:
    period = summary.loc[summary.period.eq('rolling_jan_may')].set_index('method')
    raw = period.loc['extended_tweedie']
    candidate = period.loc['composition_tweedie']
    return {
        'buyer_mae_reduction_at_least_5pct':
        bool(candidate.positive_mae <= .95 * raw.positive_mae),
        'overall_mae_increase_at_most_2pct':
        bool(candidate.mae <= 1.02 * raw.mae),
        'top20_capture_not_lower':
        bool(candidate.top20_spend_capture >= raw.top20_spend_capture),
        'aggregate_bias_abs_not_larger':
        bool(abs(candidate.aggregate_bias) <= abs(raw.aggregate_bias)),
    }


def report_markdown(summary: pd.DataFrame, checks: dict) -> str:
    lines = [
        '# 快照前购买构成特征：开发期固定候选', '',
        '方案在模型运行前固定于 `composition_protocol.md`。'
        '新特征来自每个快照前 180 天交易，仅用已成熟历史标签训练。'
        '2011 年 6–8 月已查看，以下均非独立验收。', '',
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
    lines += ['', '只有四项都满足才建议进入新的外部固定测试。'
              '不根据 6–8 月结果改变三项新特征或判定门槛。',
              '逐月、客户级结果、训练截止和来源哈希见同目录。', '']
    return '\n'.join(lines)


def main() -> None:
    data = load_candidate_data()
    reference = load_reference()
    prediction_rows, month_rows, fold_rows = [], [], []
    for date in DATES:
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if holdout.empty:
            raise ValueError(f'Empty holdout: {date.date()}')
        original = reference_for_holdout(reference, holdout)
        candidate_scores = fit_predict_composition(fit, holdout)
        period = 'rolling_jan_may' if date <= DATES[4] else 'known_jun_aug'
        fold_rows.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                          'fit_count': len(fit), 'holdout_count': len(holdout),
                          'latest_fit_label_end': (
                              fit.snapshot_date.max() + pd.Timedelta(days=30)
                          ).strftime('%Y-%m-%d')})
        for method in METHODS:
            scores = (candidate_scores if method == METHODS[2]
                      else original[method].to_numpy(dtype=float))
            month_rows.append({'period': period,
                               'snapshot_date': date.strftime('%Y-%m-%d'),
                               'method': method,
                               'zero_spend_share': float(holdout[TARGET].eq(0).mean()),
                               **score_with_group_bias(holdout, scores)})
            part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
            part['method'] = method
            part['predicted_spend_30d'] = scores
            prediction_rows.append(part)
        print(f'completed composition cutoff {date.date()}', flush=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    period_rows = []
    for period, dates in [('rolling_jan_may', DATES[:5]),
                          ('known_jun_aug', DATES[5:])]:
        for method in METHODS:
            subset = predictions.loc[predictions.snapshot_date.isin(dates)
                                     & predictions.method.eq(method)]
            period_rows.append({'period': period, 'method': method,
                                'zero_spend_share': float(subset[TARGET].eq(0).mean()),
                                **score_with_group_bias(
                                    subset,
                                    subset.predicted_spend_30d.to_numpy(dtype=float))})
    summary = pd.DataFrame(period_rows)
    checks = advancement_checks(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {'customer_predictions': OUTPUT / 'customer_predictions.csv',
             'monthly_metrics': OUTPUT / 'monthly_metrics.csv',
             'period_metrics': OUTPUT / 'period_metrics.csv',
             'folds': OUTPUT / 'folds.csv'}
    for name, frame in [('customer_predictions', predictions),
                        ('monthly_metrics', pd.DataFrame(month_rows)),
                        ('period_metrics', summary), ('folds', pd.DataFrame(fold_rows))]:
        frame.to_csv(paths[name], index=False, encoding='utf-8-sig',
                     date_format='%Y-%m-%d')
    report = OUTPUT / 'report.md'
    report.write_text(report_markdown(summary, checks), encoding='utf-8')
    manifest = {
        'status': 'development_only_not_independent_acceptance',
        'candidate': METHODS[2], 'added_features': COMPOSITION_FEATURES,
        'gate': checks, 'gate_passed': all(checks.values()),
        'composition_features_sha256': sha256(FEATURE_PATH),
        'composition_feature_manifest_sha256': sha256(MANIFEST_PATH),
        'reference_predictions_sha256': sha256(REFERENCE),
        'protocol_sha256': sha256(PROTOCOL), 'script_sha256': sha256(Path(__file__)),
        'output_sha256': {name: sha256(path) for name, path in paths.items()},
        'report_sha256': sha256(report),
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                    indent=2) + '\n', encoding='utf-8')
    print(f'wrote {report}', flush=True)


if __name__ == '__main__':
    main()
