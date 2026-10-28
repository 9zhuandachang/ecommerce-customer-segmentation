"""Fixed calendar-seasonality candidate for leakage-safe spend backtesting."""

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
from value_prediction.ranking_audit import load_predictions, period_for
from value_prediction.seasonal_features import FEATURE_PATH as SEASONAL_PATH
from value_prediction.temporal_features import FEATURE_PATH, FEATURES as EXTRA_FEATURES

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'outputs/seasonality_experiment'
PROTOCOL = HERE / 'seasonality_protocol.md'
DEV_REFERENCE = HERE / 'outputs/optimization/development_predictions.csv'
RETRO_REFERENCE = HERE / 'outputs/retrospective_check/customer_predictions.csv'
DATES = pd.date_range('2011-01-01', '2011-11-01', freq='MS')
METHODS = ('historic_180d', 'extended_tweedie', 'calendar_tweedie')
CALENDAR_FEATURES = ('month_sin', 'month_cos')
FEATURES = BASE_FEATURES + EXTRA_FEATURES + list(CALENDAR_FEATURES)


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    dates = pd.to_datetime(result.snapshot_date, errors='raise')
    if dates.isna().any():
        raise ValueError('Invalid snapshot date for calendar features')
    month = dates.dt.month.to_numpy(dtype=float)
    result['month_sin'] = np.sin(2 * np.pi * month / 12)
    result['month_cos'] = np.cos(2 * np.pi * month / 12)
    return result


def _join_source(data: pd.DataFrame, path: Path,
                 required: list[str]) -> pd.DataFrame:
    source = pd.read_csv(path, dtype={'customer_id': 'string'},
                         parse_dates=['snapshot_date'])
    if source.duplicated(['snapshot_date', 'customer_id']).any():
        raise ValueError(f'Duplicate feature source key: {path}')
    merged = data.merge(source, on=['snapshot_date', 'customer_id'],
                        how='left', validate='one_to_one', indicator=True)
    if (len(merged) != len(data) or not merged._merge.eq('both').all()
            or merged[required].isna().any().any()):
        raise ValueError(f'Feature source does not cover cohort: {path}')
    return merged.drop(columns='_merge')


def load_model_data() -> pd.DataFrame:
    check_sources()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    data = attach_labels(snapshots, labels)
    data = _join_source(data, FEATURE_PATH, EXTRA_FEATURES)
    data = _join_source(data, SEASONAL_PATH, [])
    data = add_calendar_features(data)
    if data[FEATURES].isna().any().any():
        raise ValueError('Missing calendar model feature')
    return data


def fit_predict_calendar(fit: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    if (fit.empty or target.empty
            or not fit.snapshot_date.lt(target.snapshot_date.min()).all()
            or (fit.snapshot_date + pd.Timedelta(days=30)
                > target.snapshot_date.min()).any()):
        raise ValueError('Calendar training contains current or immature labels')
    model = lgb.LGBMRegressor(
        objective='tweedie', n_estimators=120, learning_rate=.05,
        num_leaves=15, tweedie_variance_power=1.5,
        random_state=42, n_jobs=1, verbose=-1)
    model.fit(fit[FEATURES], fit[TARGET])
    scores = np.maximum(model.predict(target[FEATURES]), 0)
    if not np.isfinite(scores).all():
        raise ValueError('Calendar model produced nonfinite predictions')
    return scores


def _reference_for_holdout(reference: pd.DataFrame,
                           holdout: pd.DataFrame) -> pd.DataFrame:
    date = pd.Timestamp(holdout.snapshot_date.iloc[0])
    part = reference.loc[reference.snapshot_date.eq(date),
                         ['snapshot_date', 'customer_id', TARGET, *METHODS[:2]]]
    joined = holdout[['snapshot_date', 'customer_id', TARGET]].merge(
        part, on=['snapshot_date', 'customer_id', TARGET],
        how='left', validate='one_to_one', indicator=True)
    if (len(joined) != len(holdout) or not joined._merge.eq('both').all()
            or joined[list(METHODS[:2])].isna().any().any()):
        raise ValueError(f'Reference and calendar cohort do not match: {date}')
    return joined.drop(columns='_merge')


def _predicted_top1(frame: pd.DataFrame,
                    scores: np.ndarray) -> dict:
    ranked = frame[['customer_id', TARGET]].copy()
    ranked['score'] = scores
    n = int(np.ceil(.01 * len(ranked)))
    top = ranked.sort_values(['score', 'customer_id'],
                             ascending=[False, True], kind='stable').head(n)
    actual = float(top[TARGET].sum())
    return {
        'predicted_top1_count': n,
        'predicted_top1_actual_spend': actual,
        'predicted_top1_predicted_spend': float(top.score.sum()),
        'predicted_top1_spend_capture':
        float(actual / frame[TARGET].sum()) if frame[TARGET].sum() > 0 else np.nan,
        'predicted_top1_predicted_to_actual':
        float(top.score.sum() / actual) if actual > 0 else np.nan,
    }


def advancement_checks(summary: pd.DataFrame) -> dict:
    rows = summary.loc[summary.period.eq('rolling_jan_may')].set_index('method')
    base = rows.loc['extended_tweedie']
    model = rows.loc['calendar_tweedie']
    return {
        'buyer_mae_reduction_at_least_5pct':
        bool(model.positive_mae <= .95 * base.positive_mae),
        'overall_mae_increase_at_most_2pct':
        bool(model.mae <= 1.02 * base.mae),
        'top20_capture_not_lower':
        bool(model.top20_spend_capture >= base.top20_spend_capture),
        'aggregate_bias_abs_not_larger':
        bool(abs(model.aggregate_bias) <= abs(base.aggregate_bias)),
    }


def make_report(summary: pd.DataFrame, top1: pd.DataFrame,
                checks: dict) -> str:
    lines = [
        '# 日历月份候选：按时间向前研究', '',
        '事先固定的唯一差别是快照月份的正弦和余弦；其余模型设置与原 20 特征 Tweedie 相同。'
        '2011 年 6–11 月均已查看，只用于开发确认或回顾诊断。', '',
        '| 期间 | 方法 | 全体 MAE | 购买者 MAE | Top20 消费覆盖 | 总额偏差 | Top20 群体偏差 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | {row.top20_group_bias:+.2%} |')
    lines += ['', '## 预测分数事先选出的前 1% 客户：金额尺度诊断', '',
              '| 期间 | 方法 | 名单客户月 | 预测金额/实际金额 | 实际消费覆盖 |',
              '| --- | --- | ---: | ---: | ---: |']
    for row in top1.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.predicted_top1_count} | '
                     f'{row.predicted_top1_predicted_to_actual:.2%} | '
                     f'{row.predicted_top1_spend_capture:.2%} |')
    lines += ['', '该名单仅按预测分数选择；前 1% 指标是诊断项，'
              '不用于事后选择日历模型或调整推进门槛。', '',
              '## 仅 1–5 月的预设研发推进门槛', '']
    for key, passed in checks.items():
        lines.append(f'- `{key}`：{"满足" if passed else "未满足"}')
    lines += ['', '全部满足也只说明值得带到新的固定测试；'
              '这些已查看月份不能提供同商家独立验收。'
              '逐月、客户级预测、训练标签截止与文件哈希见同目录。', '']
    return '\n'.join(lines)


def main() -> None:
    data = load_model_data()
    reference = load_predictions()
    predictions, monthly, folds, top1_rows = [], [], [], []
    for date in DATES:
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if holdout.empty:
            raise ValueError(f'No calendar holdout on {date.date()}')
        original = _reference_for_holdout(reference, holdout)
        candidate = fit_predict_calendar(fit, holdout)
        period = period_for(date)
        folds.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                      'fit_count': len(fit), 'holdout_count': len(holdout),
                      'latest_fit_label_end': (
                          fit.snapshot_date.max() + pd.Timedelta(days=30)
                      ).strftime('%Y-%m-%d')})
        for method in METHODS:
            scores = candidate if method == 'calendar_tweedie' else original[method].to_numpy(dtype=float)
            metrics = score_with_group_bias(holdout, scores)
            monthly.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                            'period': period, 'method': method,
                            'zero_spend_share': float(holdout[TARGET].eq(0).mean()),
                            **metrics})
            top1_rows.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                              'period': period, 'method': method,
                              **_predicted_top1(holdout, scores)})
            part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
            part['method'] = method
            part['predicted_spend_30d'] = scores
            predictions.append(part)
        print(f'completed calendar cutoff {date.date()}', flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summary_rows, top1_period_rows = [], []
    for period, dates in [('rolling_jan_may', DATES[:5]),
                          ('known_jun_aug', DATES[5:8]),
                          ('viewed_sep_nov', DATES[8:])]:
        for method in METHODS:
            part = prediction_frame.loc[prediction_frame.snapshot_date.isin(dates)
                                        & prediction_frame.method.eq(method)]
            summary_rows.append({'period': period, 'method': method,
                                 'zero_spend_share': float(part[TARGET].eq(0).mean()),
                                 **score_with_group_bias(
                                     part, part.predicted_spend_30d.to_numpy(dtype=float))})
            selected = pd.DataFrame(top1_rows)
            selected = selected.loc[selected.period.eq(period)
                                    & selected.method.eq(method)]
            actual = float(selected.predicted_top1_actual_spend.sum())
            top1_period_rows.append({
                'period': period, 'method': method,
                'predicted_top1_count': int(selected.predicted_top1_count.sum()),
                'predicted_top1_actual_spend': actual,
                'predicted_top1_predicted_spend':
                float(selected.predicted_top1_predicted_spend.sum()),
                'predicted_top1_spend_capture': float(actual / part[TARGET].sum()),
                'predicted_top1_predicted_to_actual':
                float(selected.predicted_top1_predicted_spend.sum() / actual)
                if actual > 0 else np.nan,
            })
    summary = pd.DataFrame(summary_rows)
    top1_period = pd.DataFrame(top1_period_rows)
    checks = advancement_checks(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {'customer_predictions': OUTPUT / 'customer_predictions.csv',
             'monthly_metrics': OUTPUT / 'monthly_metrics.csv',
             'period_metrics': OUTPUT / 'period_metrics.csv',
             'predicted_top1_monthly': OUTPUT / 'predicted_top1_monthly.csv',
             'predicted_top1_period': OUTPUT / 'predicted_top1_period.csv',
             'folds': OUTPUT / 'folds.csv'}
    frames = {'customer_predictions': prediction_frame,
              'monthly_metrics': pd.DataFrame(monthly),
              'period_metrics': summary,
              'predicted_top1_monthly': pd.DataFrame(top1_rows),
              'predicted_top1_period': top1_period,
              'folds': pd.DataFrame(folds)}
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False, encoding='utf-8-sig',
                     date_format='%Y-%m-%d')
    report = OUTPUT / 'report.md'
    report.write_text(make_report(summary, top1_period, checks), encoding='utf-8')
    manifest = {
        'status': 'development_and_viewed_retrospective_only',
        'candidate': METHODS[2], 'added_features': CALENDAR_FEATURES,
        'development_gate': checks, 'gate_passed': all(checks.values()),
        'source_snapshot_sha256': sha256(SNAPSHOTS),
        'source_labels_sha256': sha256(LABELS),
        'source_temporal_features_sha256': sha256(FEATURE_PATH),
        'source_seasonal_features_sha256': sha256(SEASONAL_PATH),
        'development_reference_sha256': sha256(DEV_REFERENCE),
        'retrospective_reference_sha256': sha256(RETRO_REFERENCE),
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'outputs_sha256': {name: sha256(path) for name, path in paths.items()},
        'report_sha256': sha256(report),
    }
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                                    indent=2) + '\n', encoding='utf-8')
    print(f'wrote {report}', flush=True)


if __name__ == '__main__':
    main()
