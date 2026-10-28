"""Audit a fixed zero-versus-conditional-median spend decision rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.experiments import LABELS, TARGET
from value_prediction.labels import SNAPSHOTS, sha256
from value_prediction.optimization import score_with_group_bias


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'gated_median_protocol.md'
OUTPUT = HERE / 'outputs/gated_median'
HURDLE = HERE / 'outputs/calibrated_hurdle'
CONDITIONAL = HERE / 'outputs/conditional_amount'
KEYS = ['snapshot_date', 'customer_id']
METHODS = ('historic_180d', 'extended_tweedie',
           'stage6_probability_times_gamma', 'probability_gated_median')
DATES = pd.date_range('2011-01-01', '2011-08-01', freq='MS')


def gated_median(probability: np.ndarray, median_if_purchase: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    median_if_purchase = np.asarray(median_if_purchase, dtype=float)
    if (probability.ndim != 1 or median_if_purchase.shape != probability.shape
            or not np.isfinite(probability).all()
            or not np.isfinite(median_if_purchase).all()
            or (probability < 0).any() or (probability > 1).any()
            or (median_if_purchase < 0).any()):
        raise ValueError('Invalid gated median inputs')
    return np.where(probability > .5, median_if_purchase, 0.0)


def _source_checks() -> tuple[Path, Path]:
    paths = []
    for folder, script, protocol in [
            (HURDLE, HERE / 'calibrated_hurdle.py',
             HERE / 'calibrated_hurdle_protocol.md'),
            (CONDITIONAL, HERE / 'conditional_amount.py',
             HERE / 'optimization_protocol.md')]:
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if (manifest['test_rows_used'] is not False
                or manifest['script_sha256'] != sha256(script)
                or manifest['protocol_sha256'] != sha256(protocol)
                or manifest['snapshot_sha256'] != sha256(SNAPSHOTS)
                or manifest['labels_sha256'] != sha256(LABELS)):
            raise ValueError(f'Changed or ineligible source forecasts: {folder}')
        paths.append(folder / 'development_predictions.csv')
    return paths[0], paths[1]


def load_predictions() -> pd.DataFrame:
    hurdle_path, conditional_path = _source_checks()
    hurdle = pd.read_csv(hurdle_path, dtype={'customer_id': 'string'},
                         parse_dates=['snapshot_date'])
    conditional = pd.read_csv(conditional_path, dtype={'customer_id': 'string'},
                              parse_dates=['snapshot_date'])
    if (hurdle.duplicated(KEYS + ['method']).any()
            or conditional.duplicated(KEYS + ['method']).any()):
        raise ValueError('Duplicate customer-month forecast')
    if (set(hurdle.snapshot_date) != set(DATES)
            or set(conditional.snapshot_date) != set(DATES)):
        raise ValueError('Forecasts contain missing or non-development months')

    parts = []
    for method in METHODS[:-1]:
        part = hurdle.loc[hurdle.method.eq(method)].copy()
        if part.empty:
            raise ValueError(f'Missing hurdle forecast: {method}')
        part = part[KEYS + [TARGET, 'predicted_spend_30d']
                    + (['predicted_purchase_probability']
                       if method == 'stage6_probability_times_gamma' else [])]
        part = part.rename(columns={'predicted_spend_30d': method,
                                    TARGET: f'{TARGET}_{method}'})
        parts.append(part)
    median = conditional.loc[conditional.method.eq('conditional_l1_median')].copy()
    if median.empty:
        raise ValueError('Missing conditional median forecast')
    median = median[KEYS + [TARGET, 'predicted_amount_if_purchase_30d']]
    median = median.rename(columns={TARGET: f'{TARGET}_median',
                                    'predicted_amount_if_purchase_30d': 'conditional_median'})
    result = parts[0]
    for part in parts[1:] + [median]:
        result = result.merge(part, on=KEYS, how='outer', validate='one_to_one',
                              indicator=True)
        if not result._merge.eq('both').all():
            raise ValueError('Forecast customer-month cohorts differ')
        result = result.drop(columns='_merge')
    result = result.sort_values(KEYS, kind='stable').reset_index(drop=True)
    expected = result[f'{TARGET}_{METHODS[0]}'].to_numpy(dtype=float)
    for column in [f'{TARGET}_{method}' for method in METHODS[1:-1]] + [f'{TARGET}_median']:
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(values, expected, atol=1e-9, rtol=0):
            raise ValueError('Future amount labels differ between forecasts')
        result = result.drop(columns=column)
    result = result.rename(columns={f'{TARGET}_{METHODS[0]}': TARGET})
    for method in METHODS[:-1]:
        values = result[method].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Invalid saved score: {method}')
    result['probability_gated_median'] = gated_median(
        result.predicted_purchase_probability.to_numpy(dtype=float),
        result.conditional_median.to_numpy(dtype=float))
    return result


def _metrics(frame: pd.DataFrame, method: str) -> dict:
    scores = frame[method].to_numpy(dtype=float)
    metric = score_with_group_bias(frame, scores)
    metric['nonzero_forecast_count'] = int((scores > 0).sum())
    metric['ranking_has_full_top20_distinction'] = bool(
        (frame.assign(nonzero=scores > 0)
         .groupby('snapshot_date').nonzero.sum()
         >= np.ceil(frame.groupby('snapshot_date').size() * .2)).all())
    return metric


def advancement_checks(summary: pd.DataFrame) -> dict:
    rows = summary.loc[summary.period.eq('rolling_jan_may')].set_index('method')
    old = rows.loc['extended_tweedie']
    new = rows.loc['probability_gated_median']
    return {
        'buyer_mae_reduction_at_least_5pct': bool(new.positive_mae <= .95 * old.positive_mae),
        'overall_mae_increase_at_most_2pct': bool(new.mae <= 1.02 * old.mae),
        'top20_capture_not_lower': bool(new.top20_spend_capture >= old.top20_spend_capture),
        'aggregate_bias_abs_not_larger': bool(abs(new.aggregate_bias) <= abs(old.aggregate_bias)),
        'all_monthly_top20_distinguishable': bool(new.ranking_has_full_top20_distinction),
    }


def main() -> None:
    data = load_predictions()
    monthly, periods = [], []
    for date, frame in data.groupby('snapshot_date', sort=True):
        for method in METHODS:
            monthly.append({'snapshot_date': date.strftime('%Y-%m-%d'),
                            'method': method, **_metrics(frame, method)})
    for period, dates in [('rolling_jan_may', DATES[:5]),
                          ('known_jun_aug', DATES[5:])]:
        frame = data.loc[data.snapshot_date.isin(dates)]
        for method in METHODS:
            periods.append({'period': period, 'method': method,
                            **_metrics(frame, method)})
    monthly = pd.DataFrame(monthly)
    summary = pd.DataFrame(periods)
    checks = advancement_checks(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT / 'development_predictions.csv', index=False,
                encoding='utf-8-sig', date_format='%Y-%m-%d')
    monthly.to_csv(OUTPUT / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUTPUT / 'period_metrics.csv', index=False, encoding='utf-8-sig')
    hurdle_path, conditional_path = _source_checks()
    manifest = {
        'status': 'development_only_not_independent_acceptance',
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'hurdle_predictions_sha256': sha256(hurdle_path),
        'conditional_predictions_sha256': sha256(conditional_path),
        'development_predictions_sha256': sha256(OUTPUT / 'development_predictions.csv'),
        'test_rows_used': False,
        'candidate': 'probability > 0.5 ? conditional_l1_median : 0',
        'advancement_checks': checks,
        'advance_to_new_test': all(checks.values()),
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 概率门控金额中位数：开发期结果', '',
             '阈值 0.5 和所有输入模型均在本次结果前固定。1–5 月和 6–8 月都已查看，'
             '仅供研发判断；9–11 月未参与。该点预测不是期望消费额。', '',
             '| 期间 | 方法 | 全体 MAE | 购买者 MAE | Top20 覆盖 | 总额偏差 | 非零预测人数 | Top20 可完整区分 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | {row.nonzero_forecast_count:,} | '
                     f'{"是" if row.ranking_has_full_top20_distinction else "否"} |')
    lines += ['', '## 事前固定的研发推进条件', '']
    for name, passed in checks.items():
        lines.append(f'- `{name}`：{"满足" if passed else "未满足"}')
    lines += ['', f'可推进新数据检验：**{"是" if all(checks.values()) else "否"}**。'
              '原强化目标仍需单独检验，当前结果不构成独立验收。',
              '逐客户预测、逐月指标与源文件哈希见同目录。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote gated median results to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
