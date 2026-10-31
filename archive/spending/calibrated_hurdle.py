"""Forward-only amount experiment using the existing repurchase classifier."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

from archive.modelling import build_gbdt_pipeline
from src.data.validation import FEATURES as BASE_FEATURES, load_snapshots
from archive.integration.optimize import forward_parts
from archive.spending.experiments import LABELS, TARGET, attach_labels
from src.spending.labels import SNAPSHOTS, sha256
from archive.spending.optimization import (
    DEVELOPMENT_DATES, check_sources, fit_predict, fit_rows_for_date,
    score_with_group_bias,
)
from src.spending.features import FEATURE_PATH, FEATURES as EXTRA_FEATURES

from src.paths import ROOT
OUTPUT = ROOT / 'archive/artifacts/spending/calibrated_hurdle'
PROTOCOL = ROOT / 'archive/docs/spending/calibrated_hurdle_protocol.md'
STAGE5_CONFIG = ROOT / 'archive/artifacts/repurchase/extended/run_config.json'
FEATURES = BASE_FEATURES + EXTRA_FEATURES
METHODS = ('historic_180d', 'extended_tweedie', 'extended_two_stage',
           'stage6_probability_times_gamma')


def combine_probability_amount(probability: np.ndarray,
                               conditional_mean: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    conditional_mean = np.asarray(conditional_mean, dtype=float)
    if (probability.ndim != 1 or conditional_mean.shape != probability.shape
            or not np.isfinite(probability).all()
            or not np.isfinite(conditional_mean).all()
            or (probability < 0).any() or (probability > 1).any()
            or (conditional_mean < 0).any()):
        raise ValueError('Invalid probability or conditional amount')
    return probability * conditional_mean


def fit_predict_candidate(data: pd.DataFrame,
                          date: pd.Timestamp,
                          settings: dict,
                          expected_target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    classifier_fit, calibration, target = forward_parts(data, date)
    if (len(target) != len(expected_target)
            or not target[['snapshot_date', 'customer_id']].reset_index(drop=True).equals(
                expected_target[['snapshot_date', 'customer_id']].reset_index(drop=True))):
        raise ValueError('Repurchase and amount target cohorts are misaligned')
    amount_fit = fit_rows_for_date(data, date)
    if (classifier_fit.snapshot_date.max() + pd.Timedelta(days=30)
            > calibration.snapshot_date.min()):
        raise ValueError('Classifier training overlaps calibration labels')
    recent_dates = sorted(calibration.snapshot_date.unique())[-2:]
    recent = calibration.loc[calibration.snapshot_date.isin(recent_dates)]
    if (recent.snapshot_date.max() + pd.Timedelta(days=30) > date
            or amount_fit.snapshot_date.max() + pd.Timedelta(days=30) > date):
        raise ValueError('Hurdle candidate uses immature labels')
    classifier = build_gbdt_pipeline()
    classifier.set_params(clf__num_leaves=settings['num_leaves'],
                          clf__n_estimators=settings['n_estimators'])
    classifier.fit(classifier_fit[BASE_FEATURES],
                   classifier_fit.repurchase_30d.to_numpy(dtype=int))
    calibrator = CalibratedClassifierCV(FrozenEstimator(classifier), method='sigmoid')
    calibrator.fit(recent[BASE_FEATURES], recent.repurchase_30d.to_numpy(dtype=int))
    probability = calibrator.predict_proba(target[BASE_FEATURES])[:, 1]
    buyers = amount_fit[TARGET].gt(0)
    if buyers.sum() < 100:
        raise ValueError('Too few mature buyer amount rows')
    model = lgb.LGBMRegressor(
        objective='gamma', n_estimators=120, learning_rate=.05,
        num_leaves=15, random_state=42, n_jobs=1, verbose=-1)
    model.fit(amount_fit.loc[buyers, FEATURES], amount_fit.loc[buyers, TARGET])
    mean_if_purchase = np.maximum(model.predict(target[FEATURES]), 0)
    score = combine_probability_amount(probability, mean_if_purchase)
    audit = {
        'snapshot_date': date.strftime('%Y-%m-%d'),
        'classifier_fit_rows': len(classifier_fit),
        'classifier_fit_label_end': (
            classifier_fit.snapshot_date.max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
        'calibration_rows': len(recent),
        'calibration_last_label_end': (
            recent.snapshot_date.max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
        'amount_fit_rows': len(amount_fit),
        'amount_fit_label_end': (
            amount_fit.snapshot_date.max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
        'target_rows': len(target),
    }
    return score, probability, mean_if_purchase, audit


def advancement_checks(summary: pd.DataFrame) -> dict:
    rows = summary.loc[summary.period.eq('rolling_jan_may')].set_index('method')
    original = rows.loc['extended_tweedie']
    candidate = rows.loc['stage6_probability_times_gamma']
    return {
        'buyer_mae_reduction_at_least_5pct':
        bool(candidate.positive_mae <= .95 * original.positive_mae),
        'overall_mae_increase_at_most_2pct':
        bool(candidate.mae <= 1.02 * original.mae),
        'top20_capture_not_lower':
        bool(candidate.top20_spend_capture >= original.top20_spend_capture),
        'aggregate_bias_abs_not_larger':
        bool(abs(candidate.aggregate_bias) <= abs(original.aggregate_bias)),
    }


def load_development_data() -> pd.DataFrame:
    check_sources()
    snapshots, _ = load_snapshots(SNAPSHOTS)
    snapshots = snapshots.loc[snapshots.split.isin(['train', 'validation'])].copy()
    labels = pd.read_csv(LABELS, dtype={'customer_id': 'string'})
    labels = labels.loc[labels.split.isin(['train', 'validation'])].copy()
    temporal = pd.read_csv(FEATURE_PATH, dtype={'customer_id': 'string'},
                           parse_dates=['snapshot_date'])
    data = attach_labels(snapshots, labels).merge(
        temporal, on=['snapshot_date', 'customer_id'],
        how='left', validate='one_to_one', indicator=True)
    if (len(data) != len(snapshots) or not data._merge.eq('both').all()
            or data[FEATURES].isna().any().any()
            or not np.array_equal(data.repurchase_30d.to_numpy(dtype=int),
                                  data[TARGET].gt(0).to_numpy(dtype=int))):
        raise ValueError('Hurdle cohort, labels or feature reconciliation failed')
    return data.drop(columns='_merge')


def main() -> None:
    config = json.loads(STAGE5_CONFIG.read_text(encoding='utf-8'))
    if config['snapshot_sha256'] != sha256(SNAPSHOTS):
        raise ValueError('Repurchase classifier selection used different snapshots')
    settings = config['selected']['A_base_8']
    if (settings['num_leaves'], settings['n_estimators']) != (15, 64):
        raise ValueError('Precommitted classifier settings changed')
    data = load_development_data()
    predictions, monthly, folds = [], [], []
    for date in DEVELOPMENT_DATES:
        target = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if target.empty:
            raise ValueError(f'No hurdle target cohort at {date.date()}')
        fit = fit_rows_for_date(data, date)
        candidate, probability, conditional_mean, audit = (
            fit_predict_candidate(data, date, settings, target))
        folds.append(audit)
        scores = {
            'historic_180d': fit_predict('historic_180d', fit, target),
            'extended_tweedie': fit_predict('extended_tweedie', fit, target),
            'extended_two_stage': fit_predict('extended_two_stage', fit, target),
            'stage6_probability_times_gamma': candidate,
        }
        for method, values in scores.items():
            if len(values) != len(target) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Invalid hurdle score: {date.date()} / {method}')
            row = {
                'snapshot_date': date.strftime('%Y-%m-%d'), 'method': method,
                'zero_spend_share': float(target[TARGET].eq(0).mean()),
                **score_with_group_bias(target, values),
            }
            if method == 'stage6_probability_times_gamma':
                buyer = target[TARGET].gt(0).to_numpy()
                row.update({
                    'purchase_ap': float(average_precision_score(buyer, probability)),
                    'purchase_brier': float(brier_score_loss(buyer, probability)),
                    'predicted_buyer_count_bias': float(probability.sum() / buyer.sum() - 1),
                })
            monthly.append(row)
            part = target[['snapshot_date', 'customer_id', TARGET]].copy()
            part['method'] = method
            part['predicted_spend_30d'] = values
            if method == 'stage6_probability_times_gamma':
                part['predicted_purchase_probability'] = probability
                part['predicted_amount_if_purchase'] = conditional_mean
            predictions.append(part)
        print(f'completed calibrated hurdle cutoff {date.date()}', flush=True)

    forecast = pd.concat(predictions, ignore_index=True)
    monthly_frame = pd.DataFrame(monthly)
    summary_rows = []
    for period, dates in [('rolling_jan_may', DEVELOPMENT_DATES[:5]),
                          ('known_jun_aug', DEVELOPMENT_DATES[5:])]:
        for method in METHODS:
            part = forecast.loc[forecast.method.eq(method)
                                & forecast.snapshot_date.isin(dates)].reset_index(drop=True)
            summary = {
                'period': period, 'method': method,
                'zero_spend_share': float(part[TARGET].eq(0).mean()),
                **score_with_group_bias(part, part.predicted_spend_30d.to_numpy(dtype=float)),
            }
            if method == 'stage6_probability_times_gamma':
                buyer = part[TARGET].gt(0).to_numpy()
                probability = part.predicted_purchase_probability.to_numpy(dtype=float)
                summary.update({
                    'purchase_ap': float(average_precision_score(buyer, probability)),
                    'purchase_brier': float(brier_score_loss(buyer, probability)),
                    'predicted_buyer_count_bias': float(probability.sum() / buyer.sum() - 1),
                })
            summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)
    checks = advancement_checks(summary)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT / 'development_predictions.csv', index=False,
                    encoding='utf-8-sig', date_format='%Y-%m-%d')
    monthly_frame.to_csv(OUTPUT / 'monthly_metrics.csv', index=False,
                         encoding='utf-8-sig')
    summary.to_csv(OUTPUT / 'period_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(folds).to_csv(OUTPUT / 'folds.csv', index=False,
                               encoding='utf-8-sig')
    manifest = {
        'status': 'development_only_not_independent_acceptance',
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'stage5_config_sha256': sha256(STAGE5_CONFIG),
        'snapshot_sha256': sha256(SNAPSHOTS),
        'labels_sha256': sha256(LABELS),
        'temporal_features_sha256': sha256(FEATURE_PATH),
        'test_rows_used': False,
        'advancement_checks': checks,
        'advance_to_new_test': all(checks.values()),
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Probability of repurchase × Amount after purchase: Development period experiment', '',
        'The classifier reuses the Group A settings selected in stage five and is calibrated using only mature history according to the stage six method;'
        'The conditional amount still uses 20 features before the cutoff point. 1 – 5 were used to select repurchase model parameters.'
        '6 – 8 months have also been viewed; neither is an independent amount test. 9 – 11 did not participate in this experiment for months.', '',
        '| Period | Method | All MAE | Buyers MAE | Top 20 Consumption Coverage | Total Deviation | Top 20 Group Deviation |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | {row.top20_group_bias:+.2%} |')
    lines += ['', '| Period | Buy AP | Buy Brier | Predicted purchase number deviation |',
              '| --- | ---: | ---: | ---: |']
    for row in summary.loc[summary.method.eq('stage6_probability_times_gamma')].itertuples(index=False):
        lines.append(f'| {row.period} | {row.purchase_ap:.4f} | '
                     f'{row.purchase_brier:.4f} | {row.predicted_buyer_count_bias:+.2%} |')
    lines += ['', '## Fixed in advance 1 – 5 monthly advancement conditions', '']
    for name, passed in checks.items():
        lines.append(f'- `{name}`: {"satisfy" if passed else "not satisfied"}')
    lines += ['', f'Whether it is worth entering into a new independent test: **{"Yes" if all(checks.values()) else "No"}**.'
              'Development phase results may only be used for research; they will not be included in the dashboard.',
              'Month-by-month results, development-period-by-customer forecasts, training and calibration label cutoffs, and input hashes are listed in the same directory.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote calibrated hurdle results to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
