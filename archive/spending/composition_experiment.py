"""Fixed transaction-composition feature candidate on development-only cutoffs."""

from __future__ import annotations

from src.paths import ROOT

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from archive.spending.buyer_weight import (
    DATES, FEATURES as ORIGINAL_FEATURES, REFERENCE, load_development, load_reference,
    reference_for_holdout,
)
from archive.spending.composition_features import (
    FEATURE_PATH, FEATURES as COMPOSITION_FEATURES, MANIFEST_PATH,
    PROTOCOL,
)
from archive.spending.experiments import TARGET
from src.spending.labels import CLEAN, SCHEDULE, SNAPSHOTS, sha256
from archive.spending.optimization import fit_rows_for_date, score_with_group_bias

HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'archive/artifacts/spending/composition_experiment'
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
        '# Features of purchase before snapshot: fixed candidate during development period', '',
        'The scheme is fixed at `composition_protocol.md` before running the model.'
        'New features come from transactions 180 days before each snapshot, trained only with mature historical labels.'
        '2011 years 6 – 8 months have been viewed. The following are not independent acceptances.', '',
        '| Period | Method | All MAE | Buyers MAE | Top 20 Consumption Coverage | Total Deviation | Top 20 Group Deviation |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.period} | {row.method} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | {row.top20_group_bias:+.2%} |')
    lines += ['', '## Development period advancement threshold locked before running', '']
    for key, passed in checks.items():
        lines.append(f'- `{key}`: {"satisfy" if passed else "not satisfied"}')
    lines += ['', 'Only when all four are met is it recommended to enter a new external fixation test.'
              'The three new features or determination thresholds are not changed based on the 6 – 8 monthly results.',
              'Month-by-month, client-level results, training cutoffs, and source hashes are found in the same directory.', '']
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
