"""Forward-only experiment for the 30-day repurchase score's calibration.

Run ``python -m archive.integration.optimize`` from the project root. This experiment keeps
the selected A/LightGBM features and settings fixed. It chooses a calibration
window on March-May 2011, confirms it on June-August, and reports the already
examined September-November period only as a retrospective diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from src.paths import ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

from archive.modelling import build_gbdt_pipeline
from src.data.validation import FEATURES, KEYS, LABEL, load_snapshots
from archive.repurchase.experiments import top20_metrics

INPUT = ROOT / 'data/processed/customer_snapshot_pipeline/customer_snapshots.csv'
STAGE5_CONFIG = ROOT / 'archive/artifacts/repurchase/extended/run_config.json'
OUTPUT = ROOT / 'archive/artifacts/integration/optimization'
LABEL_DAYS = 30
TUNING_DATES = pd.date_range('2011-03-01', '2011-05-01', freq='MS')
CONFIRM_DATES = pd.date_range('2011-06-01', '2011-08-01', freq='MS')
DIAGNOSTIC_DATES = pd.date_range('2011-09-01', '2011-11-01', freq='MS')
BASELINE = 'sigmoid_3m'
CANDIDATES = ('sigmoid_1m', 'sigmoid_2m', 'isotonic_3m')
ALL_METHODS = ('raw', BASELINE) + CANDIDATES
RANK_TOLERANCE = 0.005  # absolute AP and Precision@20 allowance
ECE_TOLERANCE = 0.005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forward_parts(data: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reserve the last three fully labelled months for calibration."""
    target = pd.Timestamp(target)
    eligible = data.loc[
        data.snapshot_date.lt(target)
        & (data.snapshot_date + pd.Timedelta(days=LABEL_DAYS)).le(target)
    ]
    dates = sorted(eligible.snapshot_date.unique())
    if len(dates) < 4:
        raise ValueError(f'Insufficient labelled history before {target.date()}')
    calibration_dates = pd.DatetimeIndex(dates[-3:])
    first_calibration = calibration_dates.min()
    fit = eligible.loc[
        (eligible.snapshot_date + pd.Timedelta(days=LABEL_DAYS)).le(first_calibration)
    ].copy()
    calibration = eligible.loc[eligible.snapshot_date.isin(calibration_dates)].copy()
    heldout = data.loc[data.snapshot_date.eq(target)].copy()
    if fit.empty or calibration.empty or heldout.empty:
        raise ValueError(f'Empty forward split for {target.date()}')
    if fit.snapshot_date.max() + pd.Timedelta(days=LABEL_DAYS) > first_calibration:
        raise AssertionError('Classifier fit label overlaps calibration')
    if calibration.snapshot_date.max() + pd.Timedelta(days=LABEL_DAYS) > target:
        raise AssertionError('Calibration label is unavailable at prediction time')
    if heldout.snapshot_date.nunique() != 1:
        raise AssertionError('A fold must have one target month')
    return fit, calibration, heldout


def expected_calibration_error(y: np.ndarray, scores: np.ndarray) -> float:
    """Ten equal-count score bins; duplicates are combined if necessary."""
    bins = pd.qcut(pd.Series(scores), q=10, labels=False, duplicates='drop')
    table = pd.DataFrame({'bin': bins, 'score': scores, 'label': y}).groupby('bin').agg(
        n=('label', 'size'), predicted=('score', 'mean'), observed=('label', 'mean'))
    return float((table.n * (table.predicted - table.observed).abs()).sum() / len(y))


def score_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict:
    y = frame[LABEL].to_numpy(dtype=int)
    return {
        'snapshot_count': len(frame), 'positive_rate': float(y.mean()),
        'mean_score': float(np.mean(scores)),
        'calibration_gap': float(np.mean(scores) - np.mean(y)),
        'absolute_calibration_gap': float(abs(np.mean(scores) - np.mean(y))),
        'brier': float(brier_score_loss(y, scores)),
        'ece_10': expected_calibration_error(y, scores),
        'average_precision': float(average_precision_score(y, scores)),
        **top20_metrics(frame, scores, y),
    }


def score_fold(data: pd.DataFrame, target: pd.Timestamp, settings: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    fit, calibration, heldout = forward_parts(data, target)
    classifier = build_gbdt_pipeline()
    classifier.set_params(clf__num_leaves=settings['num_leaves'],
                          clf__n_estimators=settings['n_estimators'])
    classifier.fit(fit[FEATURES], fit[LABEL].to_numpy(dtype=int))
    rows: list[dict] = []
    predictions: list[pd.DataFrame] = []
    raw = classifier.predict_proba(heldout[FEATURES])[:, 1]
    for name in ALL_METHODS:
        if name == 'raw':
            scores = raw
        else:
            window = int(name.split('_')[1][0])
            method = name.split('_')[0]
            recent_dates = sorted(calibration.snapshot_date.unique())[-window:]
            recent = calibration.loc[calibration.snapshot_date.isin(recent_dates)]
            calibrator = CalibratedClassifierCV(FrozenEstimator(classifier), method=method)
            calibrator.fit(recent[FEATURES], recent[LABEL].to_numpy(dtype=int))
            scores = calibrator.predict_proba(heldout[FEATURES])[:, 1]
        rows.append({
            'target_date': target.strftime('%Y-%m-%d'), 'method': name,
            'fit_count': len(fit), 'calibration_count': 0 if name == 'raw' else len(recent),
            'last_fit_date': fit.snapshot_date.max().strftime('%Y-%m-%d'),
            'last_calibration_date': calibration.snapshot_date.max().strftime('%Y-%m-%d'),
            **score_metrics(heldout, scores),
        })
        output = heldout[KEYS + [LABEL]].copy()
        output['method'] = name
        output['score'] = scores
        predictions.append(output)
    return rows, predictions


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Pooled ranking metrics; macro means for each month's calibration."""
    result = []
    for method, part in rows.groupby('method', sort=False):
        result.append({
            'method': method, 'months': len(part),
            'mean_brier': float(part.brier.mean()),
            'mean_absolute_calibration_gap': float(part.absolute_calibration_gap.mean()),
            'mean_ece_10': float(part.ece_10.mean()),
            'mean_ap': float(part.average_precision.mean()),
            'mean_precision_at_20pct': float(part.precision_at_20pct.mean()),
        })
    return pd.DataFrame(result)


def passes(candidate: pd.Series, baseline: pd.Series) -> bool:
    """Preset calibration gain with ranking and bin-error safeguards."""
    return bool(
        candidate.mean_brier < baseline.mean_brier
        and candidate.mean_absolute_calibration_gap < baseline.mean_absolute_calibration_gap
        and candidate.mean_ece_10 <= baseline.mean_ece_10 + ECE_TOLERANCE
        and candidate.mean_ap >= baseline.mean_ap - RANK_TOLERANCE
        and candidate.mean_precision_at_20pct >= baseline.mean_precision_at_20pct - RANK_TOLERANCE
    )


def choose_candidate(tuning: pd.DataFrame) -> str:
    summary = aggregate(tuning)
    baseline = summary.loc[summary.method.eq(BASELINE)].iloc[0]
    eligible = summary.loc[summary.method.isin(CANDIDATES)].copy()
    eligible = eligible.loc[eligible.apply(lambda row: passes(row, baseline), axis=1)]
    if eligible.empty:
        return BASELINE
    return str(eligible.sort_values(['mean_brier', 'mean_ece_10', 'method'],
                                    kind='stable').iloc[0].method)


def incumbent_reference(path: Path, stage: str) -> pd.DataFrame:
    """Read previously saved scores only to audit the new recipe's ranking."""
    frame = pd.read_csv(path, dtype={'customer_id': 'string'},
                        parse_dates=['snapshot_date'])
    rows = []
    for date, part in frame.groupby('snapshot_date', sort=True):
        rows.append({'stage': stage, 'target_date': date.strftime('%Y-%m-%d'),
                     'method': 'stage6_incumbent',
                     **score_metrics(part, part.score.to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def ranking_guard(candidate: pd.Series, incumbent: pd.Series) -> bool:
    return bool(candidate.mean_ap >= incumbent.mean_ap - RANK_TOLERANCE
                and candidate.mean_precision_at_20pct
                >= incumbent.mean_precision_at_20pct - RANK_TOLERANCE)


def main() -> None:
    config = json.loads(STAGE5_CONFIG.read_text(encoding='utf-8'))
    if config['snapshot_sha256'] != sha256(INPUT):
        raise ValueError('Stage-five snapshot source changed')
    settings = config['selected']['A_base_8']
    data, _ = load_snapshots(INPUT)
    # Explicitly remove test records before any tuning or confirmation decision.
    development = data.loc[data.split.isin(['train', 'validation'])].copy()
    all_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    for stage, dates, source in [
        ('tuning', TUNING_DATES, development),
        ('confirmation', CONFIRM_DATES, development),
    ]:
        for date in dates:
            rows, predictions = score_fold(source, date, settings)
            all_rows.extend([{'stage': stage, **row} for row in rows])
            all_predictions.extend([prediction.assign(stage=stage) for prediction in predictions])
            print(f'{stage}: {date.date()} fit={rows[0]["fit_count"]} '
                  f'calibration={rows[1]["calibration_count"]}', flush=True)
    metrics = pd.DataFrame(all_rows)
    tuning = metrics.loc[metrics.stage.eq('tuning')]
    confirmation = metrics.loc[metrics.stage.eq('confirmation')]
    selected = choose_candidate(tuning)
    confirm_summary = aggregate(confirmation).set_index('method')
    baseline_confirm = confirm_summary.loc[BASELINE]
    chosen_confirm = confirm_summary.loc[selected]
    brier_wins = int((confirmation.loc[confirmation.method.eq(selected)].set_index('target_date').brier
                      < confirmation.loc[confirmation.method.eq(BASELINE)].set_index('target_date').brier).sum())
    confirmation_passed = (selected != BASELINE and passes(chosen_confirm, baseline_confirm)
                           and brier_wins >= 2)
    incumbent_validation = incumbent_reference(
        ROOT / 'archive/artifacts/integration/final/validation_predictions.csv', 'incumbent_validation')
    incumbent_summary = aggregate(incumbent_validation).iloc[0]
    incumbent_guard_passed = ranking_guard(chosen_confirm, incumbent_summary)
    promotable = bool(confirmation_passed and incumbent_guard_passed)
    # Test months have already been inspected for the earlier stage-six model.
    # They are run only after selecting/fixing the method above, and cannot
    # change the decision or be described as an independent untouched test.
    for date in DIAGNOSTIC_DATES:
        rows, predictions = score_fold(data, date, settings)
        all_rows.extend([{'stage': 'retrospective_test', **row} for row in rows])
        all_predictions.extend([prediction.assign(stage='retrospective_test')
                                for prediction in predictions])
        print(f'retrospective_test: {date.date()}', flush=True)
    metrics = pd.DataFrame(all_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUT / 'predictions.csv', index=False, encoding='utf-8-sig')
    summary = pd.concat([
        aggregate(metrics.loc[metrics.stage.eq(stage)]).assign(stage=stage)
        for stage in ('tuning', 'confirmation', 'retrospective_test')
    ], ignore_index=True)
    summary = pd.concat([summary, aggregate(incumbent_validation).assign(
        stage='incumbent_validation')], ignore_index=True)
    incumbent_test = incumbent_reference(
        ROOT / 'archive/artifacts/integration/final/test_predictions.csv', 'incumbent_retrospective_test')
    summary = pd.concat([summary, aggregate(incumbent_test).assign(
        stage='incumbent_retrospective_test')], ignore_index=True)
    summary.to_csv(OUTPUT / 'summary.csv', index=False, encoding='utf-8-sig')
    incumbent_validation.to_csv(OUTPUT / 'incumbent_validation.csv', index=False,
                                encoding='utf-8-sig')
    incumbent_test.to_csv(OUTPUT / 'incumbent_retrospective_test.csv', index=False,
                          encoding='utf-8-sig')
    decision = {
        'selected_on_tuning': selected,
        'confirmation_passed': bool(confirmation_passed),
        'incumbent_ranking_guard_passed': incumbent_guard_passed,
        'promotable': promotable,
        'brier_better_confirmation_months': brier_wins,
        'promoted_to_stage6_final': False,
        'reason_not_promoted': (
            'Ranking below existing stage-six validation model beyond the 0.005 tolerance.'
            if not incumbent_guard_passed else
            'Existing stage-six model remains unchanged pending a new independent evaluation.'),
        'snapshot_sha256': sha256(INPUT), 'stage5_config_sha256': sha256(STAGE5_CONFIG),
        'features': FEATURES, 'gbdt_num_leaves': settings['num_leaves'],
        'gbdt_n_estimators': settings['n_estimators'],
        'baseline': BASELINE, 'candidates': list(CANDIDATES),
        'label_days': LABEL_DAYS, 'calibration_reserved_months': 3,
        'tuning_dates': [date.strftime('%Y-%m-%d') for date in TUNING_DATES],
        'confirmation_dates': [date.strftime('%Y-%m-%d') for date in CONFIRM_DATES],
        'retrospective_dates': [date.strftime('%Y-%m-%d') for date in DIAGNOSTIC_DATES],
        'selection_rule': 'Tuning: lower mean Brier and absolute mean gap than sigmoid_3m; '
                          'ECE no more than +0.005; mean AP and Precision@20 no more than -0.005. '
                          'Choose lowest Brier. Confirmation repeats safeguards and needs Brier win in 2/3 months. '
                          'Additional incumbent guard: mean AP and Precision@20 each no more than -0.005 '
                          'against the saved stage-six validation ranking.',
    }
    (OUTPUT / 'decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n',
                                          encoding='utf-8')
    report_lines = [
        '# Directional optimization of repurchase model: forward calibration experiment', '',
        'Fixed the 8 features of group A and the LightGBM parameters selected in stage five; before each month\'s prediction,'
        'First, set aside the most recent three fully annotated monthly snapshots for calibration, and then use earlier months to fit the classifier.'
        'Compare raw score, recent 1 / 2 / 3 months sigmoid and recent 3 months isotonic.', '',
        '2011 year 3 – 5 month selection calibration method, 6 – 8 month confirmation; 9 – 11 month only for post-diagnosis.'
        'The existing Phase Six model looked at the 9 – 11 month results, so these months are no longer unlooked at tests for the entire project.', '',
        f'Selection method: **{selected}**; Confirmation threshold: **{"Pass" if confirmation_passed else "failed"}**;'
        f'The confirmation period Brier wins over {brier_wins} / 3 months.'
        f'Sorting threshold with existing models: **{"Pass" if incumbent_guard_passed else "failed"}**.', '',
        '| Stage | Method | AP (monthly average) | Top 20% accuracy (monthly average) | Brier (monthly average) |'
        'Absolute average calibration difference (monthly average) | ECE- 10 (monthly average) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in summary.itertuples(index=False):
        if row.method not in (BASELINE, selected, 'raw', 'stage6_incumbent'):
            continue
        report_lines.append(
            f'| {row.stage} | {row.method} | {row.mean_ap:.4f} | '
            f'{row.mean_precision_at_20pct:.2%} | {row.mean_brier:.4f} | '
            f'{row.mean_absolute_calibration_gap:.2%} | {row.mean_ece_10:.2%} |')
    report_lines += [
        '', 'AP and Top 20% accuracy are used for guard ranking; Brier, mean calibration difference and ten-bin ECE are used for checking probabilities.'
        'All calibrations use only 30 day labels that have ended before the target month.',
        'The calibrator for the existing stage six validation score was fitted on the entire 6 – 8 monthly validation set; its validation period is Brier'
        'Therefore it is not used as a strict forward calibration control. Here only its AP and Top 20% precision rates are used as ranking gatekeeping indicators.',
        f'Compared with the existing model, the new plan confirmation period AP monthly average is lower'
        f'{(incumbent_summary.mean_ap - chosen_confirm.mean_ap):.4f}，'
        f'Top 20% accuracy rate is low on average every month'
        f'{(incumbent_summary.mean_precision_at_20pct - chosen_confirm.mean_precision_at_20pct):.2%}，'
        'The 0.005 tolerance used for this experiment is exceeded and therefore does not replace existing models.'
        '9 – 11 Monthly post-mortem diagnosis does not change this decision.',
        'Existing stage 6 /outputs/final models and reports remain intact; this experiment does not automatically replace online or dashboard scoring.',
        'Complete monthly results, item-by-item ratings and fixed rules can be found in the same directories CSV and decision.json.', '',
    ]
    (OUTPUT / 'report.md').write_text('\n'.join(report_lines), encoding='utf-8')
    print(f'Selected={selected}; confirmation_passed={confirmation_passed}; '
          f'incumbent_guard_passed={incumbent_guard_passed}; '
          f'outputs={OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
