"""Apply the same strengthened spend criteria to existing fixed comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from value_prediction.labels import sha256


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'outputs/acceptance_gate'
PROTOCOL = HERE / 'acceptance_gate_protocol.md'
BASELINE = 'historic_180d'
CANDIDATE = 'extended_tweedie'
REQUIRED = ['snapshot_count', 'buyer_count', 'actual_spend', 'predicted_spend',
            'mae', 'positive_mae', 'top20_count', 'top20_actual_spend',
            'top20_spend_capture', 'aggregate_bias']
CASES = [
    {'name': 'original_jan_may', 'folder': 'optimization',
     'file': 'monthly_development.csv', 'script': 'optimization.py',
     'source_status': 'development', 'start': '2011-01-01',
     'end': '2011-05-01', 'months': 5},
    {'name': 'original_jun_aug', 'folder': 'optimization',
     'file': 'monthly_development.csv', 'script': 'optimization.py',
     'source_status': 'previously_viewed_confirmation',
     'start': '2011-06-01', 'end': '2011-08-01', 'months': 3},
    {'name': 'original_sep_nov', 'folder': 'retrospective_check',
     'file': 'monthly_metrics.csv', 'script': 'retrospective_check.py',
     'source_status': 'previously_viewed_retrospective',
     'start': '2011-09-01', 'end': '2011-11-01', 'months': 3},
    {'name': 'grocery_two_year', 'folder': 'two_year_validation',
     'file': 'monthly_metrics.csv', 'script': 'two_year_validation.py',
     'source_status': 'different_retailer', 'split': 'test',
     'date_column': 'source_day', 'months': 3},
    {'name': 'jewelry', 'folder': 'jewelry_validation',
     'file': 'monthly_metrics.csv', 'script': 'jewelry_validation.py',
     'source_status': 'different_retailer', 'split': 'test', 'months': 3},
    {'name': 'amazon', 'folder': 'amazon_validation',
     'file': 'monthly_metrics.csv', 'script': 'amazon_validation.py',
     'source_status': 'different_retailer', 'split': 'test', 'months': 3},
    {'name': 'cdnow', 'folder': 'cdnow_validation',
     'file': 'monthly_metrics.csv', 'script': 'cdnow_validation.py',
     'source_status': 'different_retailer', 'split': 'locked_test', 'months': 3},
]


def aggregate_metrics(rows: pd.DataFrame) -> dict:
    if rows.empty:
        raise ValueError('No monthly metrics')
    numeric = rows[REQUIRED].apply(pd.to_numeric, errors='raise')
    if (numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all()
            or (numeric[['snapshot_count', 'buyer_count', 'actual_spend',
                         'top20_count', 'top20_actual_spend']] <= 0).any().any()
            or (numeric[['mae', 'positive_mae', 'predicted_spend']] < 0).any().any()):
        raise ValueError('Invalid monthly amount or count metric')
    if (numeric.buyer_count.gt(numeric.snapshot_count).any()
            or not np.array_equal(numeric.top20_count.to_numpy(dtype=int),
                                  np.ceil(numeric.snapshot_count.to_numpy() * .2).astype(int))
            or numeric.top20_actual_spend.gt(numeric.actual_spend).any()):
        raise ValueError('Monthly cohort or Top20 quota is inconsistent')
    if (not np.allclose(numeric.predicted_spend / numeric.actual_spend - 1,
                        numeric.aggregate_bias, rtol=0, atol=1e-8)
            or not np.allclose(numeric.top20_actual_spend / numeric.actual_spend,
                               numeric.top20_spend_capture, rtol=0, atol=1e-8)):
        raise ValueError('Stored monthly bias or capture does not reconcile')
    actual = float(numeric.actual_spend.sum())
    top_actual = float(numeric.top20_actual_spend.sum())
    top_bias = None
    if ('top20_group_bias' in rows
            and rows.top20_group_bias.notna().all()):
        bias = pd.to_numeric(rows.top20_group_bias, errors='raise').to_numpy(dtype=float)
        if not np.isfinite(bias).all():
            raise ValueError('Invalid top20 group bias')
        top_predicted = float(np.sum(numeric.top20_actual_spend.to_numpy() * (1 + bias)))
        top_bias = top_predicted / top_actual - 1
    return {
        'snapshot_count': int(numeric.snapshot_count.sum()),
        'buyer_count': int(numeric.buyer_count.sum()),
        'actual_spend': actual,
        'mae': float(np.average(numeric.mae, weights=numeric.snapshot_count)),
        'positive_mae': float(np.average(numeric.positive_mae,
                                         weights=numeric.buyer_count)),
        'top20_spend_capture': top_actual / actual,
        'aggregate_bias': float(numeric.predicted_spend.sum() / actual - 1),
        'top20_group_bias': top_bias,
    }


def assess_monthly(monthly: pd.DataFrame, date_column: str = 'snapshot_date',
                   expected_months: int = 3) -> dict:
    if expected_months < 3 or date_column not in monthly:
        raise ValueError('At least three dated months are required')
    if not set(REQUIRED + ['method']).issubset(monthly.columns):
        raise ValueError('Missing required comparison metric')
    data = monthly.loc[monthly.method.isin([BASELINE, CANDIDATE])].copy()
    if data.duplicated([date_column, 'method']).any():
        raise ValueError('Duplicate method-month result')
    baseline = data.loc[data.method.eq(BASELINE)].set_index(date_column).sort_index()
    candidate = data.loc[data.method.eq(CANDIDATE)].set_index(date_column).sort_index()
    if (len(baseline) != expected_months or len(candidate) != expected_months
            or not baseline.index.equals(candidate.index)):
        raise ValueError('Baseline/candidate months differ or are incomplete')
    for column in ['snapshot_count', 'buyer_count', 'top20_count']:
        if not np.array_equal(baseline[column].to_numpy(), candidate[column].to_numpy()):
            raise ValueError(f'Comparison cohorts differ: {column}')
    if not np.allclose(baseline.actual_spend, candidate.actual_spend,
                       rtol=0, atol=1e-6):
        raise ValueError('Comparison actual amounts differ')
    old = aggregate_metrics(baseline)
    new = aggregate_metrics(candidate)
    stable = int((candidate.mae.le(baseline.mae)
                  & candidate.top20_spend_capture.ge(
                      baseline.top20_spend_capture)).sum())
    criteria = {
        'overall_mae_at_least_10pct_better': bool(new['mae'] <= .9 * old['mae']),
        'buyer_mae_at_least_5pct_better': bool(
            new['positive_mae'] <= .95 * old['positive_mae']),
        'top20_capture_at_least_3pp_higher': bool(
            new['top20_spend_capture'] >= old['top20_spend_capture'] + .03),
        'aggregate_bias_within_5pct': bool(abs(new['aggregate_bias']) <= .05),
        'top20_group_bias_within_10pct': (
            None if new['top20_group_bias'] is None
            else bool(abs(new['top20_group_bias']) <= .10)),
        'at_least_three_joint_stable_months': bool(stable >= 3),
    }
    return {'baseline': old, 'candidate': new, 'joint_stable_months': stable,
            'month_count': len(baseline), 'criteria': criteria,
            'technical_passed': all(value is True for value in criteria.values())}


def load_case(spec: dict) -> tuple[pd.DataFrame, dict]:
    folder = HERE / 'outputs' / spec['folder']
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('script_sha256') != sha256(HERE / spec['script']):
        raise ValueError(f'Source script has changed: {spec["name"]}')
    path = folder / spec['file']
    data = pd.read_csv(path)
    if 'split' in spec:
        data = data.loc[data.split.eq(spec['split'])].copy()
    else:
        dates = pd.to_datetime(data.snapshot_date, errors='raise')
        data = data.loc[dates.between(spec['start'], spec['end'])].copy()
    return data, {'manifest_sha256': sha256(folder / 'manifest.json'),
                  'monthly_sha256': sha256(path)}


def main() -> None:
    rows, detail, sources = [], {}, {}
    for spec in CASES:
        monthly, source = load_case(spec)
        decision = assess_monthly(monthly, spec.get('date_column', 'snapshot_date'),
                                  spec['months'])
        # These source states are fixed from the original protocols and manifests.
        # A technical pass alone never upgrades an already-viewed or other-shop case.
        independently_eligible = False
        row = {
            'case': spec['name'], 'source_status': spec['source_status'],
            'independent_same_merchant': independently_eligible,
            'month_count': decision['month_count'],
            'joint_stable_months': decision['joint_stable_months'],
            'baseline_mae': decision['baseline']['mae'],
            'candidate_mae': decision['candidate']['mae'],
            'baseline_buyer_mae': decision['baseline']['positive_mae'],
            'candidate_buyer_mae': decision['candidate']['positive_mae'],
            'top20_capture_gain_pp': 100 * (
                decision['candidate']['top20_spend_capture']
                - decision['baseline']['top20_spend_capture']),
            'candidate_aggregate_bias': decision['candidate']['aggregate_bias'],
            'candidate_top20_group_bias': decision['candidate']['top20_group_bias'],
            **decision['criteria'],
            'technical_passed': decision['technical_passed'],
            'final_accepted': bool(decision['technical_passed'] and independently_eligible),
        }
        rows.append(row)
        detail[spec['name']] = decision
        sources[spec['name']] = source
    scorecard = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(OUTPUT / 'scorecard.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'status': 'reassessment_of_existing_outputs_no_new_independent_test',
        'protocol_sha256': sha256(PROTOCOL),
        'script_sha256': sha256(Path(__file__)),
        'scorecard_sha256': sha256(OUTPUT / 'scorecard.csv'),
        'source_hashes': sources,
        'fixed_criteria': detail,
        'any_final_accepted': bool(scorecard.final_accepted.any()),
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# Unified acceptance review of consumption amount for 30 days in the future', '',
             'All indicators are recalculated using existing month-to-month results according to the original enhanced standards.'
             'These sources have been used to develop, view or belong to other merchants; this list does not generate new independent tests.', '',
             '| Data group | MAE improvement | Buyer MAE improvement | Top 20 coverage improvement | Total deviation | Top 20 group deviation | Month without regression | Technical threshold | Independent acceptance by the original merchant |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for row in scorecard.itertuples(index=False):
        mae_gain = 100 * (1 - row.candidate_mae / row.baseline_mae)
        buyer_gain = 100 * (1 - row.candidate_buyer_mae / row.baseline_buyer_mae)
        group_bias = ('Missing' if pd.isna(row.candidate_top20_group_bias)
                      else f'{row.candidate_top20_group_bias:+.2%}')
        lines.append(f'| {row.case} | {mae_gain:+.2f}% | {buyer_gain:+.2f}% | '
                     f'{row.top20_capture_gain_pp:+.2f} percentage points |'
                     f'{row.candidate_aggregate_bias:+.2%} | {group_bias} | '
                     f'{row.joint_stable_months}/{row.month_count} | '
                     f'{"satisfy" if row.technical_passed else "not fully satisfied"} |'
                     f'{"Yes" if row.final_accepted else "No"} |')
    lines += ['', 'See `scorecard.csv` and `manifest.json` for technical threshold itemized results, source lists, and file hashes.'
              'The missing Top 20 group amount deviation cannot be treated as zero; good results from other merchants cannot replace the original merchant\'s new month.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote strengthened acceptance review to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
