"""Run the precommitted CDNOW cross-retailer amount validation."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.features import PipelineConfig, build_customer_snapshots
from stage4.kmeans import FEATURES as BASE_FEATURES
from value_prediction.calibration import lagged_amount_factor
from value_prediction.experiments import TARGET
from value_prediction.labels import build_labels, sha256
from value_prediction.optimization import fit_rows_for_date, score_with_group_bias
from value_prediction.external_validation import score_method
from value_prediction.temporal_features import FEATURES as EXTRA_FEATURES, build_temporal_features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/cdnow_validation'
SOURCE_URL = 'https://brucehardie.com/datasets/CDNOW_master.zip'
SOURCE_SHA256 = '94081977a983e80dff8edfa58c83a97db33ec2bdd553151987f0dd2f63aaff49'
SOURCE = Path(tempfile.gettempdir()) / 'ecommerce_spend_external/CDNOW_master.zip'
FIRST_CUTOFF = pd.Timestamp('1997-07-01')
LAST_CUTOFF = pd.Timestamp('1998-06-01')
TEST_START = pd.Timestamp('1998-04-01')
METHODS = ('historic_180d', 'recent_90d', 'extended_tweedie',
           'extended_tweedie_lag_calibrated')


def source_path() -> Path:
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        temporary = SOURCE.with_suffix('.download')
        try:
            with requests.get(SOURCE_URL, stream=True, timeout=60,
                              headers={'User-Agent': 'Mozilla/5.0'}) as response:
                response.raise_for_status()
                with temporary.open('wb') as dest:
                    for chunk in response.iter_content(65536):
                        dest.write(chunk)
            if sha256(temporary) != SOURCE_SHA256:
                raise ValueError('CDNOW source hash mismatch')
            temporary.replace(SOURCE)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError('CDNOW source hash mismatch')
    return SOURCE


def adapt_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep every original transaction, including repeated same-day values."""
    expected = ['customer_id', 'date', 'quantity', 'amount']
    if list(raw.columns) != expected or raw[expected].isna().any().any():
        raise ValueError('CDNOW fields are incomplete')
    source = raw.copy()
    source['customer_id'] = source.customer_id.astype('string')
    if not source.customer_id.str.fullmatch(r'\d+').all():
        raise ValueError('Invalid CDNOW customer ID')
    source['invoice_date'] = pd.to_datetime(source.date.astype('string'),
                                            format='%Y%m%d', errors='raise')
    quantity = pd.to_numeric(source.quantity, errors='raise').to_numpy(dtype=float)
    amount = pd.to_numeric(source.amount, errors='raise').to_numpy(dtype=float)
    cents = np.rint(amount * 100)
    if (not np.isfinite(quantity).all() or (quantity <= 0).any()
            or not np.equal(quantity, np.rint(quantity)).all()
            or not np.isfinite(amount).all() or (amount < 0).any()
            or not np.allclose(amount * 100, cents, rtol=0, atol=1e-6)):
        raise ValueError('Invalid CDNOW quantity or dollar amount')
    if (source.invoice_date.min() < pd.Timestamp('1997-01-01')
            or source.invoice_date.max() > pd.Timestamp('1998-06-30')):
        raise ValueError('CDNOW dates exceed documented source period')
    source['source_line'] = np.arange(1, len(source) + 1)
    clean = source.loc[amount > 0].copy()
    clean['invoice'] = clean.customer_id + '/' + clean.source_line.astype('string')
    clean['stock_code'] = 'CD'
    clean['line_amount'] = np.rint(clean.amount.to_numpy(dtype=float) * 100).astype('int64') / 100
    if clean.invoice.duplicated().any():
        raise AssertionError('CDNOW transaction identifiers are not unique')
    return clean[['customer_id', 'invoice', 'stock_code', 'invoice_date',
                  'line_amount']], {
                      'source_rows': len(raw), 'eligible_rows': len(clean),
                      'excluded_rows': int((amount == 0).sum()),
                      'excluded_zero_amount_rows': int((amount == 0).sum()),
                      'source_amount_cents': int(cents.sum()),
                      'eligible_amount_cents': int(np.rint(clean.line_amount.sum() * 100)),
                      'source_customers': int(source.customer_id.nunique()),
                      'eligible_customers': int(clean.customer_id.nunique()),
                  }


def read_source(source: Path) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(source) as archive:
        names = set(archive.namelist())
        if not {'CDNOW_master.txt', 'read_me_CDNOW_master.txt'}.issubset(names):
            raise ValueError('CDNOW source archive members changed')
        with archive.open('CDNOW_master.txt') as member:
            raw = pd.read_csv(member, sep=r'\s+', header=None,
                              names=['customer_id', 'date', 'quantity', 'amount'],
                              dtype={'customer_id': 'string', 'date': 'string'},
                              engine='python')
    if len(raw) != 69659:
        raise ValueError('CDNOW source row count differs from publisher description')
    return adapt_transactions(raw)


def build_schedule(clean: pd.DataFrame) -> pd.DataFrame:
    if (clean.invoice_date.min() > pd.Timestamp('1997-01-01')
            or clean.invoice_date.max() < pd.Timestamp('1998-06-30')):
        raise ValueError('CDNOW source does not cover locked complete windows')
    dates = pd.date_range(FIRST_CUTOFF, LAST_CUTOFF, freq='MS')
    schedule = pd.DataFrame({'snapshot_date': dates})
    schedule['split'] = np.select(
        [schedule.snapshot_date.lt(pd.Timestamp('1997-10-01')),
         schedule.snapshot_date.ge(TEST_START)],
        ['initial_train', 'locked_test'], default='development')
    schedule['feature_start'] = schedule.snapshot_date - pd.Timedelta(days=180)
    schedule['feature_end_exclusive'] = schedule.snapshot_date
    schedule['label_start'] = schedule.snapshot_date
    schedule['label_end_exclusive'] = schedule.snapshot_date + pd.Timedelta(days=30)
    schedule['use_for_modeling'] = True
    return schedule


def score_forward(data: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, forecasts = [], []
    raw_history: list[pd.DataFrame] = []
    for date in schedule.loc[schedule.snapshot_date.ge(pd.Timestamp('1997-10-01')),
                             'snapshot_date']:
        fit = fit_rows_for_date(data, date)
        mature_dates = fit.snapshot_date.nunique()
        if mature_dates < 3:
            raise ValueError('Fewer than three mature CDNOW training cutoffs')
        target = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if target.empty:
            raise ValueError(f'Empty CDNOW target cohort at {date.date()}')
        prior = (pd.concat(raw_history, ignore_index=True)
                 if raw_history else pd.DataFrame())
        if not prior.empty:
            prior = prior.loc[prior.snapshot_date.add(pd.Timedelta(days=30)).le(date)]
        raw_scores = score_method('extended_tweedie', fit, target)
        scores_by_method = {
            'historic_180d': score_method('historic_180d', fit, target),
            'recent_90d': score_method('recent_90d', fit, target),
            'extended_tweedie': raw_scores,
            'extended_tweedie_lag_calibrated': (
                raw_scores * lagged_amount_factor(prior, date)),
        }
        split = target.split.iloc[0]
        for method, scores in scores_by_method.items():
            if len(scores) != len(target) or not np.isfinite(scores).all() or (scores < 0).any():
                raise ValueError(f'Invalid CDNOW scores for {method} at {date.date()}')
            metrics.append({
                'snapshot_date': date.strftime('%Y-%m-%d'), 'split': split,
                'method': method, 'fit_rows': len(fit),
                'fit_cutoffs': int(mature_dates),
                'latest_fit_label_end': (
                    fit.snapshot_date.max() + pd.Timedelta(days=30)).strftime('%Y-%m-%d'),
                **score_with_group_bias(target, np.asarray(scores, dtype=float)),
            })
            scored = target[['snapshot_date', 'customer_id', TARGET]].copy()
            scored['method'] = method
            scored['predicted_spend_30d'] = scores
            forecasts.append(scored)
        raw_history.append(pd.DataFrame({
            'snapshot_date': target.snapshot_date,
            'actual_spend_30d': target[TARGET].to_numpy(dtype=float),
            'raw_predicted_spend_30d': raw_scores,
        }))
        print(f'completed CDNOW cutoff {date.date()}', flush=True)
    return pd.DataFrame(metrics), pd.concat(forecasts, ignore_index=True)


def pooled_test_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        part = forecasts.loc[forecasts.method.eq(method)
                             & forecasts.snapshot_date.ge(TEST_START)].reset_index(drop=True)
        if part.snapshot_date.nunique() != 3:
            raise ValueError('CDNOW locked test lacks three complete months')
        scores = part.predicted_spend_30d.to_numpy(dtype=float)
        rows.append({'method': method, **score_with_group_bias(part, scores)})
    return pd.DataFrame(rows)


def acceptance(monthly: pd.DataFrame, pooled: pd.DataFrame) -> dict:
    test = monthly.loc[monthly.split.eq('locked_test')]
    baseline = pooled.set_index('method').loc['historic_180d']
    decisions = {}
    for method in ('extended_tweedie', 'extended_tweedie_lag_calibrated'):
        result = pooled.set_index('method').loc[method]
        own = test.loc[test.method.eq(method)].set_index('snapshot_date')
        base_month = test.loc[test.method.eq('historic_180d')].set_index('snapshot_date')
        criteria = {
            'mae_10pct_better': bool(result.mae <= baseline.mae * .9),
            'buyer_mae_5pct_better': bool(result.positive_mae <= baseline.positive_mae * .95),
            'top20_capture_3pp_higher': bool(result.top20_spend_capture >= baseline.top20_spend_capture + .03),
            'aggregate_bias_within_5pct': bool(abs(result.aggregate_bias) <= .05),
            'top20_group_bias_within_10pct': bool(abs(result.top20_group_bias) <= .10),
            'all_months_mae_not_worse': bool(own.mae.le(base_month.mae).all()),
            'all_months_top20_not_worse': bool(own.top20_spend_capture.ge(base_month.top20_spend_capture).all()),
        }
        decisions[method] = {'criteria': criteria, 'all_passed': all(criteria.values())}
    return decisions


def main() -> None:
    source = source_path()
    clean, audit = read_source(source)
    if audit['source_amount_cents'] != audit['eligible_amount_cents']:
        raise AssertionError('CDNOW source dollar amounts do not reconcile')
    schedule = build_schedule(clean)
    snapshot = build_customer_snapshots(clean, schedule, PipelineConfig())
    labels, label_audit = build_labels(snapshot, clean, schedule)
    temporal = build_temporal_features(snapshot, clean, schedule)
    data = snapshot.merge(labels[['snapshot_date', 'customer_id', TARGET]],
                          on=['snapshot_date', 'customer_id'], validate='one_to_one')
    data = data.merge(temporal, on=['snapshot_date', 'customer_id'], validate='one_to_one')
    if (len(data) != len(snapshot) or not label_audit.reconciled.all()
            or data[BASE_FEATURES + EXTRA_FEATURES + [TARGET]].isna().any().any()):
        raise ValueError('CDNOW snapshot, label or feature reconciliation failed')
    monthly, forecasts = score_forward(data, schedule)
    pooled = pooled_test_metrics(forecasts)
    test = monthly.loc[monthly.split.eq('locked_test')]
    feasible = bool(test.loc[test.method.eq('historic_180d')]
                    .eval('snapshot_count >= 1000 and buyer_count >= 100').all())
    decision = acceptance(monthly, pooled) if feasible else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    pooled.to_csv(OUTPUT / 'pooled_test_metrics.csv', index=False, encoding='utf-8-sig')
    label_audit.to_csv(OUTPUT / 'label_audit.csv', index=False,
                       encoding='utf-8-sig', date_format='%Y-%m-%d')
    manifest = {
        'status': 'cross_retailer_only_not_same_retailer_acceptance',
        'source_url': SOURCE_URL, 'source_sha256': sha256(source),
        'protocol_sha256': sha256(ROOT / 'value_prediction/cdnow_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'source_audit': audit,
        'date_min': clean.invoice_date.min().strftime('%Y-%m-%d'),
        'date_max': clean.invoice_date.max().strftime('%Y-%m-%d'),
        'snapshot_count': len(snapshot),
        'label_reconciled': bool(label_audit.reconciled.all()),
        'test_feasible': feasible, 'acceptance': decision,
        'raw_transactions_saved_in_repository': False,
        'customer_predictions_saved_in_repository': False,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# CDNOW cross-merchant consumption prediction test', '',
        'Use complete real transaction sources according to `cdnow_protocol.md` \'s pre-fixed rules. The original files are only placed in the system temporary directory, and only the aggregation results are retained in the warehouse.',
        'Amounts are in USD, customers are only from first-time buyers in the first quarter of year 1997; item IDs are not available, item diversity is fixed at 1. This result cannot replace the original merchant\'s independent acceptance in the new month.', '',
        f'Source transaction bank {audit["source_rows"]:,}, customer {audit["source_customers"]:,} people;'
        f'After excluding zero-amount transactions {audit["excluded_zero_amount_rows"]}, there are {audit["eligible_customers"]:,} qualified customers.'
        f'Customer monthly snapshot {len(snapshot):,} items. The sum of the amounts of zero-amount lines is 0, and the transaction amount is verified and passed point by point.',
        f'Last three months viability (monthly ≥ 1,000 snapshots, ≥ 100 future buyers): {"satisfy" if feasible else "dissatisfied"}.', '',
        '| Cutoff | Partition | Method | Number of Customers | Buyers | MAE | Buyers MAE | Top 20 Coverage | Total Deviation | Top 20 Group Deviation |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in monthly.itertuples(index=False):
        lines.append(f'| {row.snapshot_date} | {row.split} | {row.method} | '
                     f'{row.snapshot_count:,} | {row.buyer_count:,} | {row.mae:.2f} | '
                     f'{row.positive_mae:.2f} | {row.top20_spend_capture:.2%} | '
                     f'{row.aggregate_bias:+.2%} | {row.top20_group_bias:+.2%} |')
    lines += ['', '## Last three months combined', '',
              '| Method | MAE | Buyers MAE | Top 20 Coverage | Total Bias | Top 20 Group Bias |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in pooled.itertuples(index=False):
        lines.append(f'| {row.method} | {row.mae:.2f} | {row.positive_mae:.2f} | '
                     f'{row.top20_spend_capture:.2%} | {row.aggregate_bias:+.2%} | '
                     f'{row.top20_group_bias:+.2%} |')
    if decision is not None:
        lines += ['', '## Fixed strengthening goals in advance', '']
        for method, item in decision.items():
            lines.append(f'- `{method}`: {"Satisfy all" if item["all_passed"] else "Not all satisfied"}.'
                         + '；'.join(f'{name} = {"Yes" if passed else "No"}'
                                    for name, passed in item['criteria'].items()))
    lines += ['', 'The monthly training label cutoff, original amount verification and source hash can be found in the same directory file.'
              'Any cross-merchant approval is only evidence of method portability and does not constitute permission to go online with the original merchant\'s amount model.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote CDNOW aggregate results to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
