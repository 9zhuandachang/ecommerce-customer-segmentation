"""Locked 30-day-grid validation on the two-year Complete Journey file.

The Kaggle mirror is used only as a private local cache. Repository outputs
contain aggregate metrics, source hashes, and the time-split audit.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from src.features import PipelineConfig, build_customer_snapshots
from value_prediction.experiments import TARGET, evaluate
from value_prediction.external_validation import score_method
from value_prediction.labels import build_labels, sha256
from value_prediction.optimization import fit_rows_for_date
from value_prediction.temporal_features import build_temporal_features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'value_prediction/outputs/two_year_validation'
SOURCE_URL = ('https://www.kaggle.com/api/v1/datasets/download/'
              'frtgnn/dunnhumby-the-complete-journey?file_name=transaction_data.csv')
SOURCE = Path(tempfile.gettempdir()) / 'ecommerce_spend_external/transaction_data.csv.zip'
SOURCE_SHA256 = '46105dd6cd44c9813efa2cd5f8e71a62a3bf46fe2bea613ce60353de1b5a63cd'
PSEUDO_ORIGIN = pd.Timestamp('2017-01-01')
METHODS = ('historic_180d', 'recent_90d', 'extended_tweedie')


def source_path() -> Path:
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SOURCE.with_suffix('.download')
        try:
            with urlopen(SOURCE_URL, timeout=90) as response, temporary.open('wb') as dest:
                shutil.copyfileobj(response, dest)
            if sha256(temporary) != SOURCE_SHA256:
                raise ValueError('Two-year source hash mismatch')
            temporary.replace(SOURCE)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError('Two-year source hash mismatch')
    return SOURCE


def adapt_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = ['household_key', 'BASKET_ID', 'DAY', 'PRODUCT_ID',
                'QUANTITY', 'SALES_VALUE']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing source columns: {sorted(missing)}')
    day = pd.to_numeric(raw.DAY, errors='coerce')
    amount = pd.to_numeric(raw.SALES_VALUE, errors='coerce')
    quantity = pd.to_numeric(raw.QUANTITY, errors='coerce')
    valid = raw[required].notna().all(axis=1)
    valid &= day.gt(0) & day.eq(np.floor(day))
    valid &= amount.gt(0) & quantity.gt(0)
    clean = raw.loc[valid, required].copy()
    clean['source_day'] = day.loc[valid].astype('int64')
    clean['invoice_date'] = (PSEUDO_ORIGIN
                             + pd.to_timedelta(clean.source_day - 1, unit='D'))
    clean['customer_id'] = clean.household_key.astype('string')
    clean['invoice'] = (clean.customer_id + '/'
                        + clean.BASKET_ID.astype('string'))
    clean['stock_code'] = clean.PRODUCT_ID.astype('string')
    clean['line_amount'] = amount.loc[valid].astype(float)
    cents = np.rint(clean.line_amount.to_numpy() * 100)
    if not np.allclose(clean.line_amount.to_numpy() * 100,
                       cents, rtol=0, atol=1e-6):
        raise ValueError('Positive source sales are not exact cents')
    return clean[['customer_id', 'invoice', 'stock_code', 'invoice_date',
                  'line_amount', 'source_day']], {
                      'source_rows': len(raw), 'eligible_rows': len(clean),
                      'excluded_rows': int((~valid).sum()),
                      'source_day_min': int(day.min()),
                      'source_day_max': int(day.max()),
                  }


def build_schedule(max_day: int) -> pd.DataFrame:
    """30-day spaced cutoffs, with complete future labels and final three held out."""
    source_days = list(range(181, max_day - 29, 30))
    if len(source_days) < 9:
        raise ValueError('Fewer than nine complete 30-day-grid cutoffs')
    schedule = pd.DataFrame({'source_day': source_days})
    schedule['snapshot_date'] = (PSEUDO_ORIGIN
                                 + pd.to_timedelta(schedule.source_day - 1, unit='D'))
    schedule['split'] = (['train'] * (len(source_days) - 6)
                         + ['validation'] * 3 + ['test'] * 3)
    schedule['feature_start'] = schedule.snapshot_date - pd.Timedelta(days=180)
    schedule['feature_end_exclusive'] = schedule.snapshot_date
    schedule['label_start'] = schedule.snapshot_date
    schedule['label_end_exclusive'] = schedule.snapshot_date + pd.Timedelta(days=30)
    schedule['use_for_modeling'] = True
    return schedule


def main() -> None:
    source = source_path()
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if names != ['transaction_data.csv']:
            raise ValueError(f'Unexpected zip members: {names}')
        with archive.open(names[0]) as csv_stream:
            raw = pd.read_csv(csv_stream, usecols=[
                'household_key', 'BASKET_ID', 'DAY', 'PRODUCT_ID',
                'QUANTITY', 'SALES_VALUE'], low_memory=False)
    clean, source_audit = adapt_transactions(raw)
    schedule = build_schedule(int(clean.source_day.max()))
    # Source day numbers are only an ordered index, not actual calendar dates.
    clean = clean.drop(columns='source_day')
    snapshot = build_customer_snapshots(clean, schedule, PipelineConfig())
    labels, label_audit = build_labels(snapshot, clean, schedule)
    temporal = build_temporal_features(snapshot, clean, schedule)
    data = snapshot.merge(labels[['snapshot_date', 'customer_id', TARGET]],
                          on=['snapshot_date', 'customer_id'], validate='one_to_one')
    data = data.merge(temporal, on=['snapshot_date', 'customer_id'],
                      validate='one_to_one')
    if len(data) != len(snapshot) or not label_audit.reconciled.all():
        raise ValueError('Two-year source reconciliation failed')

    metrics, held_out = [], []
    for row in schedule.loc[schedule.split.isin(['validation', 'test'])].itertuples():
        date = row.snapshot_date
        fit = fit_rows_for_date(data, date)
        holdout = data.loc[data.snapshot_date.eq(date)].copy().reset_index(drop=True)
        if holdout.empty:
            raise ValueError(f'No snapshots at source day {row.source_day}')
        for method in METHODS:
            score = score_method(method, fit, holdout)
            if not np.isfinite(score).all() or (score < 0).any():
                raise ValueError(f'Invalid prediction: {method}')
            result = evaluate(holdout, score)
            metrics.append({'source_day': int(row.source_day), 'split': row.split,
                            'method': method, 'fit_count': len(fit),
                            'latest_fit_label_end_day': int((
                                fit.snapshot_date.max() + pd.Timedelta(days=30)
                                - PSEUDO_ORIGIN).days + 1),
                            'zero_spend_share': float(holdout[TARGET].eq(0).mean()),
                            **result})
            if row.split == 'test':
                part = holdout[['snapshot_date', 'customer_id', TARGET]].copy()
                part['method'] = method
                part['score'] = score
                held_out.append(part)
        print(f'completed source day {row.source_day} ({row.split})', flush=True)

    monthly = pd.DataFrame(metrics)
    pooled = []
    holdout_scores = pd.concat(held_out, ignore_index=True)
    for method in METHODS:
        part = holdout_scores.loc[holdout_scores.method.eq(method)].copy()
        pooled.append({'method': method, 'split': 'test_pooled',
                       **evaluate(part, part.score.to_numpy(dtype=float))})
    pooled = pd.DataFrame(pooled)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT / 'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    pooled.to_csv(OUTPUT / 'pooled_test_metrics.csv', index=False, encoding='utf-8-sig')
    manifest = {
        'status': 'locked_cross_retailer_test_not_original_ecommerce_acceptance',
        'source_url': SOURCE_URL,
        'source_sha256': sha256(source),
        'protocol_sha256': sha256(ROOT / 'value_prediction/two_year_protocol.md'),
        'script_sha256': sha256(Path(__file__)),
        'source_audit': source_audit,
        'snapshot_count': len(snapshot),
        'label_reconciled': bool(label_audit.reconciled.all()),
        'cutoffs': [
            {'source_day': int(row.source_day), 'split': row.split,
             'snapshot_count': int(snapshot.snapshot_date.eq(row.snapshot_date).sum())}
            for row in schedule.itertuples(index=False)
        ],
        'pseudo_origin_is_not_real_calendar_date': True,
        'raw_transactions_saved_in_repository': False,
        'customer_predictions_saved_in_repository': False,
    }
    (OUTPUT / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# 两年零售交易的固定时间检验', '',
        '真实时间信息只有 DAY 日序号；截点依 `two_year_protocol.md` 固定。',
        '同一零售商的公开一年版此前曾用于研究；本次最后三个截点处于更晚的第二年。',
        '原始行及家庭级预测没有写入仓库。', '',
        '| 截点 DAY | 分区 | 方法 | MAE | 购买者 MAE | Top20 覆盖 | 总额偏差 |',
        '| ---: | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in monthly.itertuples(index=False):
        lines.append(f'| {row.source_day} | {row.split} | {row.method} | '
                     f'{row.mae:.2f} | {row.positive_mae:.2f} | '
                     f'{row.top20_spend_capture:.2%} | {row.aggregate_bias:+.2%} |')
    lines += ['', '最后三个截点合并（仍按每截点分别选 Top 20%）：', '',
              '| 方法 | MAE | 购买者 MAE | Top20 覆盖 | 总额偏差 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in pooled.itertuples(index=False):
        lines.append(f'| {row.method} | {row.mae:.2f} | {row.positive_mae:.2f} | '
                     f'{row.top20_spend_capture:.2%} | {row.aggregate_bias:+.2%} |')
    lines += ['', '这是跨零售商的方法检验，不能代替原在线零售项目未来新月份的验收。', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote two-year aggregate report to {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
