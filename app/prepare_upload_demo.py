"""Build a real, label-free December upload demo from the audited clean data.

Run: python -B -m app.prepare_upload_demo
"""

from __future__ import annotations

import json
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd

from app.upload_inference import DEMO, ROOT, TRANSACTION_COLUMNS, digest, validate_upload


def main():
    source = ROOT / 'data/processed/customer_snapshot_pipeline/clean_transactions.csv'
    history = pd.read_csv(source, usecols=TRANSACTION_COLUMNS,
                          dtype={'customer_id': 'string', 'invoice': 'string', 'stock_code': 'string'},
                          parse_dates=['invoice_date'])
    cutoff = pd.Timestamp('2011-12-01')
    history = history.loc[history.invoice_date.lt(cutoff), TRANSACTION_COLUMNS].copy()
    history_bytes = history.to_csv(index=False, date_format='%Y-%m-%d %H:%M:%S').encode('utf-8-sig')
    snapshots, _, ignored = validate_upload(history_bytes, cutoff)
    assert ignored == 0
    DEMO.mkdir(parents=True, exist_ok=True)
    files = {'purchase_history.csv': history_bytes}
    for name, content in files.items():
        (DEMO / name).write_bytes(content)
    manifest = dict(source='UCI Online Retail II / canonical cleaned transactions',
                    source_sha256=digest(source), cutoff=str(cutoff.date()), synthetic=False,
                    future_labels_included=False, snapshot_input_required=False,
                    history_rows=len(history), customers=len(snapshots),
                    earliest_transaction=str(history.invoice_date.min()), latest_transaction=str(history.invoice_date.max()),
                    files={name: digest(DEMO / name) for name in files})
    (DEMO / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with ZipFile(DEMO / 'december_upload_demo.zip', 'w', ZIP_DEFLATED) as archive:
        for name in [*files, 'manifest.json']:
            archive.write(DEMO / name, arcname=name)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
