#Customer snapshot data pipeline

The pipeline leaves the two original CSV unchanged, producing traceable data cleaning results and monthly customer snapshots.

The cleaning rule implementation is located at `src/data_cleaning.py`, and the customer feature, label and time division implementation is located at `src/features.py`; `build_customer_snapshots.py` is responsible for calling and writing out the product. Subsequent training directly reads `customer_snapshots.csv` without repeatedly cleaning the original data.

## Cleaning rules

- Exact duplicate rows keep only the first occurrence of the row.
- Records with missing Customer IDs do not enter customer-level modeling.
- Cancellations of orders, records of incorrect quantities or prices are not considered qualifying purchases.
- Non-commodity codes such as postage, handling fees, discounts, manual adjustments, testing, samples and gift certificates do not enter the purchase features.
- Each excluded record retains the source file, original line number, and a primary reason for exclusion.

## Snapshot rules

- Snapshots are generated on 1 day every month.
- The feature window is `[t-180day, t)`, and data after the cut-off time point is strictly not used.
- Customer must have at least one qualifying purchase within the feature window.
- The tag window is `[t, t+30day)`; when there is at least one qualifying purchase within the window, `repurchase_30d=1`.
- Training, validation and testing are divided in chronological order. Label windows spanning the start of the next partition are marked as purge and are not used for modeling.

## Run

```powershell
python build_customer_snapshots.py `
  data/raw/online_retail_II_2009-2010.csv `
  data/raw/online_retail_II_2010-2011.csv `
  --output-dir data/processed/customer_snapshot_pipeline
```

Main output:

- `clean_transactions.csv`: Qualified transaction details.
- `excluded_transactions.csv`: Excluded records and reasons.
- `cleaning_audit.csv`: Number of hits and main exclusions for each rule.
- `cutoff_schedule.csv`: Snapshots, features, labels and data partition boundaries.
- `customer_snapshots.csv`: Customer characteristics and future 30 days repurchase tag.
- `validation_report.csv`: Line number reconciliation, snapshot uniqueness and anti-leakage check.
- `run_manifest.json`: Source file hash, configuration, date range and output size.

## Test

```powershell
python -m unittest discover -s tests -v
```
