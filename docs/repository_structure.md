# Repository Structure

The repository separates the running interface, reusable model code, pipeline commands, current artifacts, and earlier research.

## Where to start

1. Read the root README for the four business functions and startup command.
2. Open `app/dashboard.py` for page navigation and `app/upload_inference.py` for the upload-to-analysis flow.
3. Read the matching package under `src/` for feature calculation and scoring.
4. Use `pipelines/` when preparing data or repeating model comparisons.
5. Open `artifacts/<module>/current/report.md` for the selected approach and results.

## Code ownership

| Directory | Responsibility |
| --- | --- |
| `app/` | Streamlit interface, upload validation, business views, filters, and exports. |
| `src/data/` | Transaction cleaning, customer snapshots, shared feature schema, and time-boundary checks. |
| `src/segmentation/` | K-means, bisecting K-means, and GMM comparison and assignments. |
| `src/repurchase/` | Repurchase features, model comparison, priority ranking, and metrics. |
| `src/spending/` | Future-spending labels, temporal features, ranking comparison, and shared amount utilities. |
| `src/recommendation/` | Customer profiles, product features, recommendation algorithms, scoring models, and metrics. |
| `src/integration/` | Final packaging and scoring of the selected segmentation and repurchase models. |
| `src/paths.py` | Repository root and the main directory locations, independent of the current working directory. |
| `pipelines/` | Commands that assemble the reusable functions into data preparation and experiments. |
| `tests/` | Data-pipeline, customer-model, spending, recommendation, and directory-boundary regression tests. |

The interface and current domain packages import reusable code from `src/`. Earlier experimental code is under `archive/`. Comparison pipelines may load archived controls to reproduce the original comparisons.

## Current models and results

| Module | Location |
| --- | --- |
| Segmentation comparison and selected K-means | `artifacts/segmentation/current/` |
| Selected repurchase model and comparison | `artifacts/repurchase/current/` |
| Integrated customer model | `artifacts/integration/current/model/integrated_bundle.joblib` |
| Spending-potential ranking | `artifacts/spending/current/ranking_bundle.joblib` |
| Group-D product recommendation model | `artifacts/recommendation/current/model_bundle.joblib` |
| Recommendation reproduction audit | `artifacts/recommendation/audit/` |

The four pages use the saved integrated customer model, spending model, and recommendation model. Model weights, input values, and numerical results are preserved during this reorganization.

## Data

- `data/raw/`: the two original annual UCI CSV files.
- `data/processed/customer_snapshot_pipeline/`: cleaned/excluded transactions, audits, monthly snapshots, and split schedule.
- `data/processed/spending/`: spending labels and historical feature tables used by comparisons.
- `data/demo/2011-12-01/`: the real purchase-history upload example and its provenance manifest.

CSV data are moved without rewriting their contents.

## Earlier research

`archive/` retains earlier clustering, repurchase, amount-prediction, and recommendation experiments. Their results are under `archive/artifacts/`; earlier protocols are under `archive/docs/`; saved-result adapters are under `archive/adapters/`.

The baseline recommendation and profile-hybrid pipelines still use these controls for reproducibility. Verbatim source snapshots remain beside the corresponding archived artifacts.

Importing `archive` provides a small compatibility alias for the original `stage4.transforms.signed_log1p` function stored in older joblib models. This allows those model bytes to remain unchanged after the source file is moved.

The six Word reports are outside this repository in the parent directory's `final_report/` folder.

## Commands

Run commands from the repository root:

```powershell
# Start the unchanged business interface.
python -m streamlit run app/dashboard.py

# Prepare the canonical transaction and snapshot tables.
python -B -m pipelines.snapshots data/raw/online_retail_II_2009-2010.csv data/raw/online_retail_II_2010-2011.csv

# Compare and select customer segments.
python -B -m pipelines.segmentation compare
python -B -m pipelines.segmentation select --algorithm kmeans --k 4

# Compare repurchase models and package the selected customer models.
python -B -m pipelines.repurchase
python -B -m pipelines.integrate

# Prepare spending features and compare customer spending-potential rankings.
python -B -m pipelines.spending_features
python -B -m pipelines.spending

# Reproduce the recommendation comparisons and current-model audit.
python -B -m pipelines.recommendation_baselines
python -B -m pipelines.recommendation_profiles
python -B -m pipelines.recommendation
python -B -m pipelines.audit_recommendation

# Prepare the upload example and run all maintained tests.
python -B -m pipelines.prepare_demo
python -B -m unittest discover -s tests -q
```

Existing frozen experiments may verify their saved artifacts instead of retraining. Reorganization provenance and file hashes are recorded in `docs/repository_layout_manifest.json`.
