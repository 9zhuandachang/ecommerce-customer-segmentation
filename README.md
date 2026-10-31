# E-commerce Customer Analytics

Customer Insights uses UCI Online Retail II purchase records to segment customers, score their likelihood of buying again within 30 days, rank spending potential, and generate purchase-habit profiles and product recommendations. Upload purchase history and select an analysis date to view, filter, and export all four analyses.

Start with the [Project Learning Guide](docs/project_learning_guide.md), or download the [HTML reading edition](docs/project_learning_guide.html) and open it in a browser. See the [Upload Guide](docs/upload_guide.md) for input fields and instructions.

## Workflow diagram

![Customer Insights workflow](docs/images/project_workflow_en.png)

[Open full-size image](docs/images/project_workflow_en.png) | [Editable SVG](docs/images/project_workflow_en.svg)

## Current models and outputs

| Function | Current approach | What the page provides |
| --- | --- | --- |
| Customer segmentation | K-means, 4 clusters | Segment sizes, purchase characteristics, and customer lists. |
| Repurchase within 30 days | 14 historical features + XGBoost | Repurchase scores and customer priorities. |
| Spending potential | 20 historical features + two-stage LightGBM | Customer ranks and a top-20% priority list. |
| Profiles and product recommendations | Full feature group D, 34 features + LightGBM classifier | Purchase-habit profiles and 10 recommended products per customer. |

The default recommender is **group D + LightGBM classification**. It combines product purchase history, customer profiles, and EASE product-association signals to score candidate products and select the top ten. Personal repeat purchase remains an offline comparison method; results are explained in the learning guide and experiment reports.

## Quick start

Verified with Python 3.11. Saved models require compatible training-library versions; install the dependencies in `requirements.txt`. Data and large artifacts use Git LFS, so download their actual contents after cloning.

```powershell
git clone https://github.com/9zhuandachang/ecommerce-customer-segmentation.git
Set-Location ecommerce-customer-segmentation
git lfs install
git lfs pull
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501). If you are already in the project root, use the final command above to start the app.

1. Open **Data import** and upload `data/demo/2011-12-01/purchase_history.csv`.
2. Set **Analysis date** to **2011-12-01** and click **Generate analysis**.
3. Open the four analysis pages to view, filter, and export results.

Any date from **2011-09-01 through 2011-12-01, inclusive**, is supported. The demo contains 759,615 real eligible transaction lines, ending on 2011-11-30. An analysis date of December 1 produces snapshots for 3,444 customers, including 689 customers in the top 20% for spending potential. The system extracts purchases before the selected date and applies the saved models.

The total customer count appears beside the title; matched and displayed counts appear above each table. Median spending appears in the segment comparison. On the recommendation page, customers can be sorted by purchases in the last 180 days. Available values from ten purchase-habit indicators appear in summary cards and **Profile details**.

To change the file or date, return to **Data import** and generate a new analysis. Existing results remain available until the new analysis succeeds. Sessions using the previous rule-based recommender prompt users to regenerate results with the current model. If a code update causes a stale-module import error, press `Ctrl+C` in the server terminal and restart the app. Refresh with `Ctrl+F5` after a style update.

## Repository structure

```text
main/
|-- app/                 Streamlit pages, upload handling, filters, and exports
|-- src/                 Shared data processing and current business models
|   |-- data/
|   |-- segmentation/
|   |-- repurchase/
|   |-- spending/
|   |-- recommendation/
|   |-- integration/
|   `-- paths.py         Central repository paths
|-- pipelines/           Data preparation, comparisons, and model integration
|-- artifacts/           Current models, selected results, and recommendation audit
|-- archive/             Earlier experiments, adapters, protocols, and results
|-- data/                Original, processed, and upload-demo data
|-- docs/                Learning guide, protocols, examples, and workflow figures
`-- tests/               Data, customer-model, spending, and recommendation checks
```

| Location | Purpose |
| --- | --- |
| `app/dashboard.py` | Business interface entry point. |
| `app/upload_inference.py` | Validates purchase history and applies the four current analyses. |
| `src/data/` | Shared cleaning, snapshots, schema validation, and transformations. |
| `src/segmentation/`, `src/repurchase/`, `src/spending/`, `src/recommendation/` | Current feature calculation, scoring, and model comparison logic. |
| `src/integration/` | Combines the selected segmentation and repurchase models. |
| `pipelines/` | Explicit command-line entry points for preparing data and comparing models. |
| `artifacts/<module>/current/` | Current model packages, selection records, predictions, and reports. |
| `artifacts/recommendation/audit/` | Full-list reproduction and feature-contribution audit. |
| `archive/` | Earlier experiments and their preserved results, organized by business area. |
| `data/raw/`, `data/processed/`, `data/demo/` | Original transactions, derived datasets, and the real upload example. |
| `docs/`, `tests/` | Documentation and maintained regression tests. |

See the [repository structure guide](docs/repository_structure.md) for the main entry points and artifact locations.

## Rebuilding data and models

Saved results are included, so you can open the app directly. To repeat an experiment, run the relevant entry points in order from the project root.

```powershell
# Clean transactions, build monthly snapshots, and produce audits.
python -B -m pipelines.snapshots data/raw/online_retail_II_2009-2010.csv data/raw/online_retail_II_2010-2011.csv

# Compare three clustering methods with 2-6 clusters; select four-cluster K-means.
python -B -m pipelines.segmentation compare
python -B -m pipelines.segmentation select --algorithm kmeans --k 4

# Compare four repurchase models across three feature groups, then integrate the selected models.
python -B -m pipelines.repurchase
python -B -m pipelines.integrate

# Build spending labels and 20 historical features; compare spending-potential ranking methods.
python -B -m pipelines.spending_features
python -B -m pipelines.spending

# Compare recommenders, profile-based hybrids, and item-level scoring models.
python -B -m pipelines.recommendation_baselines
python -B -m pipelines.recommendation_profiles
python -B -m pipelines.recommendation

# Rebuild the real upload example.
python -B -m pipelines.prepare_demo
```

Experiments retain source hashes and frozen configurations. Some entry points verify existing artifacts instead of overwriting them. See the corresponding protocols for parameters and time windows.

## Results and comparison reports

| Topic | Report or model |
| --- | --- |
| Clustering methods and cluster counts | [Segmentation protocol](docs/protocols/segmentation.md), `artifacts/segmentation/current/` |
| Repurchase models | [Repurchase protocol](docs/protocols/repurchase.md), `artifacts/repurchase/current/` |
| Integrated segmentation and repurchase models | [Integrated replay report](artifacts/integration/current/report.md), `artifacts/integration/current/model/integrated_bundle.joblib` |
| Spending-potential ranking | [Ranking report](artifacts/spending/current/report.md), `artifacts/spending/current/ranking_bundle.joblib` |
| Recommendation algorithms | [Recommendation report](archive/artifacts/recommendation/v1/report.md), `archive/artifacts/recommendation/v1/selected_model.joblib` |
| Profile hybrids and item-level models | [Profile comparison](archive/artifacts/recommendation/profile_v2/report.md), [Item-level comparison](artifacts/recommendation/current/report.md) |

Training snapshots cover June 2010-May 2011; validation covers June-August 2011; test-period review covers September-November 2011. The selected repurchase model achieves AP 0.6201 and Precision@20% 66.22%. The spending-potential list captures a monthly average of 70.23% of actual spending. The default group-D recommender achieves NDCG@10 0.3001, Recall@10 17.79%, and Hit@10 76.93%. The learning guide explains the calculation units and aggregation.

## Validation and retained experiments

```powershell
python -B -m unittest discover -s tests -v
```

Earlier clustering, repurchase, amount-prediction, and recommendation iterations are retained in `archive/`, with their original results under `archive/artifacts/`. Current models and result reports are under `artifacts/`; shared data processing is in `src/data/`. The interface and current domain packages use `src/` without importing archived experiments.
