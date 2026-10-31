# Business Scope and Data Definitions (Stage 1)

## Goals and users

The interface provides a separate **Data import** page and four analyses: customer segments, repurchase priorities, spending potential, and profiles with product recommendations. Model comparisons, experiment switches, version hashes, and validation procedures remain in repository reports and audits. Business CSV exports contain visible fields, dates, customer identifiers, and relevant profiles, without technical versions or future answers.

The intended users are CRM and customer-retention teams. They review segments, 30-day repurchase scores, spending-potential ranks, and next-order top-ten product lists. The system supports human review, ranking, and export; it does not decide or send marketing campaigns.

The source is the two annual **UCI Online Retail II** CSV files, covering a UK retailer from **2009-12-01 through 2011-12-09**, including wholesale customers. The raw files contain **1,067,371 product transaction lines**. Cleaning retains 776,579 eligible lines and excludes 290,792. Row provenance and primary exclusion reasons are in the cleaning audit. Original files and hashes are in `data/raw/` and `data/processed/customer_snapshot_pipeline/run_manifest.json`.

| Unit | Meaning and use |
| --- | --- |
| Transaction line | One product entry on an invoice; the unit for raw and cleaned row counts. |
| Invoice | A distinct eligible invoice for a customer. Multiple product lines on the same invoice count as one invoice for purchase frequency. |
| Customer | A unique usable customer ID, used for joins and exports rather than prediction features. Missing-ID rows cannot support customer-level analysis. |
| Customer snapshot | One customer's state at cutoff t. Customers can appear at several dates. The 54,040 snapshots across 18 cutoffs are not 54,040 distinct customers. |

## Eligible purchases and snapshots

An eligible purchase is a positive product transaction accepted by `src/data_cleaning.py`: valid date, invoice, product code, customer ID, quantity, and unit price. Cancelled invoices, non-positive quantities/prices, exact duplicates, and non-product entries are excluded. An invoice counts when at least one eligible product line remains. Exclusion precedence, rule counts, and row-level reasons are documented in `docs/customer_snapshot_pipeline.md` and the audits. These figures describe purchases, rather than net revenue after returns.

Offline snapshots use the first day of each month as cutoff t. An existing customer must have an eligible purchase within `[t − 180 days, t)`. Recency, invoice frequency, spending, average order value, product diversity, and 30/90-day invoice counts use only earlier purchases. `customer_history_days` can use older history to find the first visible purchase.

The label `repurchase_30d=1` means at least one eligible purchase within `[t, t + 30 days)`; otherwise it is 0. Keep only cutoffs with complete future labels. Training, validation, and test partitions follow time order; exact boundaries are in `cutoff_schedule.csv`.

## Interface and exports

Users upload purchase history and choose a date. The system excludes transactions on or after that date and generates snapshots for customers who purchased in the preceding 180 days. All four pages share the resulting date and customer batch, rather than reading fixed monthly customer lists. They support customer/segment filtering, display limits, and business CSV exports. See the [Upload Guide](upload_guide.md). December demo snapshots have no future labels and are separate from the original training/evaluation table.

| Field | Meaning |
| --- | --- |
| Customer ID | Join/export identifier, not a model feature. |
| Customer segment | K-means group: 1 long inactive; 2 recent frequent buyers; 3 recent low order value; 4 recent high order value. |
| Repurchase score | XGBoost ranking score for buying again within 30 days; the new model has not been probability-calibrated. |
| Priority rank | Descending repurchase-score position within the current filtered result, starting at 1. |
| Potential rank | Spending-potential rank within the complete uploaded customer batch, preserved after filtering. |
| Analysis date | Any cutoff from 2011-09-01 through 2011-12-01, inclusive. |

Repurchase exports include recency, invoice count, historical spending, and average order value. Upload inference recalculates results with saved K-means, XGBoost, and two-stage LightGBM models rather than retrieving old customer lists.

The default recommender is full group D, 34 features + LightGBM classification. It uses frozen EASE associations and the full frozen catalog, selects the ten highest scores, and breaks ties by catalog order. Offline comparisons retain rules and alternative models; the business interface has no algorithm switch. Exports contain visible business fields, the analysis date, and relevant customer profiles, without future answers, model versions, or raw recommendation scores.

Purchase profiles use events within 180 days. Combine invoices at the same customer/timestamp and deduplicate products within each event. The first event establishes purchase history; subsequent events provide repeat-product shares, which are averaged. Change in repeat share requires at least four comparable orders. “Unseen” means not purchased in the preceding 180 days. Propensity models use historical samples with a next order and fully observed 30-day labels; no next order is not labeled as a zero repeat share. See the [profile/recommendation comparison](../recommendation/outputs/profile_v2/report.md).

## Scope and current status

The data have no marketing-contact or intervention outcomes, so they cannot establish incremental campaign effects, marketing ROI, or causality. A single retailer, missing customer IDs, wholesale behavior, and seasonality affect coverage and transferability. The version excludes campaign sending, real-time integration, new-customer scoring, dashboard training, and automatic retraining. Recommendations include temporal comparison, top-ten lists, and exports; they favor repeat purchases, cover unseen products less strongly, and have no live inventory feed.

The English PPT and Chinese proposal in the parent directory are planning versions from **September 2026**. The new Stage 4 selects four-cluster K-means; Stage 5 selects fourteen-feature XGBoost. Both are integrated into the shared replay. September-November review AP is 0.6201, compared with 0.6064 for base-eight-feature LR; these months are not a new independent test. Planned or early-prototype slides should be updated before the final presentation. Explanations should follow the slides and examples while accurately describing current results and scope.
