# Customer Segmentation Recomparison Protocol

This protocol records the final comparison design. A preliminary run allowed multiple evaluation months per held-out customer; the final design keeps one fixed snapshot per held-out customer and discards the preliminary results. It also clarifies that product breadth per invoice is not actual mean basket size. Both corrections preceded the complete comparison. Earlier Stage 4 and later-month results had already been inspected, so this is reanalysis, **not preregistration or a new independent test**. Outputs go to `stage4/outputs/segmentation_comparison/`; existing segmentation and Stage 5-7 artifacts are preserved.

## Research question and samples

- Each sample represents an eligible customer's purchase state on a month's first day, following at least one purchase in the preceding 180 days. Membership can change across months; segment names are not permanent customer identities.
- Use only information available before the snapshot date. Repurchase labels, future amounts/products, existing model scores, customer IDs, and month values are excluded from fitting.
- Retain training June 2010-May 2011, validation June-August 2011, and review September-November 2011. Select algorithms and settings from held-out training customers and **unlabeled** validation structure. Review-period results are displayed after selection.
- Hold out 20% of training customers by ID for structural evaluation. Fit on all training-month records of the remaining customers. With a fixed seed, select one snapshot per held-out customer, avoiding repeated weighting of customers with more months in silhouette and ARI. For stability, refit five times with different seeds, sampling 80% of fitting customers and keeping all their months. This resamples customers rather than individual customer-month rows. Refit the selected approach on the full training period.

## Fixed features and preprocessing

Use five dimensions:

1. `recency_days`.
2. `invoice_frequency_180d`.
3. `average_order_value_180d`.
4. `product_diversity_180d / invoice_frequency_180d`.
5. `invoice_count_90d / invoice_frequency_180d`.

They describe recency, six-month invoice frequency, average order value, distinct-product breadth relative to invoice count, and recent order share. The fourth feature divides distinct product codes over the whole period by invoices; it is not actual mean products per invoice. Total spending is approximately frequency × average order value, so it is not repeated as another distance dimension. Visible history length remains in descriptive profiles; its starting point is not interpreted as registration. Validate non-negative inputs and legal ratios.

Apply `log1p` to frequency, average order value, and breadth, then standardize all five features. Fit transformations only on the current fitting sample and reuse them for held-out, validation, and review customers. Every algorithm uses the same inputs, transformations, fitting rows, and evaluation rows.

## Candidate methods and settings

| Algorithm | Cluster counts | Other settings |
| --- | --- | --- |
| K-means | 2, 3, 4, 5, 6 | `n_init=20`, `max_iter=500`, `random_state=42` |
| Bisecting K-means | 2, 3, 4, 5, 6 | `init=k-means++`, `n_init=10`, `bisecting_strategy=biggest_inertia`, `random_state=42` |
| Gaussian mixture | 2, 3, 4, 5, 6 | Both `covariance_type=diag/full`; `n_init=5`, `reg_covar=1e-4`, `max_iter=300`, `random_state=42` |

There are 20 candidate settings. Assign GMM's most probable component as the main cluster for comparison with the other methods, while also reporting maximum-membership probabilities. Model component probabilities are not probabilities of a true customer type.

## Evaluation and selection

For every candidate, report held-out silhouette; ARI on the same held-out customers after five customer-level resamples; monthly validation group sizes and smallest-group share; original-unit medians of the five features; convergence; and the actual number of clusters. For GMM, also report BIC and membership ambiguity. Silhouette uses identical evaluation rows and transformed feature space. Compare BIC only within GMM; K-means inertia and GMM BIC are not interchangeable.

Exclude candidates that fail to converge, fail to form the requested number of groups, or produce a validation-month group with fewer than 50 customers. Fifty is approximately 2% of a validation month's customers, avoiding tiny incidental groups as business units.

Then assess separation, resampling stability, and monthly profiles together. Extra groups should have distinctions reproducible across validation months. Prefer fewer groups and simpler methods when evidence is close. Report every candidate, selection reasons, and tradeoffs; do not combine unrelated metrics into an arbitrary total. If no stable, interpretable subdivision emerges, select the most reliable candidate and describe its exploratory scope.

This stage evaluates segmentation quality only. Future repurchase, spending, and product labels do not select the model. They may later support external business checks, but are not independent evidence from this comparison.
