# Item-Level Recommendation: Four Feature Groups × Three Models, v3.1

Version v3 stopped after 64 development fits because of classification sampling weights. Its plan, code, samples, and development results remain in `outputs/item_v3`; it did not reach this round's validation/review. Before validation, v3.1 fixed the inclusion-probability correction below and reused the same historical features and sampled rows. Other candidates, metrics, and thresholds are unchanged.

Version v1 and profile_v2 are preserved. Validation months supported earlier research, and September-November product outcomes were already inspected. This round uses January-May forward development, June-August selection, and a fixed September-November review. It does not restore a new independent test or change settings from review results. The other three project goals remain unchanged.

## Task, candidates, and labels

Retain monthly first-day analysis, the original eligible customers, preceding 180-day purchases, ten recommendations per customer, and distinct products in the first purchase event within the next 30 days. Merge invoices at the same timestamp.

Training uses customers with a next order and fully observed labels: `snapshot_date + 30 days <= fit_cutoff`. Label 0 means the item is absent from that next order, not that the customer dislikes it. Inference generates lists for all eligible customers.

Candidates are all products appearing in the 180 days before the phase's fitting date. Freeze the catalog within each phase. Products outside the catalog remain in evaluation denominators and cannot be added using future answers.

## Four feature groups

- **A, base:** previously purchased flag; distinct-invoice purchase counts over 180/90/30 days (`log1p`); time since latest/first visible product purchase; mean purchase interval and at-least-two-purchases flag; interval-due ratio; share of personal orders containing the product; product popularity over 30/180 days (`log1p`), popularity change, purchasing-customer count (`log1p`), and days observed within 180 days. Quantity and amount are not interaction strength. Customer/product IDs are not features; no category or true launch date is assumed.
- **B:** A + ten historical profile features + six interactions between profiles and purchased flags/counts/popularity. Keep missing history explicit. Profiles are constant across a customer's products; interactions also let linear models use them for ranking.
- **C:** A + per-customer EASE score normalized by the maximum absolute score in the catalog + missing-signal flag. Do not add all three recommenders or duplicate rank features.
- **D:** A + B's profile additions/interactions + C's signals + repeat-product share × EASE score.

All features use earlier transactions. Customer history and observable product statistics update monthly; global product associations remain fixed within a phase. Unpurchased products have zero counts and 180-day capped timing values with a purchased flag. Missing purchase intervals use a capped placeholder and validity flag. Fit median imputation for missing profiles on training data only.

## Historical EASE signals and sampling

From June 2010 onward, fit each month's EASE using only its preceding 180 days. Compare regularization 10/30/100/300/1000/3000/10000/30000. Select regularization from mean EASE NDCG on earlier months with fully observed 30-day labels; use maximum regularization if no earlier results are available. A month's answers affect later choices only after label maturity, never that month's inputs.

For June-August, fit EASE before June 1 and freeze it; for September-November, fit before September 1 and freeze it. Save fitting boundaries, parameter choices, and hashes for historical training, validation, and review.

Keep all in-catalog positives. Per customer-month, sample `min(120, max(50, 5 * positive_count))` negatives, or all if fewer exist. Half are hard negatives alternately drawn from EASE, personal purchase-count, and popularity rankings; sample the rest uniformly without replacement from remaining negatives. Use seed 42 with month/customer-specific sampling.

All feature groups and model families share exactly the same rows. Classification weights are 1 for positives and mandatory hard negatives; uniformly sampled negatives receive remaining-negative-count / sampled-count. Normalize total weight equally per customer-month. This corrects the effect of hard-negative oversampling on the full-catalog classification distribution.

Ranking uses the same sampled query groups with query normalization; its loss is not claimed to be an unbiased full-catalog LambdaRank estimate. Customer-months whose positives are all outside the catalog cannot form a positive training query; record them and keep them in evaluation. Sampled scores are not calibrated purchase probabilities.

Evaluation and inference score the complete catalog without negative sampling or answer-based prefiltering. Batched scoring produces exact top tens. Round scores to twelve decimals to remove floating-point batch differences; break ties by frozen catalog popularity and product code. Retain samples, query groups, and sampling audits as reproducible artifacts.

## Three models, budget, and parameter selection

- **LR:** training-median imputation, weighted standardization, L2 logistic regression; `C=0.01/0.1/1/10`, four settings, LBFGS up to 600 iterations. Apply inclusion correction and equal total customer-month weights; no additional class weighting.
- **LightGBM classification:** 7/15 leaves × 60/120 trees, four settings; binary objective, learning rate 0.05, minimum 100 samples per leaf, L2 = 5, seed 42, four threads, deterministic mode; equal customer-month weighting.
- **LightGBM ranking:** the same four tree settings; `lambdarank`, binary relevance, `label_gain=[0,1]`, truncation level 15, customer-month queries, and `lambdarank_norm` enabled. No validation early stopping.

For each model family and feature group, evaluate four configurations on five complete-catalog forward folds in January-May. To isolate feature contributions, choose one shared configuration per model family using equal-weight mean NDCG across four feature groups and five months. Break exact ties toward stronger regularization or fewer leaves/trees.

Twelve shortlisted versions therefore share parameters, samples, and preprocessing boundaries within a family. The configuration budget is equal; computation costs need not be.

## Earlier comparisons, acceptance, and attribution

Keep personal repeat purchase, pure EASE, and profile_v2's strongest profile hybrid, using frozen models on the same customers and candidates. The earlier fixed hybrid weight is 1 and duplicates personal repeat purchase, so do not count it again.

Rank the twelve new versions by validation NDCG against the existing personal-repeat default. Replacement requires mean NDCG, Recall, and unseen-product Recall gains of at least 0.01 each, with NDCG no worse in at least two of three months. Among eligible versions, choose highest NDCG, then fewer features/simpler models. If none qualifies, retain the default and report the gap.

For profile attribution, compare B-A and D-C within the same model and settings. For EASE attribution, compare C-A and D-B. Report all comparisons. A selected B/D version can claim additional profile value only with NDCG gain at least 0.002, no deterioration in at least two of three months, and Recall decrease no greater than 0.005. Otherwise distinguish improvement from item-level scoring from improvement attributable to profiles.

Run 500 paired customer-level bootstrap draws for the best new version versus personal repeat purchase and its with/without-profile comparison. These are post-selection internal diagnostics, not new independent significance evidence.

Report monthly NDCG/Recall/Hit/Precision@10 and @5, seen/unseen-product Recall, coverage, out-of-catalog answer share, and profile subgroups. Freeze validation selection and final models before calculating this round's September-November review. Do not revise settings, features, thresholds, or defaults from that review.

## Delivery and checks

Write to `artifacts/recommendation/current/`: locked plan/selection, historical samples and sampling/maturity audits, EASE temporal audits, 240 development-fit results, twelve validation/review versions plus earlier controls, same-parameter attribution, model bundle, fifteen recommendation methods' lists, report, and demo. Repeated runs verify complete artifacts; changed methods require a new version.

The original experiment interface shows profiles, product purchase counts, recency, recommendation priority, and source, with twelve versions and earlier comparisons available and ten-item exports. Scores are not calibrated probabilities; future orders and individual answer matches are excluded.

Extend existing tests for formulas, missing values, future isolation, EASE maturity boundaries, negative sampling, shared samples/parameters, full-catalog and batched scoring, labels/denominators, review-independent selection, saved model/list reproduction, pages, and exports. Do not add temporary validation scripts to the repository.

References: [LightGBM ranking](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html), [parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
