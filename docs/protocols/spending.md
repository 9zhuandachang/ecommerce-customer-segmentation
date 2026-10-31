# Spending-Potential Ranking: Unified Model Comparison Protocol v1

This protocol was written before this comparison was run and its results were generated. Earlier experiments and lists are retained. Amount results for all project months had already been inspected, so this is a structured internal comparison rather than a new independent test.

## Fixed task and data

- On each month's first day, rank eligible snapshot customers with purchases in the preceding 180 days. Select `ceil(customer_count × 20%)`. The target, `spend_30d`, is eligible product spending over the following 30 days, excluding rejected returns, cancellations, and non-product transactions.
- All three learned approaches use the same 20 numerical historical features: eight snapshot features and twelve temporal features. Customer IDs support joins and tie-breaking. Future spending, repurchase labels, split labels, and dates are excluded from model inputs. No new features are added.
- The primary metric is mean monthly Top20 spending capture, giving each month equal weight. Also report pooled spending capture, the worst month, monthly differences, Top10/Top30 capture, and MAE. Individual amount accuracy is not this round's acceptance target.

## Temporal splits and freezing order

1. Tune on five forward development folds, January-May 2011. Each fold uses earlier snapshots whose full 30-day labels have ended. For example, February 1 labels end on March 3 and cannot train a March 1 model.
2. Select one configuration per learned approach by mean development-fold Top20 capture. Break ties toward fewer trees, fewer leaves, and more conservative regularization. Fit only the original `train` partition, June 2010-May 2011, then freeze and score June-August. Validation labels are excluded from fitting and early stopping.
3. Compare three learned approaches and three rules on June-August. Identify the strongest rule and learned approach. A learned approach replaces the rule only if its mean capture improves by at least 1 percentage point, it is no worse in at least two of three months, its worst monthly deficit is no more than 1 percentage point, and the lower bound of its paired customer-resampling interval is above zero. Otherwise, use the strongest rule. These are internal selection conditions, not independent-test acceptance or a significance claim.
4. Save parameters and the final choice in `locked_selection.json`. Save and hash both the validation model and the final model refitted on train + validation before calculating this round's September-November metrics. The final model stays fixed across those three months. Test labels are excluded from fitting, calibration, tuning, and selection. Previously viewed September-November results are a review, and do not change the winner.

## Candidates and budget

- Three rules: rank by spending in the preceding 30, 90, or 180 days. For supporting MAE, divide these amounts by 1, 3, or 6 to estimate 30-day spending; scaling does not affect ranks.
- LightGBM Tweedie: variance power 1.5; one model learns from both zero and positive amounts.
- XGBoost Tweedie: variance power 1.5; the same inputs and target compare another implementation.
- Two-stage LightGBM: fit a purchase classifier on all training customers and a Gamma amount regressor on actual purchasers. Rank by their product.
- Each learned approach has eight configurations: 120/240 regression trees × 15/31 maximum leaves × standard/conservative regularization, across five folds. Learning rate is 0.05, seed 42, and one thread. Standard LightGBM retains the earlier single-model settings. Conservative regularization increases minimum leaf samples and L2. XGBoost uses corresponding leaf constraints and its own minimum child-weight parameter. The two-stage classifier uses two-thirds as many trees as the regressor, with a minimum of 80. Configuration counts are equal; the two-stage approach fits more individual models, and training time is reported separately.
- The program writes parameters and source hashes to `locked_plan.json`. A changed plan, code, or input cannot silently reuse the original output directory.

## Robustness and delivery

- Run 500 paired customer-cluster bootstrap draws for the strongest learned approach and rule. Sample each customer's records across all three validation months together, and reselect the top 20% within every draw. Use seed 42. These post-selection intervals describe customer-composition variation in these months, not guarantees for future months or selection-free significance.
- Refit the development-selected configurations with seeds 7 and 2026 and report validation results for all three learned approaches. These are diagnostics and do not change selection.
- Report all rules, candidate settings, monthly results, temporal boundaries, selection reasons, and the absence of a new independent test. Deliver the rule if it wins.
- Display the selected method, rank, and within-month percentile. Exports exclude future answers and individual amount estimates. Existing segmentation, repurchase, and recommendation workflows remain unchanged.
- Extend existing regression tests and run project and page checks. Preserve earlier amount experiments and priority-list artifacts.
