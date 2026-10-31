# Buyer amount error orientation experiment: lock before running

Recorded before viewing any run results for this candidate. This experiment is a model study of the viewed development months of the original merchant 2011. It does not generate new independent tests, nor does it change the enhanced challenge goals of `optimization_protocol.md`.

## Problem and only candidate

Previously `ranking_audit` showed that the 20 feature direct amount model reduced MAE for all customers, but recently the amount of MAE for actual buyers has worsened. Fixed test for a change with a clear cost meaning: 2 is given to snapshots with a purchase amount greater than zero in the future 30 days that are fully mature during training, and 1 is given to snapshots with a purchase amount of zero. This multiple only expresses the research and development hypothesis that "the buyer's amount error deserves double the attention" and does not search from the viewed month. The future purchase status of current customers is not used when forecasting, and the 20% definition of the forecast list is not changed.

The candidate fully shares 20 pre-cutoff features, LightGBM Tweedie targets, 120 trees, learning rate 0.05, 15 leaves, variance power 1.5, random seed 42 and per-fold mature label filtering with existing `extended_tweedie`; the only difference is the training sample weight. The comparison also includes `historic_180d` from the same customer. No new features or additional post-processing for the same period last year are added so that the effect of weighting can be judged independently.

## Assessment and Discipline

- Generate monthly predictions for 2011, January-August forward in time, and only train the historical snapshot of `snapshot_date + 30 day <= target snapshot day` at each fold. The January-May are for rolling development, the June-August have been viewed previously and are only for development confirmation; the September-November are not used to select weights or models for this experiment.
- Report the overall MAE, actual buyers MAE, pre-forecast 20% customer actual consumption coverage, overall forecast total deviation, Top 20 group deviation and zero consumption ratio for the same group of customers every month and two periods; while retaining customer-level forecasts, training deadlines and source file hashes.
- Whether it is worth entering the **new external fixed test**, first look at January-May: Buyer MAE is at least 5% lower than the unweighted model, the overall MAE increase relative to the unweighted model does not exceed 2%, Top 20 consumption coverage does not decrease, and the absolute value of the forecast total deviation does not expand. June-August only check whether the judgment is fragile across months, and do not re-select the multiple. This R&D promotion threshold is not the final acceptance, and test failure must not be used as a reason to change the threshold afterwards.
- All unfulfilled indicators will be retained truthfully. Only new trusted unviewed data that meets the original hardening goals will be considered for system access.
