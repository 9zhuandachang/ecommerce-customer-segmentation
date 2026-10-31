# Stage 5: Prespecified Repurchase Model Comparison

## Task and data

On each month's first day, use eligible purchases in the preceding 180 days to predict whether a customer makes an eligible purchase in the next 30 days. Compare the same customer-month snapshots: training June 2010-May 2011; formal validation June-August 2011. September-November test labels are excluded from this experiment's fitting, tuning, selection, and metric calculations. Earlier project stages have inspected these historical data; this is a retrospective comparison within the same dataset, not newly obtained independent evidence.

## Candidate inputs

1. `base_8`: the eight existing historical features.
2. `base_8_plus_trend`: add the 90-day/180-day invoice ratio, the change between the latest 30-day order count and the preceding 60-day monthly average, and a flag for at least 90 days of visible history. For shorter histories, set the change to zero and use the flag to indicate an incomplete trend window.
3. `base_8_plus_trend_and_rhythm`: also add the number of active 30-day periods within 180 days, the gap between the last two invoices, and a flag for at least two invoices. For one-invoice customers, use a 180-day placeholder gap and preserve its validity flag. Invoice features use only cleaned purchases in the preceding 180 days. Verify each snapshot's invoice count against the base snapshot.

Manual RFM scores, old three-cluster distances, and cluster labels are excluded. The underlying recency, frequency, and monetary information remains in the base eight features. Derived features cannot use future labels. Fit all preprocessing on the current fold's training data only.

## Models and selection

Compare four models: L2 logistic regression; logistic regression with a cubic spline using four quantile knots for `recency_days`; depth-limited XGBoost with `hist`; and LightGBM with few leaves. Logistic regression applies `log1p` to right-skewed non-negative amounts/counts, then standardizes features. Trees use original values. About one-quarter of labels are positive; class weighting is not added automatically. Use `random_state=42` and CPU tree training.

Each model has two prespecified complexity settings: `C=0.1,1` for LR/spline LR, `max_depth=2,3` for XGBoost, and `num_leaves=7,15` for LightGBM. Both tree families use 300 rounds and learning rate 0.05; other regularization settings are in the implementation.

First perform forward validation on March, April, and May 2011. Each fold trains only on earlier snapshots with fully completed label windows. For each feature-group/model pair, select the complexity with the highest three-fold mean AP; break ties toward simpler settings. Refit on the complete training period and compare the twelve candidates on June-August.

Select by validation AP, subject to two conditions: forward-fold mean AP and validation Precision@20% must both be at least as high as base-eight-feature LR. If the highest-AP candidate fails, consider the next eligible candidate; base LR remains available. AP pools validation customer-month records. Select the top 20% separately within each month before combining lists. Report monthly AP, Precision@20%, ROC-AUC, Brier, mean score, and actual repurchase rate. Close results do not establish an inherent algorithm advantage.

This experiment writes separate comparison artifacts and does not modify the frozen Stage 6 model or dashboard. Subsequent stages review temporal stability, calibration, and deployment boundaries.
