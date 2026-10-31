# Stage 5: Re-comparison of repurchase models

This experiment compares four models and three sets of features on the same batch of customer monthly snapshots. The training period is 2010 - 06 to 2011 - 05, and the formal validation period is 2011 - 06 to 2011 - 08; the test label did not participate in this experiment.
The project has looked at this historical data before, so the results are a retrospective comparison and not a new independent verification.

Selection rules: First select the complexity for each combination in the three forward months of the training period, and then select the model according to the formal validation period AP; the candidate forward average AP and validation period Top 20% accuracy cannot be lower than the basic eight items LR.

| feature group | model | parameters | Forward average AP | Verify AP | Top 20% accuracy | ROC-AUC | Brier | Selection criteria | final choice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| base_8_plus_trend_and_rhythm | XGBoost | 2 | 0.5498 | 0.6009 | 0.6064 | 0.7601 | 0.1557 | Yes | Yes |
| base_8_plus_trend_and_rhythm | LightGBM | 7 | 0.5486 | 0.5997 | 0.6033 | 0.7569 | 0.1559 | Yes | No |
| base_8_plus_trend_and_rhythm | SplineLR | 1 | 0.5535 | 0.5953 | 0.6058 | 0.7557 | 0.1570 | Yes | No |
| base_8_plus_trend_and_rhythm | LR | 0.1 | 0.5521 | 0.5931 | 0.6002 | 0.7551 | 0.1573 | Yes | No |
| base_8_plus_trend | SplineLR | 0.1 | 0.5499 | 0.5907 | 0.5966 | 0.7538 | 0.1578 | Yes | No |
| base_8_plus_trend | LR | 0.1 | 0.5494 | 0.5892 | 0.5923 | 0.7535 | 0.1580 | Yes | No |
| base_8 | LR | 0.1 | 0.5460 | 0.5867 | 0.5874 | 0.7505 | 0.1587 | Yes | No |
| base_8_plus_trend | XGBoost | 3 | 0.5456 | 0.5962 | 0.6033 | 0.7559 | 0.1565 | No | No |
| base_8_plus_trend | LightGBM | 7 | 0.5457 | 0.5950 | 0.6015 | 0.7542 | 0.1567 | No | No |
| base_8 | XGBoost | 3 | 0.5415 | 0.5939 | 0.6033 | 0.7557 | 0.1568 | No | No |
| base_8 | LightGBM | 7 | 0.5412 | 0.5935 | 0.6033 | 0.7558 | 0.1568 | No | No |
| base_8 | SplineLR | 0.1 | 0.5458 | 0.5879 | 0.5917 | 0.7506 | 0.1585 | No | No |

**Selected: base_ 8 _plus_trend_and_rhythm / XGBoost, complexity parameter 2.**
Complete parameters of the selected model: `{"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 10, "reg_lambda": 3, "subsample": 0.9, "colsample_bytree": 1, "tree_method": "hist", "grow_policy": "depthwise", "random_state": 42}`.
Verification AP = 0.6009, Top 20% accuracy rate = 0.6064; average rating during the validation period = 0.2595, actual repurchase rate = 0.2672.

Afterwards, re-sample and check the AP difference according to customer groups (not involved in model selection):
| Comparison object | Observed AP difference | 95% resampling interval |
| --- | ---: | ---: |
| base_8_plus_trend_and_rhythm / LightGBM | +0.0013 | [-0.0014, +0.0036] |
| base_8 / LR | +0.0142 | [+0.0089, +0.0190] |
This range only reflects the resampling fluctuations of these validation clients; the models were selected during the same validation period, which was only three months long, so it is not a confidence guarantee of future performance.

See `monthly_validation.csv` for month-by-month results; see `forward_folds.csv` for each round of forward results; see `validation_predictions.csv` for all verification scores. The selected model is only fit in the training period and saved as `selected_model.joblib`. This experiment has not yet replaced the Stage Six or Dashboard models.

It is impossible to infer true marketing increments or future new data performance based on these historical indicators alone.
