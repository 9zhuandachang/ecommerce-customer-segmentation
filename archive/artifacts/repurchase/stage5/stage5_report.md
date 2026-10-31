# Stage 5: Six groups of repurchase prediction experiments

The two sets of features use the same 37,019 customer-month records in the training period and 8,146 customer-month records in the validation period. A is the 8 basic transaction characteristics of stage four; B plus the three cluster center distances that have been fitted in the training period. LR and GBDT are each trained once, and the soft voting weight of each group is selected from 0.25, 0.50 and 0.75 using only the verification set AP. Test period metrics are not calculated.

| feature group | model | Verify AP | ROC-AUC | Precision@20% | Recall@20% |
| --- | --- | ---: | ---: | ---: | ---: |
| A_base_8 | LR | 0.5799 | 0.7423 | 0.5874 | 0.4401 |
| A_base_8 | GBDT | 0.5800 | 0.7442 | 0.5923 | 0.4437 |
| A_base_8 | SoftVote | 0.5916 | 0.7527 | 0.6039 | 0.4525 |
| B_base_8_plus_3_distances | LR | 0.5840 | 0.7484 | 0.5868 | 0.4396 |
| B_base_8_plus_3_distances | GBDT | 0.5769 | 0.7391 | 0.5917 | 0.4433 |
| B_base_8_plus_3_distances | SoftVote | 0.5899 | 0.7510 | 0.5953 | 0.4460 |
| baseline | rfm | 0.5105 | 0.7378 | 0.5579 | 0.4180 |

The weight selected for the validation period of LR in soft voting: A is 0.50 and B is 0.75.

Changes in AP after adding clustering distance (B minus A): LR + 0.0041; GBDT - 0.0031; SoftVote - 0.0017.

The Top 20% quota is calculated as ceil ( 0.2 × number of customers in the month) for each verification month, and then the number of hits is summarized. Results only represent ranking performance for the historical validation period; probability calibration, thresholds, and final testing are left to phase six.

See `validation_predictions.csv` for item-by-item verification scores; see `weight_selection.csv` for three candidate weights and selection values; see `run_config.json` for complete features, versions, and source hashes.
