# Phase five expansion: behavioral characteristics, time rolling verification and LightGBM adjustment

The original six sets of experiments and their files are retained. Added group C: 8 basic characteristics + 4 known purchasing rhythm characteristics as of the snapshot date, comparing a total of 9 group models. Records shorter than 90 days do not count order changes, and missing values ​​are only filled with the median of the corresponding training part.

Within the training period, a rolling verification is performed with 2011 years, 3, 4, and 5 months; each round clears the 30 day training labels that span the starting point of the month, and refits the filling, scaling, three groups K-means (Group B) and classifiers. LightGBM is left inside each round of training period for another month, comparing 15 / 31 leaves, up to 500 trees, continuous 30 rounds of internal verification, stopping early when AP does not improve; then retraining with the complete training part of the round, and validating the monthly score in the outer layer. Use the outer average of three rounds AP to select the number of leaves, and use the median of the best number of rounds for the selected setting to determine the final tree number. Soft voting weights are also selected based on the three-round average AP. These rolling indicators participate in the selection of parameters and cannot be regarded as independent test results.

| feature group | model | Rolling AP mean | Official verification AP | Official Precision@ 20% |
| --- | --- | ---: | ---: | ---: |
| A_base_8 | LR | 0.5354 | 0.5799 | 0.5874 |
| A_base_8 | GBDT | 0.5394 | 0.5946 | 0.6027 |
| A_base_8 | SoftVote | 0.5470 | 0.5921 | 0.6052 |
| B_base_8_plus_3_distances | LR | 0.5424 | 0.5840 | 0.5868 |
| B_base_8_plus_3_distances | GBDT | 0.5436 | 0.5903 | 0.6045 |
| B_base_8_plus_3_distances | SoftVote | 0.5480 | 0.5905 | 0.5990 |
| C_base_8_plus_4_behavior | LR | 0.5388 | 0.5835 | 0.5868 |
| C_base_8_plus_4_behavior | GBDT | 0.5413 | 0.5940 | 0.6027 |
| C_base_8_plus_4_behavior | SoftVote | 0.5496 | 0.5933 | 0.6033 |
| baseline | rfm | 0.4720 | 0.5105 | 0.5579 |

Rolling validation of selected LightGBM parameters with soft voting weights:

| feature group | number of leaves | Number of trees | LR weight |
| --- | ---: | ---: | ---: |
| A_base_8 | 15 | 64 | 0.50 |
| B_base_8_plus_3_distances | 15 | 47 | 0.50 |
| C_base_8_plus_4_behavior | 15 | 57 | 0.50 |

Among the original six groups, the highest verified AP is A_base_ 8 / SoftVote: 0.5916. The current highest value in the extended experiment is A_base_ 8 / GBDT: 0.5946, difference + 0.0030. The highest AP in group C is 0.5940, which does not exceed the highest value in the extended experiment. These are validation period results and final testing has not yet been conducted.

The official validation period is 2011, and 6 — 8 months, and is only evaluated after rolling verification of selected configurations. The test period after 9 months was not used for parameter adjustment or calculation of indicators in this report. See `monthly_validation.csv` for month-by-month results, `rolling_folds.csv`, `lightgbm_tuning.csv`, and `weight_candidates.csv` for each fold and candidate parameters, and `validation_predictions.csv` for item-by-item ratings. If there is no improvement with new features or adjusted models, the results should prevail.
