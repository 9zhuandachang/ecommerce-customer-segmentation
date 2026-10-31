#Consumption forecast under purchase conditions: development period results

The question is "How much will this customer spend if they make a purchase 30 days in the future?"
During evaluation, the error is only calculated for customers who actually made purchases; the actual purchase status does not enter the prediction features.
This value cannot be used as a proxy for future spending by all customers or as a substitute for unconditional forecasts of buyer error.
The January-May and the already viewed June-August are only used for development analysis; the September-November are not used for model comparison.

| period | method | Buyer MAE | Buyer amount deviation |
| --- | --- | ---: | ---: |
| rolling_jan_may | historic_monthly_rate | 396.56 | -19.96% |
| rolling_jan_may | average_prior_invoice | 360.23 | -28.97% |
| rolling_jan_may | conditional_gamma_mean | 355.84 | +5.65% |
| rolling_jan_may | conditional_l1_median | 334.51 | -5.20% |
| known_jun_aug | historic_monthly_rate | 427.12 | -39.71% |
| known_jun_aug | average_prior_invoice | 410.44 | -38.24% |
| known_jun_aug | conditional_gamma_mean | 358.63 | -7.17% |
| known_jun_aug | conditional_l1_median | 352.18 | -18.08% |
| rolling_jan_aug | historic_monthly_rate | 408.46 | -28.02% |
| rolling_jan_aug | average_prior_invoice | 379.78 | -32.75% |
| rolling_jan_aug | conditional_gamma_mean | 356.93 | +0.42% |
| rolling_jan_aug | conditional_l1_median | 341.39 | -10.45% |

The historical monthly consumption rate and the average order amount of the last 180 day are simple comparisons using only the data before the snapshot.
L 1 The median of model prediction conditions, suitable for minimizing the absolute error of buyers; its total amount is usually low.
The Gamma model predicts the conditional mean, and the amount deviation is more suitable as a component of the expected amount.
Month-by-month and per-customer output and input hashes are found in the same directory.
