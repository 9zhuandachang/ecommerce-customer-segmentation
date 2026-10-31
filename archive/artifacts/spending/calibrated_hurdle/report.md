# Probability of repurchase × Amount after purchase: Development period experiment

The classifier reuses the A group settings selected in stage five and uses only mature history calibration according to the stage six method; the conditional amount still uses 20 features before the cutoff point. January-May have been used to select repurchase model parameters, and June-August have also been viewed; neither is an independent amount test. 9 – 11 did not participate in this experiment for months.

| period | method | All MAE | Buyer MAE | Top 20 consumption coverage | total deviation | Top 20 Group bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | extended_two_stage | 175.13 | 412.63 | 72.17% | +25.28% | +12.60% |
| rolling_jan_may | stage6_probability_times_gamma | 163.29 | 431.91 | 71.62% | +6.44% | -0.00% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | extended_two_stage | 188.85 | 439.10 | 73.19% | -8.32% | -14.54% |
| known_jun_aug | stage6_probability_times_gamma | 189.76 | 452.17 | 71.94% | -11.26% | -14.86% |

| period | Buy AP | Buy Brier | Deviation in forecasting the number of buyers |
| --- | ---: | ---: | ---: |
| rolling_jan_may | 0.4996 | 0.1388 | +6.08% |
| known_jun_aug | 0.5751 | 0.1606 | -3.44% |

## Fixed in advance January-Mayly advancement conditions

- `buyer_mae_reduction_at_least_5pct`: Not met
- `overall_mae_increase_at_most_2pct`: satisfied
- `top20_capture_not_lower`: satisfied
- `aggregate_bias_abs_not_larger`: satisfied

Whether it is worth entering into a new independent examination: **No**. Development phase results may only be used for research; they will not be included in the dashboard.
Month-by-month results, development-period-by-customer forecasts, training and calibration label cutoffs, and input hashes are listed in the same directory.
