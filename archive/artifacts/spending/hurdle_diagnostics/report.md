# Two-stage amount model error source audit

This report uses the fixed parameters of the existing two-stage model; 2011, January-August is evaluated forward in time, September-November line is not involved.
The purchase probability, purchase condition mean and median are only predicted by the pre-cutoff features. Actual purchase status is only used for post-assessment purposes.

`Forecast total−actual total = [Σ(p×m)−Σ(I×m)] + [Σ(I×m)−Σactual amount]`. The former item is the part where the purchase occurred, and the latter item is the part of the purchaser's amount; the sum of the two items is exactly equal to the total error.

| period | Amount characteristics | AP | Deviation in forecasting the number of buyers | Purchase occurred part | Buyer's amount | total deviation | Median buyer MAE / Historical average order price MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | base_20 | 0.5201 | +28.79% | +19.63% | +5.65% | +25.28% | 334.51 / 360.23 |
| rolling_jan_may | amount_plus_prior_year | 0.5201 | +28.79% | +17.59% | +0.94% | +18.53% | 329.21 / 360.23 |
| rolling_jan_may | base_20_buyer_lag_calibrated | 0.5201 | +28.79% | +19.04% | -0.23% | +18.81% | 334.51 / 360.23 |
| rolling_jan_may | prior_year_buyer_lag_calibrated | 0.5201 | +28.79% | +17.14% | -3.74% | +13.40% | 329.21 / 360.23 |
| rolling_jan_may | base_20_recent_weighted | 0.5201 | +28.79% | +19.22% | +4.26% | +23.48% | 334.51 / 360.23 |
| rolling_jan_may | base_20_isotonic_buyer_calibrated | 0.5201 | +28.79% | +18.96% | +0.16% | +19.12% | 334.51 / 360.23 |
| rolling_jan_may | base_20_l2_mean | 0.5201 | +28.79% | +21.23% | +18.50% | +39.73% | 334.51 / 360.23 |
| rolling_jan_may | base_20_segmented_buyer_lag_calibrated | 0.5201 | +28.79% | +19.10% | -0.14% | +18.96% | 334.51 / 360.23 |
| known_jun_aug | base_20 | 0.6106 | +0.65% | -1.15% | -7.17% | -8.32% | 352.18 / 410.44 |
| known_jun_aug | amount_plus_prior_year | 0.6106 | +0.65% | -1.19% | -11.17% | -12.37% | 347.27 / 410.44 |
| known_jun_aug | base_20_buyer_lag_calibrated | 0.6106 | +0.65% | -1.16% | -3.98% | -5.14% | 352.18 / 410.44 |
| known_jun_aug | prior_year_buyer_lag_calibrated | 0.6106 | +0.65% | -1.24% | -5.68% | -6.92% | 347.27 / 410.44 |
| known_jun_aug | base_20_recent_weighted | 0.6106 | +0.65% | -1.15% | -8.59% | -9.74% | 352.18 / 410.44 |
| known_jun_aug | base_20_isotonic_buyer_calibrated | 0.6106 | +0.65% | -1.39% | -3.74% | -5.13% | 352.18 / 410.44 |
| known_jun_aug | base_20_l2_mean | 0.6106 | +0.65% | -1.34% | +2.07% | +0.72% | 352.18 / 410.44 |
| known_jun_aug | base_20_segmented_buyer_lag_calibrated | 0.6106 | +0.65% | -1.27% | -3.67% | -4.94% | 352.18 / 410.44 |
| rolling_jan_aug | base_20 | 0.5510 | +17.83% | +11.15% | +0.42% | +11.57% | 341.39 / 379.78 |
| rolling_jan_aug | amount_plus_prior_year | 0.5510 | +17.83% | +9.92% | -4.00% | +5.92% | 336.24 / 379.78 |
| rolling_jan_aug | base_20_buyer_lag_calibrated | 0.5510 | +17.83% | +10.80% | -1.76% | +9.04% | 341.39 / 379.78 |
| rolling_jan_aug | prior_year_buyer_lag_calibrated | 0.5510 | +17.83% | +9.64% | -4.53% | +5.11% | 336.24 / 379.78 |
| rolling_jan_aug | base_20_recent_weighted | 0.5510 | +17.83% | +10.91% | -0.99% | +9.92% | 341.39 / 379.78 |
| rolling_jan_aug | base_20_isotonic_buyer_calibrated | 0.5510 | +17.83% | +10.65% | -1.43% | +9.22% | 341.39 / 379.78 |
| rolling_jan_aug | base_20_l2_mean | 0.5510 | +17.83% | +12.02% | +11.79% | +23.81% | 341.39 / 379.78 |
| rolling_jan_aug | base_20_segmented_buyer_lag_calibrated | 0.5510 | +17.83% | +10.79% | -1.58% | +9.21% | 341.39 / 379.78 |

## Monthly Diagnosis

| cutoff point | Amount characteristics | Actual/forecast purchases | AP | Purchase occurred part | Buyer's amount | total deviation |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2011-01-01 | base_20 | 606 / 1109 | 0.5169 | +70.22% | +21.02% | +91.23% |
| 2011-01-01 | base_20_buyer_lag_calibrated | 606 / 1109 | 0.5169 | +70.22% | +21.02% | +91.23% |
| 2011-01-01 | base_20_segmented_buyer_lag_calibrated | 606 / 1109 | 0.5169 | +70.22% | +21.02% | +91.23% |
| 2011-01-01 | base_20_isotonic_buyer_calibrated | 606 / 1109 | 0.5169 | +70.22% | +21.02% | +91.23% |
| 2011-01-01 | amount_plus_prior_year | 606 / 1109 | 0.5169 | +70.22% | +21.02% | +91.23% |
| 2011-01-01 | prior_year_buyer_lag_calibrated | 606 / 1109 | 0.5169 | +70.22% | +21.02% | +91.23% |
| 2011-01-01 | base_20_recent_weighted | 606 / 1109 | 0.5169 | +68.58% | +19.48% | +88.06% |
| 2011-01-01 | base_20_l2_mean | 606 / 1109 | 0.5169 | +81.51% | +39.50% | +121.01% |
| 2011-02-01 | base_20 | 635 / 885 | 0.5141 | +24.88% | +10.23% | +35.11% |
| 2011-02-01 | base_20_buyer_lag_calibrated | 635 / 885 | 0.5141 | +22.72% | +0.66% | +23.38% |
| 2011-02-01 | base_20_segmented_buyer_lag_calibrated | 635 / 885 | 0.5141 | +22.89% | +0.73% | +23.63% |
| 2011-02-01 | base_20_isotonic_buyer_calibrated | 635 / 885 | 0.5141 | +21.52% | +3.50% | +25.01% |
| 2011-02-01 | amount_plus_prior_year | 635 / 885 | 0.5141 | +20.33% | +3.50% | +23.84% |
| 2011-02-01 | prior_year_buyer_lag_calibrated | 635 / 885 | 0.5141 | +18.57% | -5.48% | +13.08% |
| 2011-02-01 | base_20_recent_weighted | 635 / 885 | 0.5141 | +24.91% | +9.66% | +34.57% |
| 2011-02-01 | base_20_l2_mean | 635 / 885 | 0.5141 | +24.31% | +26.65% | +50.96% |
| 2011-03-01 | base_20 | 699 / 865 | 0.5211 | +7.93% | +8.68% | +16.62% |
| 2011-03-01 | base_20_buyer_lag_calibrated | 699 / 865 | 0.5211 | +7.24% | -0.75% | +6.49% |
| 2011-03-01 | base_20_segmented_buyer_lag_calibrated | 699 / 865 | 0.5211 | +7.37% | -0.66% | +6.71% |
| 2011-03-01 | base_20_isotonic_buyer_calibrated | 699 / 865 | 0.5211 | +7.73% | -2.31% | +5.41% |
| 2011-03-01 | amount_plus_prior_year | 699 / 865 | 0.5211 | +4.26% | +3.43% | +7.69% |
| 2011-03-01 | prior_year_buyer_lag_calibrated | 699 / 865 | 0.5211 | +3.89% | -5.55% | -1.66% |
| 2011-03-01 | base_20_recent_weighted | 699 / 865 | 0.5211 | +7.80% | +9.37% | +17.17% |
| 2011-03-01 | base_20_l2_mean | 699 / 865 | 0.5211 | +6.97% | +25.81% | +32.78% |
| 2011-04-01 | base_20 | 662 / 804 | 0.5338 | +13.90% | +13.80% | +27.70% |
| 2011-04-01 | base_20_buyer_lag_calibrated | 662 / 804 | 0.5338 | +13.10% | +7.20% | +20.30% |
| 2011-04-01 | base_20_segmented_buyer_lag_calibrated | 662 / 804 | 0.5338 | +13.17% | +7.24% | +20.41% |
| 2011-04-01 | base_20_isotonic_buyer_calibrated | 662 / 804 | 0.5338 | +13.13% | +7.47% | +20.60% |
| 2011-04-01 | amount_plus_prior_year | 662 / 804 | 0.5338 | +12.19% | +5.57% | +17.76% |
| 2011-04-01 | prior_year_buyer_lag_calibrated | 662 / 804 | 0.5338 | +11.69% | +1.18% | +12.87% |
| 2011-04-01 | base_20_recent_weighted | 662 / 804 | 0.5338 | +13.18% | +11.83% | +25.01% |
| 2011-04-01 | base_20_l2_mean | 662 / 804 | 0.5338 | +14.36% | +22.13% | +36.49% |
| 2011-05-01 | base_20 | 812 / 733 | 0.6129 | -7.92% | -17.32% | -25.24% |
| 2011-05-01 | base_20_buyer_lag_calibrated | 812 / 733 | 0.6129 | -7.54% | -21.33% | -28.87% |
| 2011-05-01 | base_20_segmented_buyer_lag_calibrated | 812 / 733 | 0.6129 | -7.59% | -21.13% | -28.72% |
| 2011-05-01 | base_20_isotonic_buyer_calibrated | 812 / 733 | 0.6129 | -7.40% | -20.76% | -28.16% |
| 2011-05-01 | amount_plus_prior_year | 812 / 733 | 0.6129 | -8.47% | -21.10% | -29.58% |
| 2011-05-01 | prior_year_buyer_lag_calibrated | 812 / 733 | 0.6129 | -8.31% | -22.66% | -30.97% |
| 2011-05-01 | base_20_recent_weighted | 812 / 733 | 0.6129 | -7.76% | -20.57% | -28.33% |
| 2011-05-01 | base_20_l2_mean | 812 / 733 | 0.6129 | -8.93% | -11.72% | -20.66% |
| 2011-06-01 | base_20 | 745 / 703 | 0.6147 | -2.51% | -13.40% | -15.91% |
| 2011-06-01 | base_20_buyer_lag_calibrated | 745 / 703 | 0.6147 | -2.51% | -13.33% | -15.83% |
| 2011-06-01 | base_20_segmented_buyer_lag_calibrated | 745 / 703 | 0.6147 | -2.48% | -13.37% | -15.86% |
| 2011-06-01 | base_20_isotonic_buyer_calibrated | 745 / 703 | 0.6147 | -2.68% | -12.84% | -15.52% |
| 2011-06-01 | amount_plus_prior_year | 745 / 703 | 0.6147 | -2.50% | -16.21% | -18.71% |
| 2011-06-01 | prior_year_buyer_lag_calibrated | 745 / 703 | 0.6147 | -2.57% | -13.69% | -16.26% |
| 2011-06-01 | base_20_recent_weighted | 745 / 703 | 0.6147 | -2.43% | -15.17% | -17.59% |
| 2011-06-01 | base_20_l2_mean | 745 / 703 | 0.6147 | -1.47% | -8.44% | -9.92% |
| 2011-07-01 | base_20 | 717 / 737 | 0.6016 | -2.27% | +0.82% | -1.45% |
| 2011-07-01 | base_20_buyer_lag_calibrated | 717 / 737 | 0.6016 | -2.37% | +5.04% | +2.67% |
| 2011-07-01 | base_20_segmented_buyer_lag_calibrated | 717 / 737 | 0.6016 | -2.60% | +5.54% | +2.94% |
| 2011-07-01 | base_20_isotonic_buyer_calibrated | 717 / 737 | 0.6016 | -2.56% | +4.23% | +1.67% |
| 2011-07-01 | amount_plus_prior_year | 717 / 737 | 0.6016 | -2.38% | -4.42% | -6.80% |
| 2011-07-01 | prior_year_buyer_lag_calibrated | 717 / 737 | 0.6016 | -2.55% | +2.30% | -0.25% |
| 2011-07-01 | base_20_recent_weighted | 717 / 737 | 0.6016 | -2.57% | +0.60% | -1.97% |
| 2011-07-01 | base_20_l2_mean | 717 / 737 | 0.6016 | -3.87% | +16.65% | +12.79% |
| 2011-08-01 | base_20 | 715 / 751 | 0.6178 | +1.20% | -8.31% | -7.10% |
| 2011-08-01 | base_20_buyer_lag_calibrated | 715 / 751 | 0.6178 | +1.27% | -2.99% | -1.72% |
| 2011-08-01 | base_20_segmented_buyer_lag_calibrated | 715 / 751 | 0.6178 | +1.14% | -2.49% | -1.35% |
| 2011-08-01 | base_20_isotonic_buyer_calibrated | 715 / 751 | 0.6178 | +0.95% | -2.04% | -1.09% |
| 2011-08-01 | amount_plus_prior_year | 715 / 751 | 0.6178 | +1.18% | -12.37% | -11.20% |
| 2011-08-01 | prior_year_buyer_lag_calibrated | 715 / 751 | 0.6178 | +1.27% | -5.06% | -3.79% |
| 2011-08-01 | base_20_recent_weighted | 715 / 751 | 0.6178 | +1.41% | -10.49% | -9.08% |
| 2011-08-01 | base_20_l2_mean | 715 / 751 | 0.6178 | +1.09% | -0.87% | +0.22% |

## High-volume buyer tail audit (original two-stage model)

The highest monthly fee among actual buyers is approximately 1%; the actual amount is only used for post-auditing.

| cutoff point | Number of people at the end | Accounting for the actual amount of the month | Enter prediction Top 20 | Tail forecast amount/actual amount |
| --- | ---: | ---: | ---: | ---: |
| 2011-01-01 | 7 | 21.52% | 7 / 7 | 38.32% |
| 2011-02-01 | 7 | 19.27% | 7 / 7 | 62.05% |
| 2011-03-01 | 7 | 19.47% | 7 / 7 | 49.51% |
| 2011-04-01 | 7 | 15.09% | 6 / 7 | 37.21% |
| 2011-05-01 | 9 | 20.30% | 9 / 9 | 43.25% |
| 2011-06-01 | 8 | 28.27% | 8 / 8 | 33.31% |
| 2011-07-01 | 8 | 19.49% | 8 / 8 | 60.24% |
| 2011-08-01 | 8 | 28.01% | 8 / 8 | 33.51% |

This decomposition is used to locate sources of error and does not equate to any reinforcement target being passed. The original target is still based on `optimization_protocol.md` and future credible data.
