# CDNOW cross-merchant consumption prediction test

Use complete real transaction sources according to `cdnow_protocol.md` 's pre-fixed rules. The original files are only placed in the system temporary directory, and only the aggregation results are retained in the warehouse.
Amounts are in USD, customers are only from first-time buyers in the first quarter of year 1997; item IDs are not available, item diversity is fixed at 1. This result cannot replace the original merchant's independent acceptance in the new month.

The source transaction line is 69,659, and the customer is 23,570; after excluding the zero-amount transaction line 80, there are 23,502 qualified customers, and the customer monthly snapshot is 109,629. The sum of the amounts of zero-amount lines is 0, and the transaction amount is verified and passed point by point.
Last three month viability (≥ 1,000 snapshots, ≥ 100 future buyers per month): Satisfied.

| cutoff point | Partition | method | Number of customers | buyer | MAE | Buyer MAE | Top 20 coverage | total deviation | Top 20 Group bias |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1997-10-01 | development | historic_180d | 7,249 | 1,318 | 16.28 | 34.40 | 54.73% | +51.31% | +67.74% |
| 1997-10-01 | development | recent_90d | 7,249 | 1,318 | 15.27 | 36.98 | 53.07% | +35.68% | +85.80% |
| 1997-10-01 | development | extended_tweedie | 7,249 | 1,318 | 12.76 | 40.31 | 56.11% | -16.04% | -11.73% |
| 1997-10-01 | development | extended_tweedie_lag_calibrated | 7,249 | 1,318 | 12.76 | 40.31 | 56.11% | -16.04% | -11.73% |
| 1997-11-01 | development | historic_180d | 6,734 | 1,469 | 18.16 | 41.81 | 53.34% | +9.04% | +23.57% |
| 1997-11-01 | development | recent_90d | 6,734 | 1,469 | 16.80 | 43.67 | 51.96% | -5.61% | +30.64% |
| 1997-11-01 | development | extended_tweedie | 6,734 | 1,469 | 15.95 | 45.86 | 56.53% | -24.96% | -20.24% |
| 1997-11-01 | development | extended_tweedie_lag_calibrated | 6,734 | 1,469 | 16.34 | 45.04 | 56.53% | -17.79% | -12.62% |
| 1997-12-01 | development | historic_180d | 6,612 | 1,340 | 17.77 | 38.47 | 50.04% | +36.71% | +65.65% |
| 1997-12-01 | development | recent_90d | 6,612 | 1,340 | 17.04 | 40.00 | 48.37% | +30.87% | +90.44% |
| 1997-12-01 | development | extended_tweedie | 6,612 | 1,340 | 16.11 | 39.67 | 51.40% | +12.92% | +21.95% |
| 1997-12-01 | development | extended_tweedie_lag_calibrated | 6,612 | 1,340 | 17.06 | 39.05 | 51.40% | +27.96% | +38.19% |
| 1998-01-01 | development | historic_180d | 6,343 | 1,099 | 16.39 | 34.76 | 58.75% | +68.28% | +71.58% |
| 1998-01-01 | development | recent_90d | 6,343 | 1,099 | 16.69 | 37.29 | 53.40% | +71.10% | +120.09% |
| 1998-01-01 | development | extended_tweedie | 6,343 | 1,099 | 14.35 | 35.30 | 60.06% | +30.86% | +13.16% |
| 1998-01-01 | development | extended_tweedie_lag_calibrated | 6,343 | 1,099 | 14.74 | 34.81 | 60.06% | +38.47% | +19.73% |
| 1998-02-01 | development | historic_180d | 5,961 | 1,209 | 17.01 | 37.01 | 53.64% | +35.00% | +50.71% |
| 1998-02-01 | development | recent_90d | 5,961 | 1,209 | 17.72 | 38.29 | 50.63% | +43.45% | +93.39% |
| 1998-02-01 | development | extended_tweedie | 5,961 | 1,209 | 14.03 | 40.54 | 56.19% | -15.20% | -9.81% |
| 1998-02-01 | development | extended_tweedie_lag_calibrated | 5,961 | 1,209 | 13.99 | 40.63 | 56.19% | -16.12% | -10.79% |
| 1998-03-01 | development | historic_180d | 5,937 | 1,430 | 17.95 | 37.92 | 53.01% | +8.91% | +23.32% |
| 1998-03-01 | development | recent_90d | 5,937 | 1,430 | 17.90 | 41.00 | 48.98% | +1.72% | +45.48% |
| 1998-03-01 | development | extended_tweedie | 5,937 | 1,430 | 15.29 | 43.09 | 55.49% | -36.01% | -29.01% |
| 1998-03-01 | development | extended_tweedie_lag_calibrated | 5,937 | 1,430 | 15.26 | 43.20 | 55.49% | -36.70% | -29.79% |
| 1998-04-01 | locked_test | historic_180d | 6,055 | 1,074 | 16.08 | 32.82 | 57.46% | +74.48% | +83.05% |
| 1998-04-01 | locked_test | recent_90d | 6,055 | 1,074 | 15.47 | 34.41 | 54.37% | +64.20% | +111.86% |
| 1998-04-01 | locked_test | extended_tweedie | 6,055 | 1,074 | 14.21 | 32.59 | 59.91% | +43.62% | +34.52% |
| 1998-04-01 | locked_test | extended_tweedie_lag_calibrated | 6,055 | 1,074 | 14.67 | 32.35 | 59.91% | +52.09% | +42.44% |
| 1998-05-01 | locked_test | historic_180d | 5,900 | 1,136 | 15.86 | 31.68 | 55.65% | +59.44% | +72.68% |
| 1998-05-01 | locked_test | recent_90d | 5,900 | 1,136 | 15.36 | 33.26 | 52.13% | +50.81% | +101.27% |
| 1998-05-01 | locked_test | extended_tweedie | 5,900 | 1,136 | 12.63 | 35.81 | 58.00% | -3.84% | -2.37% |
| 1998-05-01 | locked_test | extended_tweedie_lag_calibrated | 5,900 | 1,136 | 12.80 | 35.41 | 58.00% | +0.36% | +1.89% |
| 1998-06-01 | locked_test | historic_180d | 5,529 | 1,089 | 16.75 | 37.15 | 55.31% | +33.68% | +44.54% |
| 1998-06-01 | locked_test | recent_90d | 5,529 | 1,089 | 16.50 | 38.20 | 54.71% | +31.09% | +65.17% |
| 1998-06-01 | locked_test | extended_tweedie | 5,529 | 1,089 | 13.79 | 40.16 | 58.58% | -13.35% | -12.24% |
| 1998-06-01 | locked_test | extended_tweedie_lag_calibrated | 5,529 | 1,089 | 13.87 | 39.92 | 58.58% | -11.39% | -10.25% |

## Last three months combined

| method | MAE | Buyer MAE | Top 20 coverage | total deviation | Top 20 Group bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| historic_180d | 16.22 | 33.86 | 56.09% | +54.97% | +66.12% |
| recent_90d | 15.76 | 35.27 | 53.75% | +47.98% | +91.53% |
| extended_tweedie | 13.54 | 36.20 | 58.81% | +7.64% | +5.90% |
| extended_tweedie_lag_calibrated | 13.79 | 35.90 | 58.81% | +12.38% | +10.53% |

## Fixed strengthening goals in advance

- `extended_tweedie`: Not all are satisfied. mae_ 10 pct_better=yes; buyer_mae_ 5 pct_better=no; top 20 _capture_ 3 pp_higher=no; aggregate_bias_within_ 5 pct=no; top 20 _group_bias_within_ 10 pct=Yes; all_months_mae_not_worse=Yes; all_months_top 20 _not_worse=Yes
- `extended_tweedie_lag_calibrated`: Not all are satisfied. mae_ 10 pct_better=yes; buyer_mae_ 5 pct_better=no; top 20 _capture_ 3 pp_higher=no; aggregate_bias_within_ 5 pct=no; top 20 _group_bias_within_ 10 pct=no;all_months_mae_not_worse=yes;all_months_top 20 _not_worse=yes

The monthly training label cutoff, original amount verification and source hash can be found in the same directory file. Any cross-merchant approval is only evidence of method portability and does not constitute permission to go online with the original merchant's amount model.
