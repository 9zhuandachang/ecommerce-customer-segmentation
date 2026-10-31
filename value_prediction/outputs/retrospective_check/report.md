# 2011, September-November fixed method review check

These three months have been viewed before, and the results cannot be called a new independent test, nor are they used to adjust parameters again.
The lag calibration formula was fixed during the January-Augustly development period of the original project; here we only review the same formula.
The amount for the same period last year and "whether the history is complete" are only determined based on the development period, and are reviewed and diagnosed here.
Only past snapshots with complete labels are used for monthly training; 9 month labels can be used in 10 month training.

| period | method | MAE | Buyer MAE | Top 20 coverage | total deviation |
| --- | --- | ---: | ---: | ---: | ---: |
| 2011-09-01 | historic_180d | 255.22 | 527.51 | 68.60% | -23.45% |
| 2011-09-01 | extended_tweedie | 245.39 | 540.15 | 71.61% | -32.41% |
| 2011-09-01 | extended_tweedie_lag_calibrated | 245.70 | 532.98 | 71.61% | -29.81% |
| 2011-09-01 | extended_tweedie_history_available | 242.79 | 555.93 | 70.76% | -37.12% |
| 2011-09-01 | extended_tweedie_previous_year | 237.92 | 542.61 | 72.31% | -36.04% |
| 2011-10-01 | historic_180d | 266.88 | 569.61 | 68.48% | -15.35% |
| 2011-10-01 | extended_tweedie | 252.55 | 566.49 | 70.63% | -18.75% |
| 2011-10-01 | extended_tweedie_lag_calibrated | 257.31 | 557.91 | 70.63% | -11.02% |
| 2011-10-01 | extended_tweedie_history_available | 248.54 | 573.16 | 70.87% | -22.44% |
| 2011-10-01 | extended_tweedie_previous_year | 247.43 | 566.99 | 71.11% | -20.71% |
| 2011-11-01 | historic_180d | 267.50 | 502.30 | 65.29% | -22.98% |
| 2011-11-01 | extended_tweedie | 258.86 | 508.01 | 66.86% | -27.14% |
| 2011-11-01 | extended_tweedie_lag_calibrated | 262.51 | 495.36 | 66.86% | -17.49% |
| 2011-11-01 | extended_tweedie_history_available | 255.93 | 510.66 | 66.85% | -30.09% |
| 2011-11-01 | extended_tweedie_previous_year | 256.13 | 509.88 | 67.55% | -27.27% |
| total | historic_180d | 263.47 | 529.85 | 67.28% | -20.77% |
| total | extended_tweedie | 252.58 | 535.01 | 69.47% | -26.16% |
| total | extended_tweedie_lag_calibrated | 255.56 | 525.20 | 69.47% | -19.25% |
| total | extended_tweedie_history_available | 249.40 | 542.77 | 69.29% | -29.87% |
| total | extended_tweedie_previous_year | 247.59 | 536.64 | 70.11% | -27.92% |

Customer-level forecasts, monthly metrics, and training deadline audits are found in the same directory.
