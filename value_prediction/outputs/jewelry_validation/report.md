# Independent e-commerce jewelry transaction timing inspection

Fixed scheme: `jewelry_protocol.md`. Source: [REES 46 data catalog](https://rees46.com/en/datasets) published and linked to [Kaggle jewelry transaction data](https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-jewelry-store).
The actual files cover 2018-12-01 to 2021-12-01; among the 95,911 product lines, 90,559 meet the scheduled cleaning conditions. The description of approximately two years and 13 in the catalog description is inconsistent with the actual content of the file. The following shall be subject to file audit.
Use the original 8 historical features plus 12 time series features and fixed Tweedie parameters; the original rows are not written to the warehouse.
Applicability: The number of purchasing customers in the test month reaches the lower limit of the plan.

| cutoff point | Partition | method | MAE | Buyer MAE | Top 20 coverage | total deviation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2021-06-01 | validation | historic_180d | 112.89 | 1272.35 | 91.79% | +131.76% |
| 2021-06-01 | validation | recent_90d | 97.04 | 1194.18 | 91.31% | +117.86% |
| 2021-06-01 | validation | extended_tweedie | 51.39 | 1220.42 | 94.74% | +6.94% |
| 2021-07-01 | validation | historic_180d | 121.51 | 1146.33 | 88.11% | +95.97% |
| 2021-07-01 | validation | recent_90d | 112.10 | 1068.75 | 87.28% | +103.24% |
| 2021-07-01 | validation | extended_tweedie | 60.24 | 1180.06 | 92.11% | -1.91% |
| 2021-08-01 | validation | historic_180d | 162.72 | 2314.01 | 85.01% | +19.77% |
| 2021-08-01 | validation | recent_90d | 155.76 | 2303.62 | 86.74% | +23.07% |
| 2021-08-01 | validation | extended_tweedie | 100.34 | 2302.24 | 93.03% | -38.59% |
| 2021-09-01 | test | historic_180d | 152.07 | 1542.45 | 90.08% | +74.79% |
| 2021-09-01 | test | recent_90d | 158.23 | 1531.13 | 91.74% | +103.98% |
| 2021-09-01 | test | extended_tweedie | 88.78 | 1481.31 | 94.48% | +28.07% |
| 2021-10-01 | test | historic_180d | 148.93 | 1362.12 | 87.23% | +164.86% |
| 2021-10-01 | test | recent_90d | 170.30 | 1522.87 | 89.60% | +221.49% |
| 2021-10-01 | test | extended_tweedie | 74.95 | 1274.93 | 95.82% | +45.58% |
| 2021-11-01 | test | historic_180d | 172.06 | 1543.29 | 85.43% | +79.69% |
| 2021-11-01 | test | recent_90d | 187.49 | 1632.07 | 87.43% | +113.56% |
| 2021-11-01 | test | extended_tweedie | 92.26 | 1596.23 | 91.95% | -27.21% |

Combined in the last three months (Top 20 selected each month):

| method | MAE | Buyer MAE | Top 20 coverage | total deviation |
| --- | ---: | ---: | ---: | ---: |
| historic_180d | 157.64 | 1491.02 | 87.58% | +99.55% |
| recent_90d | 171.89 | 1565.58 | 89.55% | +137.51% |
| extended_tweedie | 85.37 | 1464.02 | 93.85% | +11.39% |

##Explanation of results

There are 33,182 customer monthly snapshots during the test period, of which 1,220 will be purchased in the next 30 days ( 3.68% ).
The overall model MAE decreases by 45.85% relative to the historical amount baseline; the Top 20 coverage increases by 6.27 percentage points. This is a comparison of the same customers sorted by different ratings.
Buyer MAE only reduces 1.81%; the three-month combined forecast total deviation is + 11.39%, and the direction of the single-month deviation is also unstable. This set of results cannot be regarded as the amount calibration has been accurate.

This data comes from another merchant and can only provide evidence of robustness across merchants; the original project still does not have independent acceptance data for the new month.
