# Amazon purchase history: fixed cross-platform timing verification

The source is [Open e-commerce 1.0](https://doi.org/10.7910/DVN/YGLYDY), read from the public Kaggle mirror; the execution details are written to `amazon_protocol.md` first.
Source file 1,850,717 row, qualified purchase record 1,849,744 row, customer 5,023 name; the original row and customer-level forecast are not written to the warehouse.
This issuance document does not have an order ID, and the purchase record frequency is not equal to the original merchant invoice frequency. Participants volunteered to share history, and the sample selection differed from the original project.
There are rows 18 in the image with dates later than 2023-03-31; the last tag of the locked model ends at 2022-10-01, and these later dates are not entered into the input or tags.
The lower limit of the test month sample: meets the plan requirements.

| cutoff point | Partition | method | MAE | Buyer MAE | Top 20 coverage | total deviation | Top 20 Group bias |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2022-04-01 | validation | historic_180d | 126.38 | 147.01 | 51.47% | +8.55% | +21.24% |
| 2022-04-01 | validation | recent_90d | 125.85 | 149.80 | 50.64% | -2.52% | +18.01% |
| 2022-04-01 | validation | extended_tweedie | 117.30 | 134.30 | 52.47% | -0.98% | -1.12% |
| 2022-05-01 | validation | historic_180d | 128.86 | 148.63 | 50.96% | +10.87% | +24.53% |
| 2022-05-01 | validation | recent_90d | 125.10 | 148.17 | 50.47% | +0.26% | +22.43% |
| 2022-05-01 | validation | extended_tweedie | 117.24 | 133.30 | 52.37% | +2.97% | +2.61% |
| 2022-06-01 | validation | historic_180d | 131.57 | 153.55 | 52.03% | -4.48% | +4.79% |
| 2022-06-01 | validation | recent_90d | 131.42 | 156.35 | 51.68% | -8.35% | +8.54% |
| 2022-06-01 | validation | extended_tweedie | 124.09 | 142.53 | 53.76% | -8.23% | -11.25% |
| 2022-07-01 | test | historic_180d | 152.57 | 182.10 | 51.06% | -21.43% | -10.58% |
| 2022-07-01 | test | recent_90d | 155.04 | 185.75 | 50.87% | -19.47% | -3.55% |
| 2022-07-01 | test | extended_tweedie | 148.03 | 173.19 | 52.38% | -19.62% | -19.10% |
| 2022-08-01 | test | historic_180d | 130.23 | 152.33 | 51.55% | +0.84% | +13.86% |
| 2022-08-01 | test | recent_90d | 141.15 | 166.81 | 50.18% | +7.12% | +30.07% |
| 2022-08-01 | test | extended_tweedie | 137.95 | 157.70 | 52.33% | +17.27% | +18.26% |
| 2022-09-01 | test | historic_180d | 124.53 | 145.01 | 51.94% | +11.86% | +25.72% |
| 2022-09-01 | test | recent_90d | 135.89 | 160.28 | 50.11% | +19.49% | +45.74% |
| 2022-09-01 | test | extended_tweedie | 121.46 | 137.93 | 52.15% | +15.43% | +16.18% |

The last three test months combined:

| method | MAE | Buyer MAE | Top 20 coverage | total deviation | Top 20 Group bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| historic_180d | 135.75 | 159.82 | 51.47% | -4.62% | +7.91% |
| recent_90d | 144.01 | 170.95 | 50.43% | +0.37% | +21.41% |
| extended_tweedie | 135.78 | 156.28 | 52.30% | +2.37% | +3.09% |

If the purchase is indeed made 30 days in the future, the buyer error of the conditional amount model. The historical monthly amount rate is a stronger comparison added after the results to avoid mistaking the line-by-line purchase average price as a competitive monthly amount baseline:

| method | Buyer MAE | Buyer amount deviation |
| --- | ---: | ---: |
| historic_monthly_rate | 159.82 | -10.65% |
| average_prior_invoice | 232.18 | -89.46% |
| conditional_l1_median | 146.61 | -23.70% |

This is evidence of the robustness of another sample of platform voluntary participants. It does not replace the original UK merchant's acceptance in the new month, nor does it prove marketing benefits.
