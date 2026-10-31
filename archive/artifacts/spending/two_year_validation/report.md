# Two-year fixed time test for retail transactions

The real time information only has the DAY serial number; the cutoff point is fixed according to `two_year_protocol.md`.
The same retailer's public one-year version was previously used in the study; this time the last three cutoff points are in the later second year.
Raw row and household level forecasts are not written to the warehouse.

| Cutoff point DAY | Partition | method | MAE | Buyer MAE | Top 20 coverage | total deviation |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 511 | validation | historic_180d | 69.89 | 75.45 | 53.38% | +3.03% |
| 511 | validation | recent_90d | 69.64 | 76.05 | 53.13% | +4.27% |
| 511 | validation | extended_tweedie | 69.14 | 73.44 | 54.01% | +4.85% |
| 541 | validation | historic_180d | 70.88 | 76.68 | 52.21% | +4.81% |
| 541 | validation | recent_90d | 70.72 | 77.47 | 51.85% | +4.23% |
| 541 | validation | extended_tweedie | 67.87 | 73.29 | 52.72% | +1.33% |
| 571 | validation | historic_180d | 71.97 | 77.28 | 51.09% | -1.83% |
| 571 | validation | recent_90d | 70.17 | 76.18 | 51.14% | -3.15% |
| 571 | validation | extended_tweedie | 68.29 | 73.33 | 51.80% | -5.73% |
| 601 | test | historic_180d | 72.36 | 77.52 | 52.08% | -3.39% |
| 601 | test | recent_90d | 69.02 | 75.16 | 51.92% | -4.79% |
| 601 | test | extended_tweedie | 69.39 | 73.94 | 52.36% | -3.92% |
| 631 | test | historic_180d | 75.61 | 81.83 | 52.62% | -5.44% |
| 631 | test | recent_90d | 74.75 | 82.19 | 52.57% | -5.10% |
| 631 | test | extended_tweedie | 74.10 | 79.53 | 53.25% | -3.27% |
| 661 | test | historic_180d | 74.80 | 80.32 | 51.00% | -3.60% |
| 661 | test | recent_90d | 75.60 | 82.09 | 51.31% | -0.95% |
| 661 | test | extended_tweedie | 74.86 | 79.08 | 51.60% | +0.86% |

Combine the last three intercept points (still select Top 20% for each intercept point):

| method | MAE | Buyer MAE | Top 20 coverage | total deviation |
| --- | ---: | ---: | ---: | ---: |
| historic_180d | 74.26 | 79.88 | 51.90% | -4.15% |
| recent_90d | 73.12 | 79.79 | 51.94% | -3.61% |
| extended_tweedie | 72.79 | 77.50 | 52.41% | -2.11% |

This is a cross-retailer method inspection and cannot replace the acceptance of the original online retail project in future months.
