# Independent Grocery Retail Data Inspection

This time, the previously locked `external_validation_protocol.md` will be strictly implemented.
This is data from another retailer, and the unit of amount is different from the original item; the results are only used to test the robustness of the method.
Raw rows and customer-level predictions are not saved; native temporary cache with source hash see `manifest.json`.

| cutoff point | Partition | method | MAE | Buyer MAE | Top 20 coverage | total deviation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2017-10-01 | validation | historic_180d | 70.19 | 76.09 | 51.89% | +0.29% |
| 2017-10-01 | validation | recent_90d | 70.00 | 76.62 | 51.66% | -0.44% |
| 2017-10-01 | validation | extended_tweedie | 67.70 | 72.96 | 52.45% | -1.31% |
| 2017-11-01 | test | historic_180d | 71.57 | 76.37 | 50.54% | -1.92% |
| 2017-11-01 | test | recent_90d | 69.32 | 75.31 | 50.66% | -3.17% |
| 2017-11-01 | test | extended_tweedie | 68.52 | 72.61 | 50.94% | -2.58% |
| 2017-12-01 | test | historic_180d | 75.42 | 81.57 | 53.57% | -6.70% |
| 2017-12-01 | test | recent_90d | 74.00 | 81.07 | 53.65% | -6.71% |
| 2017-12-01 | test | extended_tweedie | 74.82 | 80.02 | 54.23% | -4.15% |

External inspection is not a substitute for subsequent independent acceptance of the original online retail item in new months.
