# Phase 6: Freezing model and independent test results

The validation period ( 2011, June-August) is selected by AP A_base_ 8 / GBDT. The base classifier is only fit during the training period; the validation period is used for sigmoid calibration and fixed F 1 classification threshold 0.280468. The test period is 2011, and September-November, and the above selection has not been participated. A hashed locked_config.json and loadable model were written before testing.

| indicator | Validation period | independent testing period | Test period RFM |
| --- | ---: | ---: | ---: |
| AP | 0.5946 | 0.6180 | 0.5453 |
| ROC-AUC | 0.7577 | 0.7199 | 0.6963 |
| Top 20% Precision | 60.27% | 66.16% | 61.43% |
| Top 20% Recall | 45.15% | 38.84% | 36.07% |
| Top 20% Lift | 2.255 | 1.941 | 1.802 |

There are 8,875 customer monthly snapshots during the test period, and 3,025 ( 34.08% ) as a regular example. Select ceil each month (number of customers in the current month × 20% ), a total of 1,776 quotas, 1,175 hits; RFM hits 1,091. The same customer may appear multiple times across months, and the statistical unit is the monthly snapshot of the customer.

Under fixed threshold: Precision 0.5597, Recall 0.5501, F 1 0.5549; confusion matrix TN= 4541, FP= 1309, FN= 1361, TP= 1664. The Top 20% list is independent of this threshold.

Test Brier: raw score 0.2055, post-sigmoid 0.1926, difference (raw minus calibration) + 0.0129. For binning reliability, see test_calibration.csv. The output is currently called a model score; it represents a ranking of repurchase risk on historical data and does not represent a causal incremental or revenue lift from the campaign.

See monthly_test.csv, cluster_test.csv, and test_predictions.csv for monthly, clustered, and item-by-item results respectively. The independent test results are only used for reporting, and the model or thresholds will not be modified later.
