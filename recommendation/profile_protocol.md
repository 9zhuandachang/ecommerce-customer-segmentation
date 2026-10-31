# Purchasing habits profile and personalized product recommendation: v 2 advance plan

This file is written before the calculation of this round of profile labels, tendency predictions and hybrid recommendation indicators. v 1 's September-Novemberly recommended answers have been viewed; this round only uses development and verification data for model selection, 9 – 11 is a monthly fixed plan review and cannot be restored to a new independent test. v 1 is kept intact, and the other three targets are unchanged.

## Common tasks and profiles

Use v 1: 1 days of each month, original qualified customers, records 180 days before the cutoff, the same frozen product catalog, 10 pieces per person; the answer is the deduplicated product at the first purchase time within the following 30 days. Unknown products are retained in the denominator of the answer. Customers who do not place an order are still generated a list, but their repetition ratio is not marked as 0.

The profiles are calculated in order of the customer's purchase time at `[t-180day,t)`. Multiple invoices at the same time are merged, the goods are deduplicated, and the quantity and amount are not weighted. Only the reference is established for the first visible purchase; the repeat ratio for each subsequent time = the number of products in this order that have previously appeared in the 180 day record/the number of products in this order. The profile includes: the average repeat ratio of subsequent orders, the average of the two most recent comparable orders, the difference between the last two and earlier comparable orders (only calculated if there are at least 4 comparable orders, otherwise it will be missing), the concentration of the previous 3 product purchase events, the number of orders, the number of comparable orders, the number of different products, the average number of products per order, the number of days since the most recent purchase, and the visible purchase span. Missing values ​​and record numbers are retained, and historical deficiencies are not interpreted as trying new things. They are purchasing behavior statistics, not personality or real consumer motivations.

Tendency answer = Number of items purchased 180 days before the cut-off point for the next order/Total number of items for the next order. It is only defined when there is an order, including products outside the catalog. Multiple new products within the same order during the forecast period cannot become "old products" for each other. This is a conditional proportion prediction, not the probability of repurchase, nor a direct recommendation weight.

## Trend prediction and time boundary

Use the historical sample starting from month 6 in year 2010. When fitting any model, only `snapshot_date + 30day <= fit_cutoff` is allowed to have the next record; preprocessing is only done on these records.

Compare historical repetition proportions (missing values are filled with the average answer of the fitting data), standardized ridge regression (alpha= 0.1 / 1 / 10 / 100 ), small LightGBM (number of leaves 7 / 15 × number of trees 60 / 120 × minimum leaf sample 50 / 150, total 8 group; learning rate 0.03, L 2 = 5, squared error, seed 42, 2 thread). The prediction is clipped to [ 0,1 ]; when there is no comparable order, the mean training answer is used uniformly.

2011, January-May five forward development folds, using monthly average conditions MAE to select each type of parameters and the tendency method used for profile mixing; the same is divided into historical proportion, then ridge regression, and then tree. 6 – 8 only reports the MAE /RMSE/average deviation of the three shortlisted methods every month, and does not reverse the preference method based on the verification recommendation results. The global model for June-August is fitted on before June 1; the September-Novemberly global model is fitted and frozen on before September 1. Customer input in each month can use transactions before the cutoff of that month, but global parameters cannot be updated.

## Four groups of recommendations and fixed searches

Pure replenishment, ItemKNN, ALS, EASE use v 1 only based on the shortlist parameters determined by January-May, without re-selecting September-November. Hot rules are retained as additional controls; each development cycle and final stage are refitted on known transactions at the time.

The mixture is divided into two mutually exclusive paths: purchased products are sorted by the number of individual invoices; unpurchased products are sorted by the recommendation algorithm score. Each road uses the reciprocal of the ranking `1/rank` to unify the scale, and the integration points = old product road `w/rank` or unbought product road `(1-w)/rank`; then select 10 globally. There is no fixed number of old and new places. Zero points and tied points will be filled in the order of the most popular ones in the frozen directory. Using the top 10 items of each channel plus the global popular top 10 items is enough to accurately find the total top 10 items, sorting after deduplication; no answer products are sampled.

- Fixed mixing: Each of the three algorithms tries w= 0 / 0.1 /.../ 1, with a total of 11 groups.
- Profile mixing: the same three algorithms, `w=clip(a+b*(p-mu),0,1)`, a= 0 / 0.1 /.../ 1, b= 0.5 / 1 / 2, each algorithm 33 group. p is the fold's forward predicted next order repetition ratio, mu is the fold's mature training answer mean; there is no historical customer p=mu. a and b are being compared during development and cannot be specified based on test results.
- Each algorithm only selects a fixed configuration and a profile configuration based on the average NDCG@10 of the five development months. All configurations and parameters are preserved. Fixed and profile budgets are different from the full report and do not claim to be the same as calculated budgets.

## Verify selection and acceptance

Compare pure replenishment, three pure algorithms, three fixed mixes, three profile mixes, and popular additional controls in the same three validation months. The monthly average of the main indicator buyers is NDCG@10; Recall /Hit/Precision@ 10, bought/unbought Recall, coverage rate, no historical ratio, @ 5 and the performance of different profile groups are also reported.

Press Validate NDCG to determine the strongest fixed mix and the strongest image mix. Requirements for identification of image gain: relative to the strongest fixed and original pure replenishment, the monthly average NDCG is at least + 0.002; the two comparisons each have at least 2 / 3 and do not decline for a month; the monthly average Recall does not decline by more than 0.005 relative to each comparison; the unpurchased product Recall is at least + 0.005 compared to the pure replenishment. Perform 500 customer group paired sampling on the profile minus fixed as an internal robustness diagnosis.

The default method is selected based on evidence: starting from pure replenishment, the fixed mix must meet NDCG + 0.002, 2 / 3 does not regress every month, and the loss of Recall does not exceed 0.005 before it is replaced; the pure algorithm is optional if it meets the same threshold and is better; the profile mix must meet the above profile threshold and NDCG is higher than the current default before it is replaced. The profiles, comparisons and pages are completed without proving the profile gain, and there is no claim that it improves the quality of recommendations. You can view the list of 10 files for each plan.

After completing the verification, save the selection, verification and final model, configuration and hash, and then generate the September-Novemberly mixed recommendation review results. Do not change defaults, parameters or thresholds based on review after freezing. Reruns only check the complete product; changes to the method require a new version.

## Delivery and Inspection

`recommendation/outputs/profile_v2/` Save profiles, offline-only trend answers/predictions, development and validation comparisons, locked configurations, complete models, safety lists for each solution, reports and reproducible cases. The page presents the profile, number of records, predicted condition repetition ratio and selected plan weight, 10 file recommendations and sources; the export can only contain pre-cutoff profiles and predictions, and does not include future answers or hits.

Expand the profiling formula in existing tests, first order and small sample, same time and quantity deduplication, future orders do not affect profiling, mature label screening, forward model prediction, mutually exclusive channels/same score/complement, quick calculation and complete indicators are consistent, weight is not a fixed quota, selection ignores testing, saved model recalculation, page switching and export check. Temporary test data is only placed in the system temporary directory.

Reference: [RepeatNet repeat/explore ideas](https://ojs.aaai.org/index.php/AAAI/article/view/4408), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html). This experiment uses the above two-way mechanism and does not claim to implement RepeatNet neural network.
