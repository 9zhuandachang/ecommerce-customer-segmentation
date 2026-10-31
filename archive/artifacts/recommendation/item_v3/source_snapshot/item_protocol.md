# Product-level recommendation: four sets of features × three types of models, v 3 advance plan

This solution is fixed before the current round of product-level label matrix, model and new indicator generation. v 1 and profile_v 2 are retained intact; the validation period was used for old scheme research, and the September-Novemberly product answers have also been viewed. This round of 1 – 5 will be developed monthly, 6 – 8 will be verified and selected monthly, and 9 – 11 will be reviewed monthly. It does not claim to resume new independent testing, and does not change the configuration based on the review. The goals of the other three projects remain unchanged.

## Tasks, candidates and labels

Use the deduplicated product answer on the 1 day of each month, the original qualified customer, the previous 180 days of transactions, 10 items per person, and the first purchase moment in the following 30 days. Merge multiple invoices at the same time. The training only uses mature records with `snapshot_date+30day <= fit_cutoff` that have placed an order; 0 means that the product does not appear in the next order, and does not mean that you don't like it. Inference generates a list of all qualified customers. Candidates are all products that appeared 180 days before the stage fitting time point, and the catalog in the stage is frozen; products with unknown answers retain the evaluation denominator and cannot be added to the catalog based on the answer.

## Four groups of features

- A basis: whether it has been purchased before, the number of 180 / 90 / 30 purchases in different invoice days (log 1 p), the number of days since the latest/first visible purchase of the product, the average purchase interval and whether there are two purchases, the interval expiry ratio, the proportion of personal orders containing the product; the purchase popularity of the product 30 / 180 days (log 1 p), change in popularity, number of purchasing customers (log 1 p), number of visible days within 180. Quantity/amount is not used as interaction intensity, customer ID or item ID is not used as input, and item category or true launch date is not assumed.
- B: A + realized 10 item historical profile + profile interacts with the 6 item of "purchased/number of times/product popularity". The history is insufficient and the preservation is missing. The profile is the same for each product of the same customer, and the interaction enables the linear model to test its ranking effect.
- C: A + EASE customer-by-customer normalized score (divided by the absolute maximum value within the candidate) and no available signal flag; do not add all three algorithms, and do not add the ranking repeatedly.
- D: A + B profile and interaction + C signal + repeat ratio × EASE score interaction.

All features are from pre-cutoff trades only. Customer history and observable product statistics are updated monthly, and the global product relationship stage is fixed. The number of unpurchased items is 0, and the time is represented by the 180 day cutoff value and the purchased mark. Missing intervals of less than two purchases are cut off and flags are retained; missing images are only filled in by fitting the median to the training data.

## History EASE signal and sampling

Starting from 2010 - 06, fit EASE on a monthly basis using only transactions 180 days before that time. Regular candidates 10 / 30 / 100 / 300 / 1000 / 3000 / 10000 / 30000; the regular candidates for each month are only based on the EASE average NDCG of the earlier, 30 day-tagged month with fully observed labels. Selected, select the maximum regularity when there is no earlier result. The current month's answer can only affect the parameter selection of subsequent months after its label matures, and does not affect this month's input.

June-August EASE are fitted and frozen within the stage before June 1; September-November are fitted and frozen before September 1. Save fitting boundaries, regular selections and hashes for historical training input, verification and review respectively.

Training retains all candidate positive examples, and samples `min(120,max(50,5*Number of positive examples))` negative examples per customer per month (all are taken if insufficient): half are hard negative examples generated alternately by EASE, personal purchase times, and popular rankings, and the other half are uniformly sampled from the remaining negative examples without replacement, with a fixed seed of 42 and determined by year, month and customer. The four groups and three models use exactly the same samples, and no difficult customers will be deleted due to the model. Customer months that are all unknown positive examples cannot form a training group with positive examples, and are recorded separately and remain in all evaluations. Sampling scores are not interpreted as calibrated purchase probabilities.

Evaluation and reasoning always score the complete candidate list, do not sample negative examples, and do not pre-screen candidates based on real answers; the accurate Top 10 is calculated in batches. The scoring is uniformly retained with 12 decimal places to eliminate the floating point difference in the batch. The same score is based on the popularity of the frozen catalog and the product code. Training samples, grouping and sampling audits are saved as formal reproducible products.

## Three types of models, budget and parameter selection

- LR: Training median filling, standardization, L 2 logistic regression, C= 0.01 / 0.1 / 1 / 10, a total of 4 groups, LBFGS at most 300 times; the total sample weight of each customer per month is the same, and no additional category weights are set.
- LightGBM classification: number of leaves 7 / 15 × number of trees 60 / 120, total 4 group, binary target, learning rate 0.05, minimum leaf sample 100, L 2 = 5, seed 42, 4 thread, certainty; same customer monthly equal rights.
- LightGBM sorting: Same as 4 group tree configuration, lambdarank, 0 / 1 correlation, label_gain=[ 0,1 ], truncation layer 15, query for customer month, lambdarank_norm on; do not use validation period early stop.

Each type of model and each set of features run 4 configurations and 1 – 5 five complete catalogs are folded forward. In order to isolate the role of profile/signal, similar models finally use a common set of parameters: selected according to four groups of features, five-month equally weighted average NDCG, and precise same sorting with stronger regularity/fewer leaves and trees. There are 12 shortlisted versions in total, and the four groups of the same type strictly have the same parameters, training data and preprocessing boundaries. Report the same budget but do not claim the same computing power.

## Old comparison, acceptance and attribution

Keep the original pure replenishment, pure EASE and profile_v 2 strongest profile mixture ( EASE ); directly load the old frozen model, the candidate is the same as the customer. Old fixed mix weight 1, same as pure replenishment, without double counting a different method.

Verify selection: Sort 12 new versions by NDCG, compared to the current default pure replenishment. Replacement threshold: monthly average NDCG at least + 0.01, Recall at least + 0.01, unbought items Recall at least + 0.01, at least 2 / 3 monthly NDCG without regression; when multiple are satisfied, choose the highest NDCG, with the same score, we first have fewer features and a simpler model. If neither is satisfied, the original default will be retained and the gap will be reported without passing.

Profile attribution: compare B-A, D-C with the same model and parameters; compare C-A, D-B with signal attribution, report all, don't just show the favorable combinations. The selected B/D should claim that the profile has extra value, and it needs to be compared accordingly. NDCG should be at least + 0.002, 2 / 3 should not decline every month, and Recall should not drop more than 0.005; otherwise, distinguish between "new product-level scoring is valid" and "image is valid". We will do 500 customer group paired sampling for the pure reduction replenishment of the best new solution and its comparison with/without profiles, as an internal diagnosis after multiple selections, and no new independent significance proof will be made.

Monthly reports on NDCG / Recall /Hit/Precision@ 10, @ 5, unbought/bought Recall, coverage rate, proportion of answers outside candidates, and different profile groups. This round of September-Novemberly product-level recommendation review will be calculated only after the verification freeze selection and final model are used; the old answers have been viewed, and the thresholds, parameters, features or defaults will not be adjusted accordingly.

## Delivery and Verification

Output to `recommendation/outputs/item_v3/`: locking scheme and selection, historical sample/sampling and mature label audit, EASE time audit, 240 secondary development model results, 12 version verification/review and old control, same parameter attribution, model package, 15 security recommendation list, reports and cases. Repeat the run to check the complete product; create another version if the method changes.

The page displays the profile and the number of purchases, recent purchase time, recommendation priority and source of specific products side by side. You can switch the 12 version and old comparison, and export the 10 item list. Scores are not calibrated probabilities and do not demonstrate future orders or individual hit answers. Expand the existing test module to check feature formula/missing/future isolation, EASE selection mature boundary, negative sample sampling, same sample/parameter, complete directory scoring consistent with batching, label and denominator, selection to ignore review, model and full list recurrence, page/export; do not add temporary verification scripts to the warehouse.

Reference: [LightGBM sorting](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html), [parameter](https://lightgbm.readthedocs.io/en/stable/Parameters.html), [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
