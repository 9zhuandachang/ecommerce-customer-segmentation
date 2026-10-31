# Purchase constituent features: next round of experiments locked before running

Recorded before generating or viewing the results of this round's constituent feature model. The double weighting of buyers in the previous round reduced the buyer error, but significantly increased the prediction of zero-spending customers; therefore, this round only introduces the **real transactions before the snapshot**, and no longer moves the predicted amount of all customers by weighting the positive labels. Still adheres to the original enhancement goals of `optimization_protocol.md`.

## Input and unique candidate

`clean_transactions.csv` after cleaning of the original transaction has 776,579 and 5,852 customers, and the dates are 2009-12-01 to 2011-12-09; `quantity`, `unit_price` and `country` are not missing. According to the statistics of all cleaning banks, banks with at least 12 account for 31.55% and corresponding sales account for 65.35%; transactions outside the UK involve 518 customers. These are field availability checks, **no variable selected with 30 days in the future tag**. The source file SHA- 256 is `94ca43623b014e3a3f61e23073b5ff053b9a276de966ac1a8d615566df5c3831`.

Within the `[t-180d,t)` input window for each customer month snapshot, only three features are generated:

1. `bulk_spend_share_180d`: The sum of the qualifying transaction amounts of `quantity >= 12` divided by the total qualifying amount of the classmate. 12 pieces are used to differentiate between full case or bulk purchases, the threshold is fixed before future labels are seen.
2. `median_unit_price_180d`: The median of classmate trading rows `unit_price`, calculated by row, not weighted by future orders.
3. `non_uk_180d`: Whether the classmate has a qualified trading line for `country!= "United Kingdom"` ( 0 / 1 ).

 original  `extended_tweedie`  of  20  features, LightGBM Tweedie  Target, 120  tree,  learning rate  0.05, 15  leaves,  variance power  1.5,  random seed  42  Nothing changes;  Only add the above three items,  form the only candidate  `composition_tweedie`. Compare and use the unmodified version of the same batch of customers  `extended_tweedie`  and  `historic_180d`. Feature generation requires checking the original points one by one.  180  Day Amount and Score of Wash Transactions;  If you can't match them one by one,  Stop experiment.

## Evaluation

- Evaluate 2011, January-August forward in time, each fold using only earlier snapshots 30 days in the future where the label has fully ended. The months 1 – 5 are for rolling development, the months 6 – 8 have been viewed for confirmation only; the months 9 – 11 cannot be used for this round of feature or parameter selection.
- Saves monthly, period and customer level forecasts; reports overall MAE, buyer MAE, monthly Top 20 actual spend coverage, forecast total deviation, Top 20 group deviation, zero spend ratio, training label cutoff and all source hashes.
- Only in January-May to judge whether it is worth bringing in a new external fixed test: Buyer MAE is at least 5% lower than the unmodified model, overall MAE is not higher than the unmodified model's 102%, Top 20 coverage does not decrease, and the absolute value of the total deviation does not expand. 6 – 8 only tests the vulnerability during the development period and never changes the three characteristics or thresholds. This R&D advancement condition is not final acceptance.
- If external data does not have comparable quantity, unit price, or country semantics, different defined features should not be hard mapped to the same model; reproducible adaptation and independent time periods must be locked in advance. The original merchant still needs the next new month for final acceptance.
