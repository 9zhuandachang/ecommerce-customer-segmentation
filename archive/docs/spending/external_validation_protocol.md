# Independent transaction data inspection plan ( 2026-10-23, written before reading external tags)

## Purpose and Boundaries

The public Complete Journey retail transaction is used to test the transferability of the method of "first taking a leak-free customer monthly snapshot and then predicting 30 days of consumption in the future" to another retailer. Data source: dunnhumby Source Files page and `transactions.rds` of `bradleyboehmke/completejourney` R packet. The original data is only saved in the local temporary directory and is not added to the Git repository; the source URL and SHA- 256 are recorded. This data is for grocery retail, which is not equivalent to the online retail business of the original project. It cannot be used to independently prove that the amount model on the original data has reached the standard.

## Locked data processing

- Only use trading lines with non-empty `household_id`, `basket_id`, `product_id`, `transaction_timestamp`, and `sales_value > 0`, `quantity > 0`. Each order is identified by `basket_id` and the household; the number of culls is recorded.
- For each calendar month 1 and day `t`, enter only `[t-180d,t)` and the qualifying sales amount labeled `[t,t+30d)`. Customers must have at least one order within the input window; demographic attributes, promotions, coupons, and future transactions are not used for features.
- Available cutoff points begin with the first calendar month with a full 180 day history after the start of the data and end with the last calendar month with a full 30 day label before the end of the data. Sorted by time, the last 2 available cutoff points are reserved for final testing, the previous 1 are used for development confirmation, and the others are only used for training. If the total cutoff points are less than 5, model acceptance will not be performed.
- Each future cutoff is only trained with historical cutoffs that have ended with full 30 day labels; the same customer for a monthly snapshot can appear across months, which is an actual monthly forecasting scenario, and the results should not be interpreted as "new customer generalization".

## Lock model and evaluation

The candidates are fixed to LightGBM Tweedie ( 120 tree, learning rate 0.05, 15 leaves, variance power 1.5, random seed 42 ) that have been studied during the original project development period and 8 synonymous features of the original project plus 12 time series features that are strictly earlier than the cutoff point. The comparison is the amount of `/6` in the last 180 days; the amount of `/3` in the last 90 days is also reported. Do not adjust features, cleaning, model parameters, thresholds or reporting standards based on external final two-month results.

Main indicators: customer MAE, monthly top 20% customer actual consumption coverage, overall forecast total deviation, and month-to-month relative baseline difference. Purchaser MAE is used as diagnosis; the proportion of zero-consuming customers is reported separately. The development validation period can be used to catch implementation bugs; the final evaluation is only done once every two months. The same customers are compared with the same cut-off point, and amounts in different currencies are not combined.

This is a cross-domain robustness check. The original project's hardening challenge goal remains at `optimization_protocol.md`; external data is not used to lower this goal post hoc. Passing external inspections does not replace the final verification of the original project in a new time period in the future.
