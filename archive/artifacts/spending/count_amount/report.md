# Number of orders × Average order amount: Development period experiment

The only candidate for ex-ante fixation is to use mature historical tags to predict the number of orders and the amount of each transaction separately, and then multiply them to obtain the unconditional estimated consumption amount in the next 30 days. The number of orders is calculated based on a unique invoice, and multiple items in the same order are not counted repeatedly. 2011, September-November were not used in this experiment.

| period | method | All MAE | Buyer MAE | Top 20 consumption coverage | total deviation | Top 20 Group bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | count_times_order_amount | 181.42 | 413.80 | 72.25% | +31.95% | +16.24% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | count_times_order_amount | 190.56 | 429.13 | 73.07% | -3.46% | -10.34% |

## Post-disassembly: number of orders and average order amount

| period | Actual number of orders | Forecast number of orders/actual | Forecast average order amount/actual amount weighted by actual order number |
| --- | ---: | ---: | ---: |
| rolling_jan_may | 4,865 | 133.35% | 104.90% |
| known_jun_aug | 3,175 | 105.21% | 94.24% |

This teardown is only used for error diagnosis. Actual future order numbers are not used to generate forecast scores.

## Fixed in advance January-Mayly advancement conditions

- `buyer_mae_reduction_at_least_5pct`: Not met
- `overall_mae_increase_at_most_2pct`: Not met
- `top20_capture_not_lower`: satisfied
- `aggregate_bias_abs_not_larger`: Not met

Is it worth going into a new external fixation test: **No**. Even if it is satisfied, it can only mean that it is worthy of re-inspection and cannot be used as a basis for independent acceptance or system access.
Customer-level development forecasts, month-by-month metrics, tag checks, training cutoffs and file hashes are in the same directory.
