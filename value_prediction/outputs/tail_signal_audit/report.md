# Pre-cutoff signals of high future consumption: post-event audit

See `tail_signal_protocol.md` for the definition in advance. The high-value group is defined by real future consumption and can only be used for diagnosis; all ranking scores and characteristics themselves are available before the cutoff point. All months have been viewed previously and are not independently accepted.

| period | buy snapshot | Actual number of people before 1% | The group's actual consumption share | Model prediction/actual for the group | This group enters the prediction Top 20 | Historical 180 day amount median percentile | Future/old monthly average median multiple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | 3,414 | 37 | 19.26% | 41.32% | 97.30% | 99.8% | 1.40× |
| known_jun_aug | 2,177 | 24 | 25.42% | 39.23% | 100.00% | 99.7% | 1.97× |
| viewed_sep_nov | 3,025 | 32 | 25.39% | 43.53% | 96.88% | 99.7% | 2.25× |

Historical amount percentiles are calculated across all eligible customers in the same month. The future results in "Future/Old Monthly Average" are only used for post-event fluctuation audit and cannot be used as input for forecasting.

## Pre-cutoff characteristics: high-value group and other actual buyers

| period | Features | High stakes group | other buyers |
| --- | --- | ---: | ---: |
| rolling_jan_may | monetary_value_180d | 39112.42 | 1162.98 |
| rolling_jan_may | max_invoice_value_180d | 14662.58 | 470.10 |
| rolling_jan_may | monetary_value_90d | 17085.54 | 502.86 |
| rolling_jan_may | invoice_frequency_180d | 17.00 | 3.00 |
| rolling_jan_may | recency_days | 14.00 | 37.00 |
| rolling_jan_may | bulk_spend_share_180d | 98.6% | 47.6% |
| rolling_jan_may | median_unit_price_180d | 1.69 | 1.95 |
| rolling_jan_may | non_uk_180d | 35.1% | 9.3% |
| rolling_jan_may | prior_year_available | 100.0% | 100.0% |
| known_jun_aug | monetary_value_180d | 34820.21 | 960.90 |
| known_jun_aug | max_invoice_value_180d | 8804.95 | 411.87 |
| known_jun_aug | monetary_value_90d | 19124.97 | 527.53 |
| known_jun_aug | invoice_frequency_180d | 15.50 | 3.00 |
| known_jun_aug | recency_days | 11.50 | 28.00 |
| known_jun_aug | bulk_spend_share_180d | 99.0% | 46.2% |
| known_jun_aug | median_unit_price_180d | 1.98 | 1.95 |
| known_jun_aug | non_uk_180d | 37.5% | 9.3% |
| known_jun_aug | prior_year_available | 100.0% | 100.0% |
| viewed_sep_nov | monetary_value_180d | 35737.47 | 880.93 |
| viewed_sep_nov | max_invoice_value_180d | 5433.57 | 411.60 |
| viewed_sep_nov | monetary_value_90d | 21415.03 | 475.03 |
| viewed_sep_nov | invoice_frequency_180d | 20.50 | 3.00 |
| viewed_sep_nov | recency_days | 7.50 | 32.00 |
| viewed_sep_nov | bulk_spend_share_180d | 96.0% | 45.5% |
| viewed_sep_nov | median_unit_price_180d | 1.77 | 1.95 |
| viewed_sep_nov | non_uk_180d | 28.1% | 9.9% |
| viewed_sep_nov | prior_year_available | 100.0% | 100.0% |

## Two-stage amount decomposition (explanatory supplement after the first round results)

Only months 1 – 8 have saved two-stage out-of-sample forecasts; the following are calculated based on the actual high-value group after the fact.

| period | Actual number of people in the high stakes group | average purchase probability | Estimated amount/actual amount when assumed to be purchased | Probability×amount forecast/actual | Direct model prediction/actual |
| --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | 37 | 74.67% | 59.29% | 46.22% | 41.32% |
| known_jun_aug | 24 | 81.90% | 46.45% | 39.88% | 39.23% |

The amount prediction "when assumed to buy" is still from the pre-cutoff model, here only the real results are used to select the post-audit group. When the conditional amount is also significantly lower, the underestimation cannot only be attributed to the reduction in purchase probability; but group selection based on the actual extreme value itself will also cause a downward selection bias.

## Top 1% customers selected in advance based on predicted scores

| period | list customer month | Hit the actual extremely high consumption group | List forecast amount/actual amount | List actual consumption share |
| --- | ---: | ---: | ---: | ---: |
| rolling_jan_may | 167 | 81.08% | 102.26% | 27.50% |
| known_jun_aug | 83 | 91.67% | 79.10% | 32.90% |
| viewed_sep_nov | 90 | 84.38% | 73.88% | 31.56% |

This list is selected only based on predicted scores before the cutoff, so its predicted/actual amount ratio is more suitable for checking the calibration of high-scoring clients; see `predicted_tail_monthly.csv` for month-by-month data.

See `monthly_tail_metrics.csv` for the number of people in high-value groups month by month, fluctuations, and the percentile of historical largest orders. See `feature_signals.csv` for a grouped summary of all features. These differences cannot prove that a certain feature can causally explain future high-value purchases, nor can they be used to adjust parameters in the months that have been seen.
