# Probability gated median amount: development period results

The threshold 0.5 and all input models are fixed before this result. Months 1 – 5 and months 6 – 8 have been viewed and are for R&D judgment only; months 9 – 11 have not participated. This point forecast is not the expected consumption amount.

| period | method | All MAE | Buyer MAE | Top 20 coverage | total deviation | Non-zero predicted number of people | Top 20 can be completely distinguished |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | 16,435 | Yes |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | 16,435 | Yes |
| rolling_jan_may | stage6_probability_times_gamma | 163.29 | 431.91 | 71.62% | +6.44% | 16,435 | Yes |
| rolling_jan_may | probability_gated_median | 129.05 | 517.75 | 57.77% | -36.21% | 1,424 | No |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | 8,146 | Yes |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | 8,146 | Yes |
| known_jun_aug | stage6_probability_times_gamma | 189.76 | 452.17 | 71.94% | -11.26% | 8,146 | Yes |
| known_jun_aug | probability_gated_median | 158.05 | 524.35 | 61.25% | -46.30% | 932 | No |

## Fixed R&D promotion conditions in advance

- `buyer_mae_reduction_at_least_5pct`: Not met
- `overall_mae_increase_at_most_2pct`: satisfied
- `top20_capture_not_lower`: Not met
- `aggregate_bias_abs_not_larger`: Not met
- `all_monthly_top20_distinguishable`: Not met

New data inspection can be advanced: **No**. The original enhancement targets still need to be independently inspected, and the current results do not constitute independent acceptance.
Customer-by-customer forecasts, month-by-month metrics and source file hashes are in the same directory.
