# Calendar month candidates: study forward in time

The only difference that is fixed a priori is the sine and cosine of the snapshot month; the rest of the model settings are the same as the original 20 feature Tweedie. 2011, 6 – 11 months have been viewed and are only used for development confirmation or review diagnosis.

| period | method | All MAE | Buyer MAE | Top 20 consumption coverage | total deviation | Top 20 Group bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | calendar_tweedie | 166.95 | 439.67 | 73.16% | +13.98% | +8.48% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | calendar_tweedie | 208.43 | 410.84 | 73.49% | +21.61% | +9.56% |
| viewed_sep_nov | historic_180d | 263.47 | 529.85 | 67.28% | -20.77% | -16.92% |
| viewed_sep_nov | extended_tweedie | 252.58 | 535.01 | 69.47% | -26.16% | -24.09% |
| viewed_sep_nov | calendar_tweedie | 267.48 | 502.95 | 69.10% | -8.83% | -13.61% |

## Prediction score for the top 1% customers selected in advance: amount scale diagnosis

| period | method | list customer month | Forecast amount/actual amount | actual consumption coverage |
| --- | --- | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 167 | 146.84% | 29.91% |
| rolling_jan_may | extended_tweedie | 167 | 102.26% | 27.50% |
| rolling_jan_may | calendar_tweedie | 167 | 110.70% | 27.65% |
| known_jun_aug | historic_180d | 83 | 98.97% | 30.62% |
| known_jun_aug | extended_tweedie | 83 | 79.10% | 32.90% |
| known_jun_aug | calendar_tweedie | 83 | 87.01% | 32.87% |
| viewed_sep_nov | historic_180d | 90 | 73.90% | 31.29% |
| viewed_sep_nov | extended_tweedie | 90 | 73.88% | 31.56% |
| viewed_sep_nov | calendar_tweedie | 90 | 77.47% | 31.94% |

This list is selected by forecast score only; the former 1% indicators are diagnostic and are not used to select calendar models or adjust advance thresholds after the fact.

## Only January-May's default R&D advancement threshold

- `buyer_mae_reduction_at_least_5pct`: Not met
- `overall_mae_increase_at_most_2pct`: satisfied
- `top20_capture_not_lower`: satisfied
- `aggregate_bias_abs_not_larger`: satisfied

All satisfaction only means that it is worth bringing to a new fixed test; these months that have been viewed cannot provide independent acceptance by the same merchant. Month-by-month, client-level forecasts, training label cutoffs and file hashes are in the same directory.
