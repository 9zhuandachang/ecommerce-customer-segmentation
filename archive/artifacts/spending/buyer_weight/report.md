# Viewed development period: Amount model with double training weight for buyers

See `buyer_weight_protocol.md` for the advance plan. Only give 2 times weight to historical training lines that are mature and will be purchased 30 days in the future; the future purchase status is not known when making current predictions.
This is not a new independent test, nor does it change the original hardening goals.

| period | method | All MAE | Buyer MAE | Top 20 consumption coverage | total deviation | Top 20 Group bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | buyer_weight_2x_tweedie | 207.14 | 387.25 | 71.48% | +61.54% | +31.66% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | buyer_weight_2x_tweedie | 214.63 | 402.47 | 73.16% | +22.67% | +3.72% |

## Development period advancement threshold locked before running

- `buyer_mae_reduction_at_least_5pct`: satisfied
- `overall_mae_increase_at_most_2pct`: Not met
- `top20_capture_not_lower`: Not met
- `aggregate_bias_abs_not_larger`: Not met

## Post hoc error decomposition (explanation of trade-offs, not used for weight adjustment)

| period | method | Proportion of zero-spending customers | Zero consumption customer MAE | Buyer MAE |
| --- | --- | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 79.23% | 143.36 | 396.56 |
| rolling_jan_may | extended_tweedie | 79.23% | 106.64 | 420.92 |
| rolling_jan_may | buyer_weight_2x_tweedie | 79.23% | 159.92 | 387.25 |
| known_jun_aug | historic_180d | 73.28% | 122.16 | 427.12 |
| known_jun_aug | extended_tweedie | 73.28% | 96.92 | 440.47 |
| known_jun_aug | buyer_weight_2x_tweedie | 73.28% | 146.12 | 402.47 |

The actual amount of zero-spending customers is 0, so the group MAE is the average predicted amount. Double weighting reduces buyer error, but significantly increases zero-spending customer predictions, indicating that the two goals are currently in conflict.
Only when all the first four advancement conditions are met is it recommended to enter the fixed test of new data; any results cannot be used as independent acceptance by the original merchant.
Month-by-month metrics, customer-level forecasts, training label cutoffs, and source hashes are found in the same directory.
