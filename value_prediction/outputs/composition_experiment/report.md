# Features of purchase before snapshot: fixed candidate during development period

The scheme is fixed at `composition_protocol.md` before running the model. New features come from transactions 180 days before each snapshot, trained only with mature historical labels. 2011, June-August have been viewed. The following are not independent acceptances.

| period | method | All MAE | Buyer MAE | Top 20 consumption coverage | total deviation | Top 20 Group bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | composition_tweedie | 172.30 | 415.14 | 72.06% | +22.17% | +10.96% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | composition_tweedie | 188.96 | 439.06 | 73.25% | -6.54% | -12.12% |

## Development period advancement threshold locked before running

- `buyer_mae_reduction_at_least_5pct`: Not met
- `overall_mae_increase_at_most_2pct`: satisfied
- `top20_capture_not_lower`: satisfied
- `aggregate_bias_abs_not_larger`: Not met

Only when all four are met is it recommended to enter a new external fixation test. The three new features or determination thresholds are not changed based on the June-Augustly results.
Month-by-month, client-level results, training cutoffs, and source hashes are found in the same directory.
