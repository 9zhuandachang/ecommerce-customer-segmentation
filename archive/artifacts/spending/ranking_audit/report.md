# High-value customer list stability and historical activity error audit

The rules are recorded in `ranking_audit_protocol.md` in advance. This is a diagnostic for the month viewed, not a new independent acceptance; no model retraining or selection is required.
A total of 33,456 unique customer monthly snapshots, ranging from 2011 - 01 to 2011 - 11.

## Top 20 list of mutual customers in adjacent months

| new month period | method | moon pair | Range of number of common customers | List retention average | Jaccard mean | Crescent list consumption coverage average | Average consumption coverage of last month's ranking list in the new month |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 4 | 2,856–3,241 | 88.79% | 79.92% | 70.96% | 71.58% |
| rolling_jan_may | extended_tweedie | 4 | 2,856–3,241 | 82.14% | 69.75% | 71.51% | 70.80% |
| rolling_jan_may | extended_tweedie_previous_year | 4 | 2,856–3,241 | 81.76% | 69.16% | 72.53% | 72.35% |
| known_jun_aug | historic_180d | 3 | 2,436–2,555 | 84.43% | 73.22% | 71.99% | 71.45% |
| known_jun_aug | extended_tweedie | 3 | 2,436–2,555 | 85.84% | 75.35% | 73.44% | 72.65% |
| known_jun_aug | extended_tweedie_previous_year | 3 | 2,436–2,555 | 86.53% | 76.29% | 73.04% | 71.98% |
| viewed_sep_nov | historic_180d | 2 | 2,563–2,753 | 84.29% | 72.85% | 67.43% | 67.10% |
| viewed_sep_nov | extended_tweedie | 2 | 2,563–2,753 | 85.81% | 75.15% | 68.61% | 68.23% |
| viewed_sep_nov | extended_tweedie_previous_year | 2 | 2,563–2,753 | 83.33% | 71.44% | 70.20% | 68.61% |

For monthly data, see `list_stability.csv`. The list of common customers is a diagnostic tool for controlling the inflow and outflow of customers; the rankings of last month are also rearranged within the mutual customers limited after the event, and are not a replay of the old list online.

## The most recent purchase interval known at the time of snapshot

| period | Recent purchase interval | customer months | Snapshot of future purchases | 20 feature model MAE relative baseline | Model MAE for the same period last year relative to the baseline | 20 feature model buyer MAE relative to baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| known_jun_aug | 0 - 30 days | 2,958 | 1,134 | +5.29% | +9.85% | -3.69% |
| known_jun_aug | 31 - 90 days | 2,984 | 757 | +7.56% | +12.65% | -3.78% |
| known_jun_aug | 91 - 180 days | 2,204 | 286 | +13.71% | +21.97% | +1.21% |
| rolling_jan_may | 0 - 30 days | 4,167 | 1,513 | +12.01% | +14.77% | -3.61% |
| rolling_jan_may | 31 - 90 days | 6,002 | 1,208 | +11.90% | +19.15% | -14.05% |
| rolling_jan_may | 91 - 180 days | 6,266 | 693 | +13.29% | +32.47% | -2.05% |
| viewed_sep_nov | 0 - 30 days | 3,530 | 1,503 | +3.35% | +4.75% | -1.82% |
| viewed_sep_nov | 31 - 90 days | 2,988 | 1,027 | +2.89% | +4.98% | -0.81% |
| viewed_sep_nov | 91 - 180 days | 2,357 | 495 | +9.24% | +12.63% | +2.59% |

Positive values indicate a decrease in error, and negative values indicate an increase in error. The complete MAE, buyer MAE, monthly Top 20 coverage and forecast total deviation within the group are shown in `recency_period_metrics.csv` and `recency_monthly_metrics.csv`.

Stability, grouping advantages and list consumption coverage do not prove the accuracy of individual amounts, nor do they replace the original merchant's multi-index acceptance in new months in the future.
