#Future 30 Customer list with sky-high spending potential: Course demonstration definition

This plan was recorded before the list function was implemented, but the experimental results of the original project amount 2011, 1 – 11 month have been reviewed**. Therefore, the following comparisons are historical replays, not new independent tests, and do not prove that campaigns result in incremental revenue.

## Fixed input and usage

- Customer range: All customers with qualifying purchase records in the past 180 days in the snapshot at the beginning of each 2011, 6 – 11 month. Independent ranking every month, the list quota is `ceil(Number of customers × 20%)`.
- Fixed model: Existing time-forward prediction scores for `extended_tweedie` only use their **relative order** and do not interpret the scores as reliable individual expected amounts. Fixed comparison: `historic_180d`, that is, the consumption amount in the past 180 days is divided by 6 and then sorted.
- Forecast files and snapshot sources must pass `ranking_audit.load_predictions()` integrity, hash, cohort and historical tag maturity time checks. The customer ID is only used for deterministic sorting when correlating and tying scores; the actual consumption amount in the next 30 days is only used for offline evaluation and will never be exported to the list file or dashboard.
- 2011, June-August is called "viewed development confirmation"; September-November is called "viewed retrospective diagnosis". No paragraph may be renamed an independent test.

## Fixed evaluation and display threshold

The first 20% is taken every month in descending order of score and ascending order of customer ID. `consumption coverage = Customers on the list in the future 30 Total actual qualifying consumption per day / The sum of the actual qualifying consumption of all snapshot customers in the month`. The coverage ratio is calculated by adding up the amount for each of the two three-month periods, and comparing the historical amount baseline for the same customers and quotas.

The conditions under which the course dashboard can be displayed as a **History Replay Demonstration** are:

1. Model consumption coverage for both March periods was at least **1 percentage points** higher than the baseline;
2. The monthly coverage rate for six months is not lower than the baseline;
3. The input source verification passed, the primary key of the list is unique, the monthly quota is correct, and the export does not have future tags or original estimated amounts.

This is a limited practicality threshold and does not replace the original enhanced acceptance of "accurately predicting specific consumption amounts". Thresholds are set after the results have been viewed and should only be used as clear rules for course demonstrations, with no claims of pre-enrollment, statistical significance, or future business benefits. If any condition fails, all results are reported and the dashboard does not display the model list.

## Interface meaning

"Spending-potential" means the top 20% customers in the current snapshot, for manual review and contact priority. The exported percentiles only represent the relative ranking for the month, not the probability of purchase nor the expected pound amount. Formal operation requires forward-looking verification in the new time period, and further evaluation of marketing costs, reach and incremental effects.
