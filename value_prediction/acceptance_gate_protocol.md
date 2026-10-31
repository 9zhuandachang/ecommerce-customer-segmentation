# Unified acceptance review rules for the amount model

This document only writes the existing enhancement goals of `optimization_protocol.md` into unified calculation rules, without adding models, changing thresholds, or redefining viewed data as independent tests. Record the rules first and then generate a unified comparison table.

## The same customers, the same future label

In each evaluation month, the number of snapshots, number of future buyers, actual total amount, and Top 20 quota for comparison `historic_180d` and fixed `extended_tweedie` must be the same; each method has a maximum of one row per month. The cross-month aggregation is weighted by the customer's monthly line number MAE, the buyer's line number is weighted MAE, and the Top 20 coverage and amount deviation are summarized by the true total amount. The derived coverage and total deviations in existing reports must be consistent with the recalculation of the underlying amounts. Incomplete or inconsistent records shall not result in a passing conclusion.

## Original strengthening target

Compared with the historical amount baseline of the same batch of customers, fixed requirements: the overall MAE is reduced by at least 10%; the actual purchaser MAE is at least reduced by 5%; the 20% customers before the predicted amount are selected each month, and the combined actual consumption coverage increases by at least 3 percentage points; the overall forecast total deviation is within ± 5%; the forecast total deviation of the Top 20 list is ± 10%; For at least three months forward in time, the model monthly MAE and Top 20 coverage is no worse than the baseline at the same time. Top 20 If the group deviation is not saved, it will be recorded as "missing evidence" and cannot be passed by default. Missing three months or any of the required fields will not pass.

## Source and conclusion boundaries

On the existing output, only the fixed original merchant 2011, January-May, June-August, September-November and four cross-merchant results that have been done are reviewed; the source status is development, viewed and confirmed, viewed review and other merchants. The above source status does not meet the original merchant's independent acceptance in the next new month. Even if a certain set of calculation indicators all meet the standards, it only shows its historical technical threshold; formal access must also have new months from the same merchant that have not been used for selection, training or viewing results, and the sources and labels have been independently audited. This audit cannot release the current data just because the caller provides a `independent=True`.
