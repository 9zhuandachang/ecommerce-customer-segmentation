# Number of orders × average order amount: fixed development experiment before running

Recorded before any results are calculated for this candidate. This experiment only uses the customer monthly snapshots of the original merchant from 2011 to 1 to 8; the amount tags from 2011 to 9 to 11 do not participate in training, candidate selection or result tables. The months 1 – 5 are under rolling development, and the months 6 – 8 have been viewed for vulnerability checking only. The target is still the **unconditional spending** of each existing customer in the next 30 days, not the conditional amount when they are known to buy. The original enhanced acceptance criteria remain unchanged; the conditions of this document only determine whether it is worthwhile to introduce new independent inspections at a later date.

## The only candidate

- Tag `invoice_count_30d_future`: **Number of unique invoices** per snapshot customer in qualifying transactions for `[t,t+30d)`, no order recorded as 0. Multiple product lines in the same invoice are only counted as one; check with the audited `spend_30d`, the number of orders is zero if and only if the amount is zero.
- Order number model: all mature training snapshots, LightGBM Poisson, 20 existing pre-cutoff customer features, 120 trees, learning rate 0.05, 15 leaves, random seeds 42.
- Average order amount model: only use the snapshot of `invoice_count_30d_future>0` in mature training, target `spend_30d/invoice_count_30d_future`, LightGBM Gamma, and the same 20 features and parameters. Multiply the forecast order number and the forecast order average amount to output a non-negative estimated consumption amount. Does not use actual order numbers, purchase status, or future amounts for the forecast month.
- The comparison is the historical 180 day amount `/6` and the existing 20 feature Tweedie for the same customer and the same month. Only the target decomposition method is changed, and features, calendar variables or post-event multipliers are not added.

## Check and advance conditions

Monthly and two period reports of 1 – 5 and 6 – 8: consumption coverage of all MAE, actual purchasers MAE, Top 20, total deviation, Top 20 group deviation and zero amount ratio. Save customer-by-customer forecasts during the development period, latest training label cutoffs for each fold, source and script hashes, order counts and amount checks. Each fold training label window must end completely: `snapshot_date+30day<=target cutoff`.

In **January-May**, the R&D promotion threshold for existing Tweedie is fixed in advance as: the actual purchaser MAE must be reduced by at least 5%; all MAE must not increase by more than 2%; the top 20 consumption coverage must not decrease; the absolute value of the total deviation must not expand. All four items must be met. 6 – 8 only observes stability and does not adjust the model or threshold based on it. Full results are reported regardless of whether they are advanced or not. Even if the conditions are met during the development period, it cannot pretend to be a new independent acceptance; the dashboard will not be connected until there is credible new data.
