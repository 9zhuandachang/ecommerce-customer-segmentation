# Monthly seasonal signal: locked before running the model

This scenario is logged before a new candidate is run and new results are viewed. The problem comes from the drift of the amount scale in the viewed months; therefore, this round only checks whether the calendar month for which the prediction date is known can allow the model to learn the cross-month pattern, and does not multiply the amount after prediction. Does not change `optimization_protocol.md` 's enhancement target.

## The only candidate

- Retain the original 20 pre-snapshot features of `extended_tweedie`, LightGBM Tweedie targets, 120 trees, learning rates 0.05, 15 leaves, variance power 1.5, random seeds 42, training label maturity rules and customer monthly samples.
- Only add two deterministic calendar features: `sin(2π×month/12)` and `cos(2π×month/12)`, the month is the natural month of the customer's monthly snapshot, 1 – 12; do not add the year, future sales of the current month or any post-event tags.
- Compared with the original `extended_tweedie` and `historic_180d` of the same batch of customers, the number of trees, loss function, monthly encoding or post-processing is not searched in this round.

## Fixed evaluation

- 2011 years, January-May are rolling development, June-August are viewed development confirmations, September-November are viewed retrospective diagnoses. Each target month only trains earlier snapshots 30 days in the future where the label window has completely ended. The September-November, now that they have been viewed, are in no way used for candidate selection or claims of independent acceptance.
- Monthly and segmented reports of all MAE, buyers MAE, predicted Top 20 customers' actual consumption coverage, overall amount deviation, Top 20 group deviation and zero consumption ratio; retain customer-level predictions, source and result hashes, and training label cutoffs. In addition, the former 1% customers selected according to the prediction scores will be used as a **diagnostic group** to report their prediction/actual amount ratio, and this post-report reporting item will not be added to the promotion threshold.
- The fixed development promotion threshold is still based on the 1 – 5 Tsukihara 20 characteristic model: the purchaser MAE at least decreases by 5%, the increase in all MAE does not exceed 2%, the top 20 consumption coverage does not decrease, and the absolute value of the total amount deviation does not expand. 6 – 11 Monthly results only reveal stability and cannot be used to change candidates or thresholds. Meeting the threshold will only allow entry into the new external fixed test, which does not equal the final acceptance by the original merchant.
- The calendar month is fully known at the forecast date, but individual merchants only have a short history. Any improvement is likely to be year-specific seasonality and must be tested on unviewed longer time series before system access can be considered.
