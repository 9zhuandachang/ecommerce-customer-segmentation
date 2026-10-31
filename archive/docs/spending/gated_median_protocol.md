# Probabilistically gated median amount: pre-run scenario

Recorded before reading this candidate result. Purpose: To test whether a single amount point forecast can take into account the absolute error of a large number of zero-spending customers and actual buyers. The original enhanced acceptance criteria and future independent verification requirements remain unchanged.

## Fixed candidate

- Reuse the saved per-customer purchase probabilities of the stage-five classifier calibrated with the stage-6 sigmoid, and the saved L 1 median prediction under purchase conditions for the 20 feature. No retraining, no adjustment of parameters or scaling.
- The predicted amount is: when the purchase probability **is strictly greater than 0.5**, the conditional median is output; otherwise, 0 is output. 0.5 comes from the judgment of "whether zero accounts for at least half the probability" in the zero-inflated distribution, and is not selected from the current amount label. The conditional median is only an approximation of the median of a nonzero mixture distribution and cannot automatically be treated as an expected amount.
- Compare the historical 180 day amount / 6 of the same batch of customers, the original 20 feature directly Tweedie, and the saved purchase probability × Gamma conditional mean. You cannot combine the best indicators with different outputs into the same model.

## Data and judgment

- Only read the existing customer-by-customer and forward-verification prediction files from 2011, January-August; check the one-to-one correspondence between customers and cutoff points, the consistent amount labels, and the hash of the original files and model scripts. 2011, September-November do not participate.
- January-May are used for development judgment. June-August have been seen and are only used for direction confirmation. Neither paragraph is a new independent test.
- Report the actual consumption coverage, total deviation and Top 20 group deviation of all MAE, actual buyers MAE, Top 20 on a monthly and consolidated basis; record the number of non-zero predicted people. If the number of non-zero predictions in a certain month is less than the number of people on the Top 20 list, the Top 20 ranking of that month does not have complete distinction, and the candidates shall not be promoted.
- Compared with the original 20 feature directly Tweedie, the January-May must simultaneously meet: the buyer MAE decreases by at least 5%; the overall MAE increases by at most 2%; the Top 20 coverage does not decrease; the absolute value of the total deviation does not expand. As long as one item is not satisfied, new independent verification will not be promoted. The original enhancement target is also fully listed, and the relative historical amount baseline will not be adjusted due to the results of this candidate.
- Even if the development conditions are met, new real, unviewed time series still need to be independently accepted before they can be connected to the system.
