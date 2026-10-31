#Characteristics of order amount changes: customer sampling uncertainty

The purchase records of the same customer across months are resampled together, and the two models use the same sample. The range only describes fluctuations in customer composition for a fixed viewed month and does not include differences in future months.

| period | indicator | point estimate | 95% customer sampling interval | Number of customers |
| --- | --- | ---: | ---: | ---: |
| rolling_jan_may | Buyer MAE Decrease | +1.78% | [+0.65%, +2.96%] | 1,720 |
| rolling_jan_may | Absolute amount deviation change | + 0.98 percentage points | [+ 0.11 percentage points, + 1.56 percentage points] | 1,720 |
| known_jun_aug | Buyer MAE Decrease | -0.01% | [-0.97%, +0.99%] | 1,376 |
| known_jun_aug | Absolute amount deviation change | - 0.07 percentage points | [- 0.55 percentage points, + 0.37 percentage points] | 1,376 |

A positive "Change in Absolute Amount Bias" indicates that the new feature makes the amount more biased. This range does not replace prior 5% improvement thresholds, cross-month stability, or independent acceptance.
