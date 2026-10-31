#Characteristics of order amount changes: R&D results of amount after purchase

This experiment only tests "how much it will cost if you buy it in the future." The January-May are used for development; the viewed June-August are only used for direction verification; the September-November are not read. The conditional amount cannot be regarded as the estimated consumption amount of all customers.

| period | method | Buyer MAE | Buyer amount deviation | Number of buyers |
| --- | --- | ---: | ---: | ---: |
| rolling_jan_may | historic_monthly_rate | 396.56 | -19.96% | 3,414 |
| rolling_jan_may | average_prior_invoice | 360.23 | -28.97% | 3,414 |
| rolling_jan_may | conditional_l1_median | 334.51 | -5.20% | 3,414 |
| rolling_jan_may | conditional_l1_order_dynamics | 328.56 | -6.18% | 3,414 |
| known_jun_aug | historic_monthly_rate | 427.12 | -39.71% | 2,177 |
| known_jun_aug | average_prior_invoice | 410.44 | -38.24% | 2,177 |
| known_jun_aug | conditional_l1_median | 352.18 | -18.08% | 2,177 |
| known_jun_aug | conditional_l1_order_dynamics | 352.23 | -18.00% | 2,177 |

## Fixed R&D promotion conditions in advance

- `buyer_mae_reduction_at_least_5pct`: Not met
- `at_least_four_of_five_months_not_worse`: satisfied
- `absolute_buyer_amount_bias_not_larger`: Not met

Worth entering into new independent inspection: **No**. This does not change the original unconditional amount enhancement target, nor can it be used as a basis for system access.
Month-by-month, customer-by-customer, training label cutoff, feature-by-point verification and hashing are in the same directory.
