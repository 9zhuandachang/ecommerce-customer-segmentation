# Consumption amount in 30 days in the future: Cross-dataset evidence comparison

Updated: 2026-10-23. Compare the fixed `extended_tweedie` with the same customer's `historic_180d`. The amount unit varies from merchant to merchant, so only the relative changes are compared horizontally, and the absolute MAE of each merchant is not combined. The list of Top 20 customers in each month is selected based on the predicted amount of that month. A positive MAE improvement indicates that the model is better; a positive Top 20 change indicates that the list covers more actual consumption.

| Data and Evaluation Period | nature of evidence | Overall MAE improvement | Buyer MAE Improvement | Top 20 consumption coverage improvement | Model combined total deviation | Single month MAE wins | Model monthly total deviation range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original merchant 2011 - 06 to 08 | Viewed development confirmation period | +7.33% | -3.13% | + 1.82 percentage points | -7.12% | 3/3 | - 13.96% to + 0.80% |
| Original merchant 2011 - 09 to 11 | Viewed lookback period | +4.13% | -0.97% | + 2.19 percentage points | -26.16% | 3/3 | - 32.41% to - 18.75% |
| External Grocery Merchants Last Three Months | Pre-fixed cross-merchant testing | +1.99% | +2.98% | + 0.50 percentage points | -2.11% | 2/3 | - 3.92% to + 0.86% |
| External jewelry merchant 2021 - 09 to 11 | Pre-fixed cross-merchant testing | +45.85% | +1.81% | + 6.27 percentage points | +11.39% | 3/3 | - 27.21% to + 45.58% |
| External Amazon 2022 - 07 to 09 | Pre-fixed cross-merchant testing | -0.03% | +2.22% | + 0.83 percentage points | +2.37% | 2/3 | - 19.62% to + 17.27% |
| External CDNOW 1998 - 04 to 06 | Pre-fixed cross-merchant testing | +16.49% | -6.91% | + 2.71 percentage points | +7.64% | 3/3 | - 13.35% to + 43.62% |

Calculation: `Whole MAE improve = (baseline MAE - model MAE) / baseline MAE`; buyer MAE similarly; `Top20 Increased consumption coverage = Model coverage - baseline coverage`. The " MAE wins in a single month" here only count the months that are strictly lower than the baseline, and cannot replace the original challenge goal which also requires that the Top 20 in a single month is not bad. The combined total bias may offset overestimation and underestimation in different months, so must be seen together with the last column.

Data source: The original merchant development period is `outputs/optimization/development_comparison.csv` and `monthly_development.csv`; the original merchant review period is `outputs/retrospective_check/overall_metrics.csv` and `monthly_metrics.csv`; the four sets of external data are `outputs/two_year_validation/`, `outputs/jewelry_validation/`, `outputs/amazon_validation/`, and `pooled_test_metrics.csv` and `monthly_metrics.csv` among `outputs/cdnow_validation/`. The `manifest.json` of each directory and the corresponding protocol record source, data processing and test period. CDNOW is historical CD retail data denominated in US dollars. The product ID is not available. 80 zero-amount transactions are excluded according to ex-ante rules. The original merchant's review period has been reviewed and can no longer be used to select solutions; external merchant testing only tests the transferability of the method and cannot pretend to be the original merchant's future independent acceptance.

## Research and Judgment

1. The current direct amount model has relatively consistent help in prioritizing customer ranking, but the overall MAE advantage does not appear stably across merchants, nor does buyer error steadily improve. CDNOW's MAE and list coverage are both better than the baseline, but buyers MAE are higher than 6.91%.
2. The strong overall MAE advantage of jewelry data cannot represent a general effect: its baseline total overestimation is close to 100%, and the model's single-month total deviation still fluctuates significantly. The overall MAE on the Amazon data is almost the same as the baseline.
3. The original merchant's point estimate of 30 days' future consumption has not yet reached the pre-locked multi-indicator challenge target. When there are no real transactions with new merchants, the existing months can only be used for diagnosis and development and cannot generate a new independent acceptance.
4. If the purpose is to screen priority customers, the list ranking and prediction interval can be studied separately; they cannot be interpreted as individual amount accuracy, nor can the amount prediction be integrated into the official dashboard based on this.
