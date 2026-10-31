# Forecast of customer consumption in the next 30 days

Labels are based on qualifying transaction line amounts and cover 54,040 customer monthly snapshots; zero purchases are counted as zero. The original transactions, cleaning results, and original repurchase model remain unchanged.
Amount represents spend on qualifying transactions, not net profit, CLV or marketing incrementality.

The model only uses 8 features from 180 days before the snapshot date. The training period lasts to 2011, and 5 months; 2011, June-Augustly model selection; the September-Novemberly results will be reported only after the model selection is written into the locked configuration. These test months have been viewed in the repurchase task and cannot be called a period of time when the entire project has not been viewed at all.

Acceptance: The overall MAE is lower than the 30 -day baseline converted from the historical 180 -day amount, the actual consumption coverage of Top 20% is higher than the baseline, at least 2 / 3 months will not regress, and the total forecast deviation does not exceed ± 10%. Models are selected during the validation period, and only fixed solutions are checked during the testing period.

Top 20% coverage is based on the monthly snapshot of the top 20% customers selected based on their monthly ratings, and then the actual consumption of the selected customers is divided by the actual consumption of all customers in the current period.

| period | method | MAE | Top 20% actual consumption coverage | WAPE | total deviation | Top 20% Amount increase multiple |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | historic_30d | 203.66 | 71.38% | 108.89% | +8.15% | 3.56 |
| validation | zero | 187.03 | — | 100.00% | -100.00% | — |
| validation | direct_tweedie | 188.47 | 72.05% | 100.77% | -17.22% | 3.60 |
| validation | two_stage | 189.92 | 72.77% | 101.55% | -17.48% | 3.63 |
| validation | direct_l2 | 198.36 | 71.59% | 106.06% | -5.68% | 3.58 |
| test | historic_30d | 263.47 | 67.28% | 94.08% | -20.77% | 3.36 |
| test | zero | 280.03 | — | 100.00% | -100.00% | — |
| test | direct_l2 | 281.39 | 67.52% | 100.49% | -5.58% | 3.37 |

Historical amount baseline during the validation period: MAE 203.66, Top 20% consumption covers 71.38%.
Selected plan: **direct_l 2**.
Acceptance during the validation period: **passed**; post-test threshold for the viewed time period: **failed**. Offline model conclusion: **Not accepted yet**.
The zero prediction is only used to test MAE and does not provide usable customer ranking; therefore its Top 20% indicator is not calculated. The zero prediction of MAE in the validation period is lower than the candidate model, indicating that looking only at MAE will bias all predictions and not buy.
For direct_tweedie, the 3 – 5 monthly forward prediction within the training period resulted in the amount scaling factor 1.0425; it was used for MAE 190.01 and the total deviation - 13.70% after the validation period. It still did not pass the preset threshold, so this scaling was not used.
The fixed scheme's MAE in the September-November is 281.39, and the historical amount baseline is 263.47; among the three test months, only the 0 month's MAE is lower. Top 20% consumption coverage is 67.52%, and the baseline is 67.28%. Total deviation only measures the group total and cannot prove that individual customer amounts are accurate.
The actual purchase proportion during the validation period is 26.72% and during the test period 34.08%; the actual average consumption of each customer's monthly snapshot increases from 187.03 to 280.03. This is an observation of distribution changes during the period, and the cause of the error cannot yet be determined based on summary indicators alone.
Monthly indicators, item-by-item forecasts, label audits, frozen configurations and adoption decisions are shown in the output directory. It is not currently connected to the dashboard; the test month has been viewed in the repurchase task and cannot be used as a new independent verification of the entire project.
