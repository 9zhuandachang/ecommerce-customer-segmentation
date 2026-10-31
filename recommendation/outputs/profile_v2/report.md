# Purchasing Habits Profiles and Product Recommendations: Comparison Results

Default plan: **Purely personal replenishment**.

The profile is based on the statistics of orders 180 days before the cut-off point; what machine learning predicts is "if there is an order 30 days later, how much of it will be old products", not the probability of repurchase. The recommendation weight is determined by the ranking effect during the development period and is not directly equivalent to proportional prediction.

Whether the profile value threshold is met: **False**.

| Plan | Verify NDCG@10 | Verify Recall@10 | Verify not buying Recall | Review NDCG@10 | Review Recall@10 | Reviewed but not bought Recall |
|---|---:|---:|---:|---:|---:|---:|
| Popular products | 0.1270 | 6.64% | 3.53% | 0.0862 | 3.65% | 1.76% |
| Purely personal replenishment | 0.3226 | 20.01% | 0.17% | 0.2863 | 17.17% | 0.14% |
| Pure ItemKNN | 0.2015 | 11.65% | 4.91% | 0.1619 | 8.61% | 3.21% |
| Pure ALS | 0.2526 | 14.87% | 3.34% | 0.2124 | 11.37% | 2.47% |
| Pure EASE | 0.2779 | 16.81% | 2.28% | 0.2418 | 13.91% | 1.74% |
| Fixed Mix + ItemKNN | 0.3226 | 20.01% | 0.17% | 0.2863 | 17.17% | 0.14% |
| Image Mix + ItemKNN | 0.3227 | 20.02% | 0.21% | 0.2861 | 17.15% | 0.18% |
| Fixed Mix + ALS | 0.3226 | 20.01% | 0.17% | 0.2863 | 17.17% | 0.14% |
| Image Mix + ALS | 0.3226 | 20.01% | 0.19% | 0.2860 | 17.13% | 0.17% |
| Fixed Mix + EASE | 0.3226 | 20.01% | 0.17% | 0.2863 | 17.17% | 0.14% |
| Image Mix + EASE | 0.3228 | 20.04% | 0.22% | 0.2861 | 17.13% | 0.15% |

Each month, it is first calculated among customers with next orders, and then the monthly equal-weighted average is calculated. Each person recommends 10 items; unpurchased refers to those who have not purchased 180 days before the cutoff, and unknown products retain the denominator of the answer. Each pure algorithm can still recommend old products; the mixed old/unbought two ways are mutually exclusive.

## Profile Gain and Final Selection

- The strongest fixed mix: fixed mix + ItemKNN. The strongest image blending: Image blending + EASE.
- The profile is relatively fixed: {"ndcg_gain": 0.0001811941659189964, "recall_gain": 0.0003081408259041907, "nonworse_months": 3, "passes": false}.
- The profile is relatively pure replenishment: {"ndcg_gain": 0.0001811941659189964, "recall_gain": 0.0003081408259041907, "nonworse_months": 3, "passes": false}.
- The image of the unbought product Recall is improved compared to pure replenishment: + 0.05%.
- Paired customer sampling interval: `{"ndcg_10": {"point": 0.00018119416591900678, "lower_95": -9.797117305425186e-05, "upper_95": 0.0005035872506525017}, "recall_10": {"point": 0.00030814082590421973, "lower_95": -2.8927049719750072e-05, "upper_95": 0.0007411682942854684}}`.
The range is internal diagnosis after selection; no new monthly income is promised. The profile solution must be more than a fixed mix and a pure replenishment at the same time, and cannot be called an improvement just because the model is added. See locked_selection.json for final parameters.

## Conditional repetition ratio prediction

The method selected during the development period: small LightGBM, parameter `{"id": "lgbm_15_120_150", "family": "lightgbm", "leaves": 15, "trees": 120, "minimum": 150}`.

| method | Verify MAE | Verify RMSE | Verify average deviation | Review MAE |
|---|---:|---:|---:|---:|
| Historical repeat-product share | 0.2526 | 0.3131 | -0.0752 | 0.2542 |
| Ridge regression | 0.2220 | 0.2723 | +0.0166 | 0.2353 |
| Small LightGBM | 0.2188 | 0.2682 | +0.0224 | 0.2354 |

MAE = the average absolute difference between the predicted proportion and the actual proportion, for example, 0.15 means the average difference is 15 percentage points. The first order has no previous reference and is not recorded as a new one; when there is no comparable record, the mean answer value during the fitting period is used. Recent changes require at least 4 comparable orders, the rest are missing. Both the record quantity and missing status are entered into audit.

## Time, fair comparison and opening

January-May forward development: Each fold fit only uses answers that have been fully observed 30 days as of that fold; June-August validation determines the default. Models and selections are locked before calculating the September-Novemberly mixed recommendation results. 9 – 11 Monthly answers have been used previously for v 1 and are a review, not a new independent test. The global model and catalog are frozen during the stage, and customer history is only updated to the cut-off point of the month.

Fixed mixing per algorithm 11 group, profile mixing per algorithm 33 group, both report complete development results. See validation_groups.csv/test_groups.csv for profile group performance; the safe list and offline answers are saved in separate files.

`python -B -m recommendation.profile_experiment`

`python -m streamlit run app/dashboard.py` → **Product recommendations** → Select the profile recommended version.

All 18 Number of monthly profile lines 54,040; run for about 80 seconds. Models, directories, and CSV all have hash checks.
