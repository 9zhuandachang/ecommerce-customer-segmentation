# Purchasing Habits Profiles and Recommended Lists: Step by Step View

> This document retains the original research interface steps. The current front-end has converged to four business entrances, and experimental versions and algorithm switching are no longer provided. For actual use, please select "Image and Product Recommendation"; the results of this round of research can be viewed through `outputs/profile_v2/report.md`.

This round has completed profiles, trend predictions, comparisons of four groups of recommendations, pages and exports. **The default is still personal replenishment**: The recommendation increase for profile mixing is too small and has not reached the pre-set threshold. You can toggle on the same page to see how the profile affects the roster.

## Open

In PowerShell run:

```powershell
# Run the following commands from the repository root.
python -B -m recommendation.profile_experiment
python -m streamlit run app/dashboard.py
```

The first Python command checks the existing complete products and will not re-select the products. The address displayed when the browser opens the terminal, usually `http://localhost:8501`.

## Demo: Clients 12348, 2011-06-01

1. Open **Product recommendations** on the left, recommend the experimental version to choose **Buying Habits Profiles and Mixed Recommendations**.
2. Recommended dates are **2011-06-01**, and recommended customer IDs are **12348**.
3. View image: There were **3 purchases, 2 comparable orders** on the previous 180 days, historical repetition ratio **0%**, and previous 3 product concentration **14.3%**. Leave recent changes blank because at least 4 comparable orders are required. 0% only means that there are no duplicates in this visible record, but it does not mean that it will not be repeated in the future.
4. The page displays the next order conditional repetition ratio prediction of approximately **22.8%**. It answers "If there is an order within 30 days, how many old items may account for it", not the probability of customer purchase; there are only two comparable records, and the page will show "fewer records".
5. Check first **Purely personal replenishment**: The item 10 is **21982**, which I bought before.
6. Switch **Image Mix + EASE**: The replenishment weight is about **90.4%**, and the discovery weight is about **9.6%**. The first 9 pieces are the same as pure replenishment, and the first 10 pieces are replaced by **21212** which were not purchased before.
7. Click **Download the customer profile and recommendation list** to export the same profile, weight and 10 products; the file does not contain the next real product or whether it is a hit.

This is a real input example to show that the list does change with weight, selected by the first customer with "list change and at least two comparable orders", and is not used to prove overall improvement.

## How to see the comparison results

- A comparison of four sets of plans is expanded at the bottom of the page; the original five plans can still be switched in the version menu.
- Validation period: pure replenishment NDCG@10 **0.3226**, the strongest image mixture ( EASE ) **0.3228**, the difference is only **0.00018**; the customer sampling interval includes 0, and stable improvement cannot be considered.
- The three fixed mixtures all selected the replenishment weight **1** during the development period, so the results are consistent with pure replenishment. This is the allowed boundary solution without forcing the addition of unbought items to show mixing.
- Validation of propensity prediction MAE: historical proportion **0.2526**, ridge regression **0.2220**, small LightGBM **0.2188**. Improvements in propensity predictions do not automatically represent improvements in recommendations.
- 9 – 11 Monthly answers have been checked in the previous round. After being fixed in this round, they will only be reviewed and the parameters will not be changed based on the review.

## File

| File | Purpose |
|---|---|
| `profile_protocol.md` | Pre-written formulas, time boundaries, search and selection rules |
| `outputs/profile_v2/report.md` | Full comparison results and limitations |
| `outputs/profile_v2/profiles.csv` | 17,021 security profiles and predictions for six months on the page, no future answers |
| `outputs/profile_v2/validation_lists.csv`, `test_lists.csv` | 11 methods, a total of 187,231 lists and weights of 10 pieces per person, no future answer |
| `outputs/profile_v2/propensity_development_monthly.csv` | Results of five rounds of development of the three-category propensity method |
| `outputs/profile_v2/hybrid_development_monthly.csv` | Complete results for 132 hybrid configurations per round |
| `outputs/profile_v2/validation_groups.csv`, `test_groups.csv` | Month-by-month performance grouped by profile |
| `outputs/profile_v2/locked_selection.json`, `model_bundle.joblib` | Frozen default selections, parameters and recalculable models |

`training_repeat_share_labels.csv` and `*_propensity_predictions.csv` are offline training/checking materials, containing real answers, without entering the page and user export.

## Verify

```powershell
python -B -m unittest discover -s tests
```

Checks cover profiling formulas, historical deficiencies, future transaction isolation, maturity tags, model predictions, fusion rankings, complete denominators, saved model recalculations, page switching and safe export. Other goals maintain the original model and data.
