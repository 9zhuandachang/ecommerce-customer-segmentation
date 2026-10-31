# Product-level recommendation: four groups of features and three types of models

Default selection: **Original Pure Replenishment**.

The overall replacement threshold is met: False; the additional value of the profile of the selected solution: False.

All plans require 10 pieces per person, and the complete catalog of the same product is scored; the main indicator is the monthly average NDCG@10 of customers with next orders, with equal weight for each month. The denominator is reserved for real products outside the catalog. If there is no next order, the list will still be generated. It does not mean that you do not like all the products.

| Plan | Verify NDCG | Verify Recall | Verify not buying Recall | Review NDCG | Review Recall | Reviewed but not bought Recall |
|---|---:|---:|---:|---:|---:|---:|
| A Basics · Logistic Regression | 0.3408 | 20.85% | 0.24% | 0.2952 | 17.33% | 0.15% |
| A Basics · LightGBM Classification | 0.3415 | 20.89% | 0.23% | 0.2996 | 17.76% | 0.13% |
| A Basics · LightGBM Sorting | 0.0701 | 3.36% | 2.86% | 0.0659 | 2.68% | 3.09% |
| B Basics + Profile · Logistic Regression | 0.3445 | 21.21% | 0.24% | 0.2992 | 17.67% | 0.12% |
| BBasics + Profiles · LightGBM Classification | 0.3424 | 21.08% | 0.21% | 0.2999 | 17.80% | 0.14% |
| BBasics + Profiles · LightGBM Sorting | 0.1090 | 7.29% | 2.72% | 0.0987 | 5.76% | 3.13% |
| C Basics + EASE · Logistic Regression | 0.3394 | 20.52% | 0.61% | 0.2928 | 17.19% | 0.27% |
| C Basics + EASE · LightGBM Classification | 0.3422 | 21.02% | 0.56% | 0.3000 | 17.60% | 0.32% |
| C basics + EASE · LightGBM sorting | 0.0870 | 4.44% | 3.26% | 0.0789 | 3.43% | 3.37% |
| D Complete · Logistic Regression | 0.3443 | 21.01% | 0.83% | 0.2978 | 17.67% | 0.43% |
| DComplete · LightGBM Category | 0.3448 | 21.21% | 0.63% | 0.3001 | 17.79% | 0.32% |
| DComplete · LightGBM sort | 0.1054 | 6.69% | 3.63% | 0.1070 | 5.87% | 3.41% |
| Original replenishment | 0.3226 | 20.01% | 0.17% | 0.2863 | 17.17% | 0.14% |
| Original pure EASE | 0.2779 | 16.81% | 2.28% | 0.2418 | 13.91% | 1.74% |
| Previous version profile blend | 0.3228 | 20.04% | 0.22% | 0.2861 | 17.13% | 0.15% |

## Feature attribution of the same model and the same parameters

| control | model | NDCG difference | Recall difference | Month of no regression | Reach additional value threshold |
|---|---|---:|---:|---:|---|
| B_lr − A_lr | lr | +0.0037 | +0.36% | 3/3 | True |
| D_lr − C_lr | lr | +0.0049 | +0.50% | 3/3 | True |
| C_lr − A_lr | lr | -0.0014 | -0.34% | 1/3 | False |
| D_lr − B_lr | lr | -0.0001 | -0.20% | 1/3 | False |
| B_classifier − A_classifier | classifier | +0.0009 | +0.18% | 3/3 | False |
| D_classifier − C_classifier | classifier | +0.0026 | +0.19% | 2/3 | True |
| C_classifier − A_classifier | classifier | +0.0007 | +0.13% | 2/3 | False |
| D_classifier − B_classifier | classifier | +0.0024 | +0.13% | 2/3 | True |
| B_ranker − A_ranker | ranker | +0.0389 | +3.94% | 3/3 | True |
| D_ranker − C_ranker | ranker | +0.0184 | +2.26% | 3/3 | True |
| C_ranker − A_ranker | ranker | +0.0169 | +1.08% | 3/3 | True |
| D_ranker − B_ranker | ranker | -0.0036 | -0.60% | 1/3 | False |

A basic; B basic + profile and interaction; C basic + EASE; D complete. Four groups of similar models use the same parameters, and the parameters are only determined by the average NDCG of the four groups over five development months. B-A and D-C check the image, C-A and D-B check the EASE signal.

## Selection, parameters and bounds

- Best New Solution: D Complete · LightGBM Category. Relatively pure restock: `{"challenger": "D_classifier", "reference": "personal_repeat", "ndcg_gain": 0.02217870449366066, "recall_gain": 0.01202996217324806, "new_recall_gain": 0.004549881179631312, "nonworse_months": 3}`.
- Default replacement threshold: NDCG + 0.01, Recall + 1 percentage points, unbought Recall + 1 percentage points, at least 2 / 3 months NDCG does not regress.
- The additional value threshold for images: the same model and the same parameters NDCG + 0.002, 2 / 3 does not decline for at least one month, and Recall declines by no more than 0.5 percentage points.
- Common parameters of various types: `{"lr": {"id": "lr_0.01", "family": "lr", "c": 0.01}, "classifier": {"id": "classifier_15_120", "family": "classifier", "leaves": 15, "trees": 120}, "ranker": {"id": "ranker_15_120", "family": "ranker", "leaves": 15, "trees": 120}}`.
- Phase fixed EASE parameter: `{"id": "ease_10", "method": "ease", "regularization": 10}`.
- Best New Solution Reduction Pure Replenishment Customer Sampling: `{"ndcg_10": {"point": 0.022178704493660672, "lower_95": 0.01741174604238775, "upper_95": 0.027038263219146885}, "recall_10": {"point": 0.012029962173248088, "lower_95": 0.00860986372676089, "upper_95": 0.015750682466942112}}`.
Sampling is an internal diagnosis after multiple selections and is not a new independent proof of significance.

Historical training EASE only uses previous transactions every month, and its regularity is only determined by earlier mature labels. Meta-model fitting only took 30 days and the answer was mature customer months. Training negative examples include difficult examples and uniform samples, which are shared by all feature groups; classification training corrects the sampling probability, and is equally weighted according to customer months. Ranking optimizes LambdaRank on the same sampled query group and does not claim to be unbiased in restoring the full catalog loss; verification and review score the full catalog. Scores are only used to rank recommendations by the same method for the same customer, not to calibrate purchase probability.

January-May are developed in five rounds forward for a total of 240 model fittings; June-August are determined by default and the final model is locked; September-Novemberly results have been viewed before, and are only fixed for review here, and the parameters will not be changed accordingly. The global model/catalog stage is frozen, and customer history and observable product statistics are updated monthly.

## Open and reproduce

`python -B -m recommendation.item_experiment`

`python -m streamlit run app/dashboard.py` → Product recommendations → Product-level scoring comparison.

development_monthly.csv is the development results of 240 times; validation_ablations.csv is the complete comparison with the same parameters; training_audit.json is the time/sampling audit; training_samples.joblib is the official reproducible training data, including labels, and is only for offline use. The page only reads profiles.csv and *_lists.csv, and does not read future answers in *_customers.csv.

The operation takes about 1447 seconds; all models, codes, sources and products are hash checked.
