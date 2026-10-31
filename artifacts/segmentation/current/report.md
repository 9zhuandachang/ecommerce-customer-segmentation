# Rework of the three customer segmentation algorithms: results and choices

**Final selection: K-means, 4 group.** The parameters are `n_clusters=4`, `init=k-means++`, `n_init=20`, `max_iter=500`, `random_state=42`. The five features, transformations and remaining complete parameters can be found in [selected.json](selected.json), and the loadable model can be found in [selected_model.joblib](selected_model.joblib). This is a new version of the clustering experiment that has been saved independently and has not replaced the existing dashboard or stage five and six models.

## Compare definition

Use the 37,019 training period customer month records from 2010, 6 month to 2011, 5 month. First, 20% customers are reserved according to customer ID, and each reserved customer takes a fixed one-month status for structural evaluation; the remaining customers are resampled 5 times to check whether the new fitting model is consistent in the grouping of the same evaluation set. 2011, June-August only checks unlabeled monthly group numbers and purchasing behavior profiles. All candidates use the same five historical features and preprocessing, with cluster numbers 2 – 6; GMM comparing both diagonal and complete covariances. After the initial test run, the monthly weighting issue for duplicate customers in the reserved set was corrected, and the report only uses the results of the rerun after the correction; for details, see [Experimental Caliber](../../../docs/protocols/segmentation.md).

The five inputs are the recent purchase interval, the number of orders in half a year, the average amount per order, the number of deduplicated product codes in half a year divided by the number of orders, and the proportion of orders in the last 90 days to orders in half a year. The fourth term is an approximation of merchandise breadth, not the mean of the actual number of merchandise per invoice. Future repurchases, amounts, products, and existing model scores are not included in fitting or model selection.

## All candidates

The silhouette coefficient measures the degree of separation of fixed set-aside customers, the higher it is, the clearer it is; ARI is the median consistency of five resamplings by customer, the closer it is to 1, the more stable it is. The "minimum group" is the minimum number of customers in all months and groups in years 2011 – 6 – 8. For the complete accuracy and the BIC and attribution degree of GMM, see [candidates.csv](candidates.csv); for the monthly number and profiles, see [monthly_sizes.csv](monthly_sizes.csv), [validation_profiles.csv](validation_profiles.csv).

| algorithm | Number of groups | Leave the silhouette coefficient | Stability ARI | Minimum group size |
| --- | ---: | ---: | ---: | ---: |
| K-means | 2 | 0.3580 | 1.0000 | 702 |
| K-means | 3 | 0.3227 | 0.9883 | 690 |
| **K-means** | **4** | **0.3056** | **0.9525** | **545** |
| K-means | 5 | 0.2665 | 0.9641 | 307 |
| K-means | 6 | 0.2657 | 0.9824 | 282 |
| Two points K-means | 2 | 0.3580 | 1.0000 | 702 |
| Two points K-means | 3 | 0.3211 | 0.9882 | 689 |
| Two points K-means | 4 | 0.2885 | 0.9487 | 509 |
| Two points K-means | 5 | 0.2561 | 0.9354 | 308 |
| Two points K-means | 6 | 0.2414 | 0.9319 | 144 |
| GMM (diagonal covariance) | 2 | 0.3580 | 1.0000 | 701 |
| GMM (diagonal covariance) | 3 | 0.3127 | 1.0000 | 701 |
| GMM (diagonal covariance) | 4 | 0.2195 | 0.8047 | 236 |
| GMM (diagonal covariance) | 5 | 0.1412 | 0.9762 | 121 |
| GMM (diagonal covariance) | 6 | 0.1504 | 0.9908 | 69 |
| GMM (complete covariance) | 2 | 0.3580 | 1.0000 | 701 |
| GMM (complete covariance) | 3 | 0.2713 | 1.0000 | 701 |
| GMM (complete covariance) | 4 | 0.2183 | 0.6768 | 341 |
| GMM (complete covariance) | 5 | 0.1521 | 1.0000 | 121 |
| GMM (complete covariance) | 6 | 0.1410 | 0.9971 | 121 |

20 candidates all form the specified number of groups and pass the threshold of at least 50 people per month; GMM candidates all converge. BIC favors more components in the tried GMM, but is not consistent with group separation and interpretability; it is only used for GMM internal diagnosis and is not added to the indicators of other algorithms. The validation-period average highest degree of belonging for GMM is high in the 2 – 4 group setting; on these settings, soft belonging does not exhibit large-scale fuzzy boundaries.

## Why choose 4 group

The 2 group has the highest silhouette score, but it mainly only distinguishes between "have not bought in the last 90 days" and "have bought recently", which is not enough to show the purchasing differences within active customers. The 3 group increases the high-frequency customer base. The 4 group further stably distinguishes two types of recent low-frequency customers: one type has higher average order amount and product breadth, and the other type has lower average order amount and product breadth. This difference appears repeatedly in the three verification months; the 4 group's remaining silhouette coefficient is still 0.3056, the customer resampling ARI is the median 0.9525, and the worst is 0.9208; each group has no less than 545 people per month during the validation period. This is the visible price paid for choosing the 4 group: it has a lower silhouette coefficient and lower stability than the 3 group, and is not a champion in all statistical indicators.

When they are both in the 4 group, the silhouette scores of the bisection K-means are 0.2885 and ARI 0.9487; the diagonal/complete covariances of GMM are 0.2195 / 0.2183 and ARI 0.8047 / 0.6768 respectively. They do not give a stable additional picture that is sufficient to offset these gaps. The 5 and 6 groups continue to subdivide customers who have not purchased for a long time or high-frequency customers, and there are also observable differences, but the silhouette score drops to 0.2665 / 0.2657, and increases interpretation and maintenance costs. Therefore, this round uses the **K-means 4 group** as a compromise between "interpretable segmentation and stable structure", rather than claiming that the data naturally only has four types of customers.

## Customer profile of the selected model

The table below gives the range of month-by-month results for 2011, 6 – 8 using the refitted model over the entire training period. Group numbers are model numbers only; customers may change groups in subsequent months. The amount is the original data unit.

| group description | Number of customers per month | Median days since last purchase | Half-year median number of orders | Median average amount per order | Product breadth ratio median |
| --- | ---: | ---: | ---: | ---: | ---: |
| C 0: Recently low single transactions, small product breadth | 579–604 | 40–46 | 1–2 | 145–153 | 7.0–7.5 |
| C 1: Haven't purchased for a long time | 688–746 | 123–131 | 1 | 247–253 | 14.5–15.0 |
| C 2: Recent high frequency | 547–606 | 16–18 | 5 | 318–321 | 11.7–11.9 |
| C 3: recent low frequency, high single transaction and product breadth | 815–832 | 36–41 | 1–2 | 375–384 | 24.0 |

After selection, a historical review profile was made of 2011, September-November: all four groups exist, each group has no less than 526 people per month, and the main purchasing characteristics are still identifiable. See [selected_profiles.csv](selected_profiles.csv) and [selected_assignments.csv](selected_assignments.csv) without future tags] for details. Since these months have been previously viewed in other tasks on the project, they cannot be called new independent tests. Clustering itself cannot be proven to improve repurchase prediction, consumption ranking, or product recommendation; these are separate tests in subsequent stages.

## Recurrence and Boundaries

Run in the project root directory:

```powershell
python -m src.segmentation.comparison compare
python -m src.segmentation.comparison select --algorithm kmeans --k 4
```

See [comparison_manifest.json](comparison_manifest.json) and [selected.json](selected.json) for input, protocol, code, and output hashes. The experiment did not alter the original data, existing formal clustering files, phase five and six frozen products, or dashboards. The current grouping scope is still limited to customers who have qualified purchases in the past 180 days; coverage of customers who have not purchased for a longer period of time is another data definitions issue.
