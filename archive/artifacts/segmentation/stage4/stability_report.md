# Phase 4: Stability check of three groups and four groups

A fixed number of 2,000 customer-month records are set aside from the training period; all repeated models are fitted using only the remaining training records. "Just change random seeds" keeps the fitting records the same; "Change training sampling" extracts 80% of the remaining records each time, and fixes the K-means seed to 42. Fill, transform, scale and K-means are refitted each time. No validation or testing period labels are used.

Silhouette coefficients are calculated on the same batch of set-out records; ARI measures their grouping consistency with the reference model for this check, 1 means complete consistency, and cluster number exchange does not affect ARI. The reference model has also not been seen in this batch leaving records.

| k | Change mode | Leave a range of silhouette scores | Relative reference model ARI range | Minimum training group proportion range |
| --- | --- | --- | --- | --- |
| 3 | Only change the initialization seed | 0.2244–0.2244 | 0.9986–1.0000 | 27.4%–27.4% |
| 3 | Change training samples | 0.2239–0.2243 | 0.9603–0.9927 | 26.3%–27.4% |
| 4 | Only change the initialization seed | 0.2215–0.2216 | 0.9764–0.9885 | 17.2%–17.3% |
| 4 | Change training samples | 0.2208–0.2216 | 0.9737–0.9910 | 16.7%–17.4% |

Under the same setting, the remaining silhouette score range of the three groups minus the four groups is **+ 0.0026 to + 0.0032**; the three groups are 6 / 6 times higher.

For the number of records set aside for each cluster, the number of deduplicated customers, the average silhouette coefficient, the proportion of negative silhouettes and the RFM median, see [stability_cluster_details.csv](data/stability_cluster_details.csv). Only random factors and training sample perturbations are examined here; the same customer can appear across months, so the set-aside records may share customers with the fitted records, and this is not a generalization test across customers.

See [stability_config.json](stability_config.json) for complete running parameters; see [stability_runs.csv](data/stability_runs.csv) for step-by-step results.
