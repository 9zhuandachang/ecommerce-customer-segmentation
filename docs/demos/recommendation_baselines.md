# Product recommendation demonstration: Customer 12348

In PowerShell run:

```powershell
# Run the following commands from the repository root.
python -B -m pipelines.recommendation_baselines
python -m streamlit run app/dashboard.py
```

When a complete product is available, the first command checks the source and output without re-adjusting parameters. Open the local dashboard in the browser:

1. Enter **Product recommendations**.
2. Recommended date selection **2011-09-01**, customer selection **12348**.
3. The default scheme is **Individual Repeat Purchase**: View 10 items and the "Purchased in the past 180 days" tag, download CSV.
4. Switch between **ItemKNN, implicit feedback ALS, and EASE** in "View Recommended Plans" to compare different lists of the same customer.
5. Expand the comparison table to view five methods; after switching dates and customers, there are still 10 products each.

Offline answer to this example: The customer's next qualifying purchase occurred at **2011-09-25 13:13**, and there are two products **23077 and 23078** in the order. The default recommendation is to rank them at **1 and 3** respectively, Recall@10 is 100%, and NDCG@10 is about 0.9197. This is a single example for verification; the overall effect is calculated using all test customers, and this success case cannot be used to represent the overall accuracy.

Command line verification:

```powershell
Import-Csv .\recommendation\outputs\v1\test_customer_metrics.csv |
  Where-Object { $_.snapshot_date -eq '2011-09-01' -and $_.customer_id -eq '12348' -and $_.method -eq 'personal_repeat' } |
  Select-Object snapshot_date, customer_id, recommended_items, ndcg_10, recall_10, hit_10
```

Future orders and hit results only exist in offline evaluation files, and page lists and downloads do not include these answers. See [Comparison Report](../../archive/artifacts/recommendation/v1/report.md) for all models, rules and adverse results.
