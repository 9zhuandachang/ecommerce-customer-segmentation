# Upload Purchase History and View the Four Analyses

## 1. Start the app

From the repository root, run:

```powershell
python -m streamlit run app/dashboard.py
```

Open **http://localhost:8501**. The app starts on **Data import**, which is also available in the sidebar. Upload one purchase-history CSV; the system creates customer snapshots automatically. The four analysis pages become available after analysis succeeds.

## 2. Use the real December demo

Upload `data/demo/2011-12-01/purchase_history.csv`.

1. Select the file in **Purchase history CSV**.
2. Set **Analysis date** to **2011-12-01**.
3. Click **Generate analysis**. The app opens **Customer segments** when processing finishes.
4. Open the four analysis pages to filter customers and export the current list.
5. On **Product recommendations**, select customer **12347** to view purchase habits and ten recommended products. Customers can be sorted by purchase count in the past 180 days. Expand **Profile details** to see available measures such as recent repeat-product share, its change, and product concentration.
6. Use **Data import** in the sidebar or **Change data** to upload another file or select another date. Invalid input leaves the current results available. Click **Return to current analysis** or open an analysis page from the sidebar.

The December 1 example generates **3,444 customer snapshots** from **759,615 real eligible product transaction lines**, covering 2009-12-01 through 2011-11-30 17:37. It contains no synthetic customers, synthetic transactions, or future labels. The complete subsequent 30 days are unavailable for December, so the page does not display prediction accuracy for this batch. The ZIP contains only purchase history and its provenance manifest; no snapshot upload is required.

## 3. Prepare your own file

Use a UTF-8 CSV of at most 150 MB, keeping identifiers as text. Use the template's field names. Do not append training labels, future amounts, model scores, or ranks.

The purchase-history file has eight columns:

| Field | Meaning |
| --- | --- |
| `customer_id` | Customer identifier; multiple rows can belong to the same customer. |
| `invoice` | Invoice identifier; an invoice can contain several products. |
| `invoice_date` | Purchase date and time. |
| `stock_code` | Product code. |
| `description` | Product description. |
| `quantity` | Purchased quantity. |
| `unit_price` | Unit price in pounds sterling. |
| `line_amount` | Quantity × unit price, rounded to two decimal places. |

Upload cleaned positive product purchases, excluding cancelled orders, non-product entries, and exact duplicates. The page reports invalid dates or amounts.

Provide the **complete available purchase history**. Most features use the preceding 180 days, while first-purchase and cumulative-spending features also use older records. Newer customers can have less than 180 days of history. The system builds snapshots and each model's additional historical features; users confirm history completeness rather than calculating snapshots themselves.

Choose **any date from 2011-09-01 through 2011-12-01, inclusive**. Both the calendar and backend enforce this range. The system uses purchases strictly before midnight on that date and includes customers with at least one eligible purchase in the preceding 180 days. Later transactions in an upload are excluded, and their count is shown.

For example, selecting **2011-11-15** uses purchases before November 15 to score the following 30 days. Selecting **2011-11-01** generates snapshots for **3,188 customers**.

## 4. How the pages generate results

| Page | Calculation after upload |
| --- | --- |
| Customer segments | Applies saved K-means assignments, displayed as Groups 1-4. |
| Repurchase priorities | Calculates purchase-rhythm features, applies XGBoost, and sorts by score. |
| High-value customers | Calculates spending-window features, applies two-stage LightGBM, and selects the top 20% of the current batch. |
| Product recommendations | Builds 34 group-D features from purchase records, profiles, and EASE associations. The saved LightGBM classifier scores the candidate catalog and selects ten products. |

The saved models and product-code system come from the same UCI retailer. The supported dates follow model preparation and remain within the demo's coverage. Recommendations use the frozen catalog. Customer IDs identify records; they are neither model features nor keys for retrieving old scores.

The default recommendation method is **group D + LightGBM classification**. Sessions containing results from the previous rule-based method prompt users to upload and generate again. All four pages then use the new results.

Daily date selection is an inference feature: the historical window remains 180 days and the prediction horizon remains 30 days. Existing month-start training and evaluation records are unchanged. Daily date support does not imply identical measured accuracy on every day.

Repurchase scores support ranking and have not undergone a new probability-calibration step. Spending-potential ranks are presented rather than reliable individual spending estimates. The catalog has no live inventory feed. Uploading runs inference with saved models; it does not retrain them.
