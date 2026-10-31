# Online Retail II Overlap and independence check of two original documents

Verification date: 2026-10-23. Data source: [UCI Machine Learning Repository: Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii). The original files, clean files, and snapshots of this project have not been altered as a result of this review.

## Original files overlap

| File | Original number of rows | date range |
| --- | ---: | --- |
| `online_retail_II_2009-2010.csv` | 525,461 | 2009-12-01 07:45 to 2010-12-09 20:01 |
| `online_retail_II_2010-2011.csv` | 541,910 | 2010-12-01 08:26 to 2011-12-09 12:50 |

There are **1,088 invoice numbers** in both files, and each has **22,523 rows** belonging to those invoices; their invoice dates range from 2010-12-01 to 2010-12-09. Comparing all eight normalized transaction fields, different transaction lines on both sides can find the same records ( 22,202 each) on the other side; there are no mismatching lines on both sides. The "different transaction lines" here are the set after all fields are the same and deduplicated, and do not represent 22,202 orders.

`src/data/cleaning.py` concatenates the two files in sequence and uses eight standardized transaction fields to deduplicate identical transaction lines. In the generated `excluded_transactions.csv`, the **22,523 lines in the overlapping area of ​​the second file are all recorded as `exact_duplicate`**; the 321 lines in the overlapping area of ​​the first file are completely repeated internally. The overlapping area is therefore not double-counted in generated wash transactions, customer snapshots, and future amount tags. De-duplicate by complete transaction lines instead of just by invoice number, retaining different product lines in the same invoice. The existing `clean_unique_rows`, line number check and amount label check of `validation_report.csv` all passed; the new regression test covers the same product rows across files and different product rows in the same invoice.

## Independent verification boundaries

Project `cutoff_schedule.csv` generates customer monthly snapshots starting from **2010-06-01**, 2010, 6 – 12 months and 2011, January-May are marked as the training period. The forward training file for the existing model also confirms that 2011-01-01 predictions have been made using the 2010 years of maturity snapshot labels ( 20,584 training snapshot, 5,932 buyer training snapshot). Therefore, although the first CSV provides an earlier same-merchant transaction, the results of 2010 have been used by the current model training and cannot be reused to claim independent acceptance**. Results for 2011, June-August and September-November have also been viewed. There is no new, same-merchant time period that is completely isolated from existing development and evaluation.

This verification only proves that "overlapping original rows have been deduplicated according to normalized transaction rows" and "early months do not have new independence." It does not prove that the amount model has reached the multi-index enhancement standard, nor does it change the requirements for subsequent new data acceptance.
