# Olist cross-merchant fixed window: source and sample size audit

The anonymous orders officially disclosed by Olist are only used for the feasibility check of this pre-fixed window. The amount is the total price of the ordered goods, excluding shipping costs; it is not filtered according to the final performance status to avoid introducing post-order information into pre-cutoff characteristics.
Source order 99,441, product line 112,650; qualified order 98,666, stable customer 95,420; customer monthly snapshot 436,432. The product amount and future tags are verified and approved.

| Lock intercept point | Number of snapshots | future buyers | Whether the monthly 1,000 / 100 threshold is reached |
| --- | ---: | ---: | --- |
| 2018-06-01 | 39,162 | 123 | Yes |
| 2018-07-01 | 40,087 | 98 | No |
| 2018-08-01 | 39,049 | 98 | No |

Three-month feasibility: **Not Satisfied**. According to ex-ante rules, if the threshold is lower than the threshold in any month, the candidate model acceptance indicators will not be calculated.
The original files are only in the system temporary directory; the source hash, status distribution and monthly audit are in the same directory. This cross-merchant source also does not replace the original UK merchant's independent acceptance of the new month.
