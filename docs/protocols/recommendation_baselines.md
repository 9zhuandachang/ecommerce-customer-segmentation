# Product recommendation for next order: Compare plans v 1

This plan is written before the product labels and recommended indicators for this task are generated. The original September-Novemberly repurchase/amount results have been used for other tasks; in this round, the recommendation method and selection will be frozen first, and then the product recommendation results for this period will be calculated at once, and will not be used for selection.

## Questions, Customers and Answers

- For all qualified customers of the original customer monthly snapshot, 10 different products will be recommended on 1 days of each month; already purchased products are allowed to be recommended to accommodate repeated purchases. It is recommended to only use eligible commodity transactions before the cut-off point.
- The correct answer is a collection of different items in the next purchase event within 30 days from the cutoff. The event is the goods of the same customer at the same invoice time; multiple invoices at the same time are merged. The window is `[t,t+30day)`. If there is no purchase, the answer is empty. All orders for the current month will not be mixed into the next order.
- Multiple lines with the same customer, invoice, time, and product are only counted as one purchase interaction; buying many items at one time does not equal multiple preferences. Do not use piece count or dollar amount for recommended interaction strength for wholesale transactions.
- Recommendations are generated for all qualified customers every month. The main indicator is only evaluated on customers who subsequently place an order; hits for all customers are reported separately. Whether to buy or not is only grouped during evaluation and is unknown during prediction.

## Common input and product range

- The global training data for all methods is the number of customer-commodity different invoices 180 days before the fitting time point. ItemKNN / EASE uses `x=log(1+times)`; ALS uses purchased preference 1, unobserved preference 0 and confidence level `c=1+alpha*x`. Unobserved behavior only has a low confidence level and is not interpreted as clear dislike.
- Candidate products are all products that appear in the global fitting window. Each plan is exactly the same and is not filtered based on whether it will be sold or in stock in the future. The product description uses the last visible description before the fitting time point, and does not fill in the name from subsequent records. There is no inventory data, so there is no guarantee that items will be available at that time.
- Global parameters and product scope are frozen at each stage. Only the purchase vectors 180 days before the customer cutoff are updated each forecast month. ALS only solves the customer vector under the fixed product vector and does not update the global product parameters; ItemKNN / EASE uses the same customer input to multiply the frozen product relationship matrix. New customers can also rate as long as they have purchased known products.
- Score the entire collection of products, and do not sample negative products; do not block old products by default. It is recommended that the same score be sorted by the global 180 days with different invoice purchase times in descending order and product code in ascending order. If there is no known product or no similar relationship, the position will naturally be filled with a common popular product, and the proportion will be reported separately.
- When the correct answer includes new products outside the candidate set, it is retained as a miss; the main indicator does not delete these products, and the proportion of recommendable products is reported separately.

## Comparison methods and fixed budgets

- Popular product rules: 30 / 90 / 180 days before the fitting time point, sorted by different number of invoices, three groups of candidates.
- Individual repeat purchase rules: The number of individual invoices for 180 days before the cut-off will not decay or decay according to the half-life of 30 / 90 days. There are three groups of candidates; the shortfall of 10 pieces will be made up with common popular products.
- ItemKNN: cosine similarity and shrink by the number of co-purchasing customers; the number of neighbors 20 / 50 / 100 / 200 × shrink 0 / 20, a total of 8 groups; the diagonal is 0.
- Implicit feedback ALS: implicit vector dimension 16 / 32 × regular 0.1 / 1 × alpha 1 / 10, total 8 group; 15 round exact alternating least squares, seed 42. Complete implementation of Hu et al.'s weighted squared error objective without using ordinary explicit scoring ALS impersonation.
- EASE: regular intensity 10 / 30 / 100 / 300 / 1000 / 3000 / 10000 / 30000, total 8 group; diagonal 0, regularized closed-form solution. Compare the actual running time and do not claim that the same number of configurations equals the same computing power.

## Time process and selection

1. 2011, January-May 5 forward development fold. The global model for each fold only uses transactions 180 days before the 1 days of the month. Use the average value of the five-month condition NDCG@10 to select unique parameters for each type of method; select simpler parameters for the same score.
2. The validation global model for all selected configurations only fits 2011, 6 months 1 days before 180 days. Monthly customer inputs for 6, 7, and 8 respectively end on 1 of the current month; parameters are not updated with the verification answers.
3.  First press the three verification monthly averages NDCG@10 Determine the strongest rules and strongest learning plan. The study program must be average NDCG at least high 0.002,  At least 2/3 months NDCG Don't regress,  average Recall@10 no less than rule more than 0.005;  Otherwise, use the strongest rule. For the same score, choose the simpler option. Range and seed diagnostics no longer change the winner.
4. The final model of model selection, all parameters, verification model and fitting with transactions of 2011, before September 1 180 days is saved and locked, and then September-November product labels, ratings and one-time results are constructed. Global final models and candidate items remain fixed over the three months; customer input is monthly using old orders known at that time. Do not use test indicators to change algorithms, parameters, thresholds or change winners.

## Metrics, Robustness and Delivery

- Main indicator: Macro average NDCG@10 of customers with next order, and then equal-weighted average for each month. The correlation is Buy = 1; the ideal DCG is calculated based on the smaller of the complete number of items in the next order (including items outside the candidate set) and 10.
- Simultaneously report Recall@10, Hit@10, Precision@ 10, all customer hits, product coverage, average number of next orders, number of buyers, proportion of answers outside the candidate set, and number of customers without known products; @ 5 performs sensitivity analysis.
- Report the Recall of purchased/unpurchased items in the past 180 days to avoid relying solely on replenishment to cover up the lack of new product recommendations. Grouping is for evaluation only and cannot be used to block difficult-to-predict customers afterwards.
- Perform 500 verification customer group pairing resampling for the strongest learning plan and the strongest rule, and report the NDCG / Recall difference interval; three months of the same customer are sampled together, and the conditional buyer denominator changes with sampling. Ranges are not interpreted as guarantees of future months.
- ALS shortlisted configuration adds seed 7 / 2026 diagnosis; ItemKNN / EASE has no random training and does not create seed differences. All parameter comparisons and adverse results are retained.
- The page only displays the recommendation date, customer, ranking, product code, current product name, whether the product has been purchased in the past 180 days, and a well-founded recommendation description. Future orders, hits, and test answers do not enter the page recommendation table or its export.
- Verify data boundaries, next order labels, formulas, small mathematical solutions, unknown product denominators, sorting, save model score reproduction, pages and exports; add long-term regression testing of this module under tests, and use the existing page testing module to expand page testing. The original CSV and the other three models are not modified.

Reference: Sarwar et al. Item-based CF ( 2001 ); Hu/Koren/Volinsky Implicit Feedback ( 2008 ); Steck EASE ( 2019 ).
