# E-commerce customer analysis project Final Report outline

The report is divided into four parts according to the teacher's requirements, and they are answered in order: **What was used → How the system works → What is the effect → What was learned from the results.** The project background and four goals are placed at the beginning of the first part, without adding another chapter.

The following proportions are for reference only: Tools and techniques used 15%, system design and model 35%, system effects and evaluation 30%, discovery and discussion 20%.

## 1. Tools and Techniques

This section explains what tools are needed to implement the project and what problems they are used to solve.

### 1.1 Project background and goals

Briefly describe the needs of e-commerce customer analysis and lead to four goals:

- Customer segmentation: Understand the purchasing characteristics of different customers.
- Repurchase prediction for 30 days in the future: identify customers with high repurchase tendency.
- Sorting of spending potential: Form a list of customers worthy of priority viewing.
- Purchasing habit profiling and product recommendations: Understand customers' replenishment and trying new habits, and recommend products that may be purchased next.

### 1.2 Technology Stack and Purpose

Use a "tool-purpose" table to introduce: Python, Pandas/NumPy, scikit-learn, XGBoost, LightGBM, Streamlit, Plotly/Matplotlib, Joblib, Git/Git LFS.

Focus on the use of each tool in this project, such as data processing, model training, page display, model saving and version management.

### 1.3 Main Techniques

Overview of data cleaning, time window features, validation by time, parameter selection, feature ablation, stability checks, and negative sample sampling in recommendation training.

The purpose of using these technologies is explained, and the specific model design is placed in the second part.

### 1.4 Reproduction and Quality Check

Describes how dependency management, model preservation, experimental configuration recording, and automated testing can help reproduce results.

**Suggested materials: A technology stack and usage table.**

## 2. System Design and Models

This part is the main line of the report, allowing readers to follow the data flow to understand the entire system.

### 2.1 Overall structure and workflow

Use a flow chart to distinguish between two paths:

- **Offline training:** Transaction cleaning → Build sample → Model comparison → Save final model.
- **Actual usage:** Upload transaction details → select analysis date → automatically calculate features → call the model → display and export four results.

### 2.2 Data and sample design

Introduce the data sources, time ranges, fields and scale of UCI Online Retail II, and explain how cleaned transactions form a customer monthly snapshot.

Next, explain the three types of data units, as well as their respective inputs and labels:

- **Commodity Transaction Line:** A commodity detail in an invoice.
- **Customer Snapshot:** The purchase status of a customer on an analysis day, used for customer grouping, repurchase prediction and spending potential sorting.
- **Customer + product sample:** A combination of a customer and a candidate product, used for product-level scoring.

### 2.3 Model design of four modules

| module | What needs to be written clearly |
| --- | --- |
| Customer segmentation | Which purchasing features are used; why compare K-means, K-means and GMM; why compare 2 — 6 groups; and finally what are the characteristics of the four K-means groups. |
| Repurchase Forecast | How labels are defined; the differences between 8, 11, and 14 feature groups; why LR, spline LR, XGBoost, and LightGBM are compared; how the final XGBoost score is generated. |
| Consumption potential ranking | 20 historical characteristics; the difference between the amount rule, the direct amount model and the two-step model; explain how "purchase propensity × conditional amount after purchase" forms the ranking score. |
| Profiles and product recommendations | How to form the 10 purchasing habit profile; first compare the five recommendation methods, and then compare the four groups of features A-D × three types of models; explain how the current group D 34 features score "customers + products" and generate Top 10. |

### 2.4 User interface and function connection

Briefly describe the data import and four analysis pages, showing how users select customers, view results and export lists.

Focus on how the page calls the saved model.

**Suggested materials: overall flow chart, data sample examples, four-module model design table, screenshots of main pages.**

## 3. System Performance

This part uses experimental results to answer "How does the solution perform and why was it chosen?"

### 3.1 Experimental setup and fair comparison

Explain the time division:

- **Training period:** 2010 years 6 months - 2011, 5 months.
- **Validation period:** 2011, 6 — 8 months.
- **Review during test period:** 2011, 9 — 11 months.

Explain the selection and final fitting range of each module, and how to ensure that the features only use information before the analysis day.

### 3.2 Evaluation indicators and calculation methods

A brief description of each indicator: how to calculate it, what to measure, whether bigger or smaller is better.

| module | Key indicators |
| --- | --- |
| Customer segmentation | Silhouette coefficient, resampling stability ARI, differences in the number and characteristics of each group. |
| Repurchase Forecast | AP, ROC-AUC, Precision@20%, Recall@20%, Brier. |
| Consumption potential ranking | The consumption coverage rate of the former 20% list is mainly, and MAE is the supplement. |
| Product recommendations | NDCG@10, Recall@10, Hit@10, unbought item Recall@10. |

### 3.3 Model Comparisons and Final Results

The four modules present comparison tables and month-by-month results respectively.

The recommendation part presents "comparison of five recommendation methods" and "comparison of four sets of features × three types of models" separately.

The results of the 2011, 9 — 11 monthly review of the current plan can be reported as follows:

| module | Current plan | Key results |
| --- | --- | --- |
| Repurchase Forecast | 14 items + XGBoost | AP is 0.6201, and the previous 20% list is accurate to 66.22%. |
| Consumption potential ranking | Two steps LightGBM | The average monthly consumption coverage of the former 20% list is 70.23%. |
| Product recommendations | full feature group D + LightGBM classification | NDCG@10 is 0.3001 and Recall@10 is 17.79%. |

The grouping part shows the comparison results of the algorithm and the number of groups, as well as the stability and purchasing characteristics of the final four groups.

### 3.4 System Function Verification

Show a real upload case: Using 2011-12-01 as the analysis date, a snapshot of 3,444 customers is generated, the top 20% spending-potential list contains 689 customers, and 10 recommended products are generated for each customer.

Introduce what behaviors are covered by 145 automated tests, such as time window, model scoring recurrence, customer filtering, page switching and list export.

If the response time is reported, the actual measured value and test environment should be written.

**Recommended materials: data division chart, model comparison table, monthly performance chart, and a complete use case.**

## 4. Findings and Discussion

This section explains the reasons behind the results and avoids repeating the numbers from section 3 again.

### 4.1 Main findings

Answer based on experiments:

- What is the difference between high-frequency customers and high-customer customers?
- Does adding features always help?
- Why are individual repeat purchase rules competitive?
- Why are more complex ranking models not necessarily better?

### 4.2 Actual contribution of customer profile and product association

Combined with the comparison of B versus A, D versus C, etc., discuss how much improvement the profile brings and whether it is stable.

Distinguish between the improvement of the entire new model and the contribution of the profile itself.

### 4.3 Business value

Explain how each of the four outputs helps operations personnel understand customers, determine viewing order, and prepare product lists.

Use a specific scenario to connect customer grouping, repurchase prediction, spending potential ranking and product recommendation.

### 4.4 Limitations and subsequent improvements

Discusses issues such as the impact of single-retailer and wholesale transactions, review coverage of viewed months, and weak coverage of unpurchased items by recommendations.

Based on this, 2 - 3 corresponding improvements are proposed, such as adding independent data, checking cross-month performance, and improving new recommendations.

**Suggested materials: Main findings and evidence comparison table, and "Issues - Directions for Improvement" table.**
