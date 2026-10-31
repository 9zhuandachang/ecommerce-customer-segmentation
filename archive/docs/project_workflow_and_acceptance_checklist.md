# E-commerce customer segmentation and 30 day repurchase prediction: task process and acceptance checklist

> This list records the early eight-phase plan and its acceptance status at that time. The current course demonstration has been switched to the XGBoost unified historical replay of the K-means four groups and 14 features; the new process and retrospective results can be found in [Project Study Guide](../../docs/project_learning_guide.md) and [Unified Replay Report](../../artifacts/integration/current/report.md). The following expressions of the old three groups, RFM, SoftVote and "independent testing" are only used to understand early experiments and do not represent the current page.

> The current extended functions also include [two-stage LightGBM spending potential ranking](../../artifacts/spending/current/report.md) and [next order product recommendation](../artifacts/recommendation/v1/report.md). In the first round of recommendation, three algorithms and two rules were compared, and the individual repeated purchases won and became the subsequent comparison; the following "no product recommendation" belongs to the early MVP range record.

> The recommendation target is then combined with [purchasing habit profiles and compared with four groups of recommendations](../artifacts/recommendation/profile_v2/report.md): the verification improvement of the profile mixture is very small, and individual repeat purchases are retained in this round. The record of this round can be found in [Profile Recommendation Case](../../docs/demos/recommendation_profiles.md), which does not represent the current default recommendation.

> The latest round is [product-level scoring of [four sets of features × three types of models] ](../../artifacts/recommendation/current/report.md), [open with case](../../docs/demos/recommendation_items.md). The current project has set **full feature group D + LightGBM classification** as the default recommendation, verifying NDCG 0.3448, Recall 21.21%, 9 — 11 monthly review NDCG 0.3001, Recall 17.79%. After uploading the details, 34 features are recalculated and Top 10 are generated; the front end retains data import and four analyses, and does not provide algorithm switching. The original experimental selection record is retained, and the other three goals remain unchanged.

## Project goals and execution sequence

For **existing customers with identifiable purchase records**, use historical transactions to build customer segments and predict their likelihood of purchasing again in the next 30 days. The final delivery is a working Streamlit dashboard that supports viewing groups, sorting by model score, and exporting manual review lists.

Advance in the following order; after each stage passes acceptance, the next stage will be carried out with its products:

**1. Clarify the scope and scope → 2. Data cleaning and auditing → 3. Customer snapshots and labels → 4. Grouping and reference baseline → 5. Six sets of prediction experiments → 6. Evaluation and model finalization → 7. Dashboard integration → 8. Testing and final delivery.**

The project proposal is located in the upper level `pre2` directory. This catalog already has two original CSV, data cleaning and snapshot scripts, corresponding outputs and unit tests; the clustering, modeling and dashboard prototypes have now been connected to customer snapshots and generated local training products, but the experimental design still needs to be aligned with this list, and there is no final evaluation report yet. The check boxes below all indicate that items are to be verified one by one, and "existing documents" are not directly equated with "accepted".

## Stage 1: Clear business scope and consistent definitions

**Task details**

1. The user is identified as the customer relationship management or customer retention team; the core questions are "What are the differences in customer purchasing behavior" and "Which existing customers are more likely to repurchase in the next 30 days".
2. Fixed study object: UCI Online Retail II's two annual transactions CSV. Transaction detail lines are not individual customers; customer IDs are used for correlation and export, not as predictive features.
3. Unify the meaning of "qualified purchase", "existing customer", "customer snapshot" and "model score", and record inapplicable situations. The score is not directly referred to as the true repurchase probability until the calibration check is completed.
4. Fixed MVP boundary: supports historical deadline selection, group viewing, filtering, Top N sorting and CSV export; does not do marketing sending, product recommendation, real-time system access, new customer scoring, in-dashboard training or marketing incremental effect inference.

**Stage output**: [Business scope and data definitions](../../docs/business_scope_and_data_definitions.md). The PPT and Chinese proposal are the planned versions of the year 2026 and month 9; the current implementation status and planned differences are explained item by item in this document and README.

**Acceptance Checklist**

- [x] PPT, Chinese proposal, code and subsequent interface have consistent statements about goals, data ranges and output fields; the plan and the current prototype progress have been distinguished.
- [x] Clarify the four statistical units of transaction line, invoice, customer and customer snapshot, and purchase frequency is counted according to different qualified invoices.
- [x] All features beyond the MVP and conclusions that cannot be supported by data are documented.

## Stage 2: Data cleaning and auditing

**Task details**

1. Read two copies of the original CSV, unify the field names and types, and retain the source file and original line number; the original file remains unchanged.
2. Remove completely duplicate rows; exclude records such as missing customer IDs, invalid dates or key fields, canceled orders, incorrect quantities or prices, non-commodity codes, etc. Each exclusion record retains one main reason, and counts the number of hits for each rule.
3. Check the number of records before and after cleaning, and explain the impact of rules such as missing customer IDs on the coverage of modelable data; keep a list of rules for easy reproduction.

**Stage products**: `clean_transactions.csv`, `excluded_transactions.csv`, `cleaning_audit.csv`, and description of cleaning rules. These products are already available for `pipelines/snapshots.py` and `data/processed/customer_snapshot_pipeline/`.

**Acceptance Checklist**

- [ ] Original number of rows = Number of qualifying transaction rows + Number of excluded rows; the audit table's primary cause count is aligned with the original number of rows.
- [ ] Qualifying transactions have no exact duplicate rows, no missing customer IDs, and both quantity and unit price are positive.
- [ ] Ability to trace the source file, original line number and primary exclusion reason from any exclusion record.
- [ ] Cleansing rules, exclusion sizes and their impact on customer coverage are documented.

## Stage 3: customer snapshot, characteristics and repurchase tags

**Task details**

1. Set a cut-off time `t` on 1 day every month. Snapshots are only generated for customers with at least one qualifying purchase within the previous 180 days.
2. Only transactions within `[t−180 day, t)` are used to construct features such as the latest purchase interval, frequency of qualified invoices, consumption amount, average order amount, product diversity, number of purchases in the past 30 / 90 days; indicators that require earlier history should make their sources and meanings clear.
3. Generate a binary tag using whether there is at least one qualifying purchase within `[t, t+30 day)`. Exclude deadlines with incomplete observation periods 30 days in the future.
4. Divide training, verification, and testing in time order; if the label window of an earlier partition spans the starting point of the next partition, the corresponding snapshot will be excluded from modeling. Preserve date boundaries and partition tables.

**Stage products**: `cutoff_schedule.csv`, `customer_snapshots.csv`, `validation_report.csv`, `run_manifest.json`. The existing pipeline has generated the above files.

**Acceptance Checklist**

- [ ] Unique for each Expiration Date + Customer ID combination where the customer had at least one qualifying purchase within the observation window.
- [ ] Transactions on and after the cut-off time do not enter the features of that day; the label only takes qualified purchases in the next 30 days, and only contains 0 / 1.
- [ ] All label windows are complete; training, validation and testing are arranged in chronological order, and cross-boundary labels do not participate in modeling.
- [ ] Able to reconstruct snapshots of the same definition using source files, configurations and scripts, and retain source hashes and running records.

## Stage 4: Customer segmentation and reference baseline

**Task details**

1. Determine the skew feature transformation, missing value imputation, and scaling methods on the training data and fit K-means; compare to `k = 3, 4, 5, 6`.
2. Look at the silhouette coefficient, the size of each group and the group profile at the same time. The profile contains at least the number of customers, recent purchase interval, invoice frequency and consumption amount; name the group after viewing the profile.
3. Calculate the distance from the customer to each fitted cluster center for subsequent prediction experiments. In addition, the most recent purchase time sorting and RFM sorting are established as a simple reference baseline.

**Stage outputs**: Grouping pipelines under `archive/segmentation/kmeans.py`, `archive/segmentation/stability.py`, `archive/artifacts/segmentation/stage4/`, comparison of each `k`, group profile, center distance, reference baseline, operation report and three-group/four-group stability check. By default, the customer snapshot in the warehouse is read; for dependent versions, see the root directory `requirements.txt` and the stage four running record.

**Acceptance Checklist**

- [x] Filling, transformation, scaling and K-means are only fitted on the corresponding training part; only the fitted steps are applied to the validation set and test set.
- [x] `k = 3–6` was chosen based on metrics, group size, and interpretability; arbitrary group names were not treated as pre-given facts.
- [x] Cluster center distances can be stably recalculated by saved clustering models; the reference baseline uses the same qualified samples and time divisions.

## Stage 5: Complete six sets of prediction experiments

**Task details**

1. Prepare two sets of features: A is the basic transaction feature; B is the basic feature plus cluster center distance. Neither set contains customer IDs, tags, or information beyond the cutoff time.
2. Each group separately trains logistic regression ( LR ), gradient boosting decision tree ( GBDT ) and soft voting models of both, for a total of six groups of experiments. The six groups used the same snapshots, labels, and time divisions.
3. Soft votes are calculated as `w × p(LR) + (1−w) × p(GBDT)`; weights are chosen from `w = 0.25, 0.50, 0.75` on the validation data for the respective feature group.
4. If rolling validation is used, each fold refits the padding, scaling, clustering, and classification models of the fold's training part. Record the configuration, random seed, version and results to ensure that the experiment is reproducible.

**Stage output**: Reproducible training code, configuration, validation set prediction and result table under `archive/repurchase/experiments.py` and `archive/artifacts/repurchase/stage5/`.

**Acceptance Checklist**

- [x] All six sets of experiments can be run, and the comparison only changes the specified feature set and model type.
- [x] All preprocessing, clustering and classifiers are only fitted with corresponding training data; the test set does not participate in parameter adjustment or weight selection.
- [x] The soft voting weight is determined based on the verification set results, and the calculation method and final selection value can be checked.
- [x] Ability to compare to a simple ranking baseline and faithfully record cases where "adding clustering distance" or "soft voting" does not improve.

### Phase 5 Enhanced Experiment: Acceptance Criteria

The original six sets of experiments are retained as controls; the code and results of the enhanced experiments are located at `archive/repurchase/extended.py` and `archive/artifacts/repurchase/extended/`.

- [x] Added group C to only use the 4 purchase rhythm features visible before the snapshot date; A/B/C each run LR, GBDT, and soft voting, a total of nine groups, and use the same official training and verification divisions.
- [x] Perform three rounds of rolling verification on 2011, 3, 4, 5 month within the training period; the 30 day window of each fold training label ends before the start of the verification month, and the clustering of group B is refitted with all filling, scaling, and classification steps fold by fold.
- [x] LightGBM only compares 15 / 31 leaves, up to 500 trees; each fold leaves one month inside the training part to monitor AP and stops early, and the outer validation month is only used for parameter comparison. The selection values ​​and candidate results of the number of trees, leaves and soft voting weights are traceable.
- [x] Nine groups of models, RFM baseline, monthly indicators, item-by-item verification scores and source hashes have been exported; the official validation period is only evaluated after the configuration is selected, the test period does not participate in parameter adjustment or this round of index calculation, and the negative feature experiment results are truthfully recorded.

## Stage 6: Evaluate, select and save model

**Task details**

1. Select the model based on the average accuracy of the validation set ( AP ), and report ROC-AUC, Precision, Recall, and F 1 at the same time. For the Top 20% list, report Lift, Precision@ 20% and Recall@20%.
2. View confusion matrix, class proportions, probability calibration, performance split by time period and group; calculate Brier score when applicable. Classification thresholds, if used, should be determined prior to final testing.
3. After fixing the model and configuration, only make a final evaluation on the later test period set aside; distinguish between verification results, test results and ordinary historical view results.
4. Saves the preprocessing step, fitted K-means, selected classifier and configuration, and model version. If a simple model is better, you can select it and keep six sets of experimental records.

**Stage outputs**: model comparison and discussion report, final test results, loadable model pipeline and version records.

**Acceptance Checklist**

- [x] The model is selected according to the official verification AP; only the RFM baseline is retained and reported in the same test period; Top 20% are allocated monthly quotas, and the test period is 2011, September-November; no accuracy or income improvement conclusion is preset.
- [x] Test period binning still shows certain underestimation; stage six reporting is called "model score", the old dashboard is clearly marked as prototype score and is not interpreted as a marketing incremental effect.
- [x] Test results are generated after the model, calibration and F 1 thresholds are written to the frozen configuration with hashing; the Top 20% list is calculated based on the fixed monthly 20% quota.
- [x] Preprocessing, training epoch K-means, classifier, calibrator and version saved; reload to reproduce clustering and scoring on the same test input.

## Stage 7: Integrated Streamlit dashboard

**Task details**

1. Users only upload purchase details, optionally any day between 2011-09-01 to 2011-12-01; the front-end and back-end limit range, the system automatically excludes transactions on and after that day and generates customer snapshots, then loads the frozen model to re-group, score and rank, and does not read the pre-stored customer results.
2. Only provides four business entrances: customer grouping, repurchase priority list, high spending potential, profile and product recommendation, and supports filtering and CSV export.
3. The page and export display the analysis date and useful business fields, but do not display model versions, experimental comparisons or future answers; there are clear prompts for input errors and empty filter results.

**Stage product**: Runnable Streamlit prototype and its startup instructions.

**Acceptance Checklist**

- [x] There is application testing from upload to four pages of analysis, filtering and export; real December files containing 3,444 customers, excluding future tags.
- [x] The export is consistent with the currently visible business list; the new inference entrance reproduces the original 11 month's grouping, repurchase scores, spending potential rankings and customer 12347 product recommendations.
- [x] The date is selected by the user, and the snapshot is automatically summarized by details; the system rejects missing fields, duplicate details, non-numeric values, and contradictory amounts, and automatically excludes transactions on or after the analysis day to ensure that the four analyses are not entered.
- [x] The model is not retrained on the interface; 12 only provides predictions and does not report the accuracy of missing complete answers. The original experimental and model files of each stage are retained.

## Phase 8: Integration testing, documentation and final delivery

**Task details**

1. Run existing unit tests and regression tests corresponding to new behaviors; check the number of cleaning lines, time windows, leak prevention, model reloading and export consistency. Then use the original CSV to completely run the pipeline and dashboard.
2. Organize data dictionary, source and cleaning audit, experimental configuration, model comparison, limitations, installation and configuration guide and reproduction commands.
3. Update the presentation to be consistent with the actual implementation and measured metrics; record a video of the end-to-end demonstration from selecting the deadline to exporting CSV. The explanation is based on the PPT content and examples. If necessary, a brief explanation will be added without expanding the conclusion.

**Stage outputs**: code and saved pipelines, runnable prototypes, evaluation and discussion reports, data documents, installation guides, final PPT and demonstration videos.

**Acceptance Checklist**

- [ ] Existing tests and project inspections are passed; new functions are verified in a targeted manner, and failed items are recorded and dealt with.
- [ ] The new environment can follow the instructions to install dependencies, rebuild data products, load models and start the dashboard.
- [ ] The functions, numbers and limitations in the final PPT, report, interface and video are consistent; unfinished functions are not written as implemented.
- [ ] Demonstration complete coverage of "Select deadline → View group → Filter and sort → Export CSV ".

## Progress reference

 The original proposed plan was:  No.  1  week  (9  moon  28  day —10  moon  4  day ) Data preparation and snapshots;  No.  1—2  Weekly clustering and baseline;  No.  2—3  Weekly Modeling and Assessment;  No.  3—4  Weekly dashboard;  No.  4—5  weekly test,  Reports and Videos;  The final submission time is  **2026  Year  10  moon  31  day**. This is the proposal schedule,  The actual implementation is subject to the acceptance results of each stage..
