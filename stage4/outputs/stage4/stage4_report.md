#Customer segmentation: Who buys frequently, who only comes occasionally, and who hasn't bought for a long time?

> What this report wants to answer is what are the differences between customers that deserve attention, and what these differences can bring to subsequent analysis.

## Let's look at the three most noteworthy discoveries first.

1. **People who buy often may not necessarily be particularly expensive every time.** The number of purchases made by the Frequent Buying Group in half a year is about 5.0 times that of the low-frequency group, but the typical single transaction amount is only about 1.5 times. Frequency of purchase is a very prominent difference between the two groups.
2. **A single transaction costs a lot, and it may not be active in the near future.** When divided into four groups, the typical single transaction amount of one group is 417, which is higher than the 345 of the most frequent purchasing group; however, the days since the last purchase are 53 days and 13 days respectively.
3. **Just because I have bought a lot recently does not mean I will definitely continue to buy next month.** The newly added "Purchase Rhythm Change" feature discovered a counter-intuitive phenomenon, and specific comparisons are given later.

Reading definition: A customer records one record every month, which is called "Customer Monthly Record". The same person may belong to different groups in different months. Unless otherwise specified, the profiles are recorded using the training period ( 2010, 6 month - 2011, 5 month); the "typical value" refers to the median, which is the number in the middle after sorting. The amount follows the original unit of CSV.

## 1. Divided into three groups, what kind of people are they?

**k Just want to divide into several groups. k= 3 means divided into three groups; C 0, C 1, C 2 are just numbers, there is no order of priority.**

| group | Record proportion | since last purchase | Number of purchases in half a year | Half-year consumption amount |
| --- | --- | --- | --- | --- |
| Frequently come and buy frequently (C 1 ) | 27.3% | 19 days | 5 times | 1,746 |
| Low frequency purchase type (C 2 ) | 40.1% | 43 days | 1 times | 390 |
| Long time unpurchased type (C 0 ) | 32.6% | 128 days | 1 times | 318 |

![Comparison of purchasing behavior of three types of customers](figures/customer_group_story.png)

### Frequently come and buy frequently: The core characteristic is that they often buy and buy more products.

The typical number of purchases for this group of people in half a year is **5 times**, and they have purchased **74 kinds of goods**; the low-frequency group has 1 times and 23 kinds respectively. The typical single amount is **378**, and the low-frequency group is **253**. This is more like a group of customers who continue to buy and have a wide range of products. The single amount is the median of the average order amount per customer, and several medians cannot be directly multiplied.

Let's look at another example of not counting customers repeatedly: **2011, 05 month 01 day**, this group only has **837 people**, accounting for **27.5%** of customers on that day, but accounting for **73.2%** of the consumption amount of these customers in the past 180 days. This ratio indicates that maintaining the continuity of their purchases is a concern; it is the share of consumption within the customer range for that day.

### Low-frequency purchasing type: There are many people, but frequent purchasing habits have not yet been formed.

This group accounts for all records **40.1%**, of which **52.3%** has only purchased once in the past six months; **15.0%**'s first visible purchase was in the last 30 days. Therefore, this group includes both customers who have recently appeared in the data, as well as older customers who do not buy frequently.

**You can continue to ask:** For people who only have one purchase record, what historical behaviors are related to the second purchase? "First visible purchase" is the first purchase record in the data and cannot be directly regarded as the actual registration time.

### Long time no purchase type: The main difference is the long silence time

This group has typically had **128 days** since their last purchase, and records for **94.9%** have not purchased in the past 90 days. But its typical single transaction amount **266** is close to that of the low-frequency group **253**. Describing it as "not purchased for a long time" is more accurate than just labeling it "low value".

**You can continue to ask:** Are there different replenishment cycles for people who have not purchased for a long time? The existing snapshot does not have the complete order interval and product category, so we can only raise this pending verification issue for the time being.

## 2. Why should we watch the fourth, fifth and sixth groups?

Divide into more groups to see the profile more closely. It will reclassify all customers, so the four-group solution does not mean simply splitting one of the three groups.

The "degree of separation" in the table corresponds to the silhouette coefficient: the higher it is, the more similar the same group of customers are and the easier it is for different groups to separate. It's not accuracy.

| Plan | degree of separation | Minimum group proportion | What did you see more than the overview? |
| --- | --- | --- | --- |
| 3 group | 0.2242 | 27.3% | Establish a general outline: frequent purchases, low-frequency purchases, and no purchases for a long time. |
| 4 group | 0.2234 | 17.1% | A "high single transaction, low frequency" group appears: a typical single transaction is 417, purchased 2 times in half a year, and 53 days since the last purchase. |
| 5 group | 0.2034 | 12.9% | Identify a group of customers with low consumption and few types of goods purchased: half-year amount 129, type of goods 7. |
| 6 group | 0.2074 | 7.1% | High-frequency customers continue to be segmented: the most frequent group buys 9 times in half a year, accounting for 7.1%; groups with similar consumption and different recent activities are also separated. |

**The current main model follows the 3 group.** The score difference between the third group and the fourth group is only 0.0007, so it cannot be said that the third group clearly wins. Three groups are suitable for talking about the overall structure; four groups provide valuable business details that are worth retaining in the display. The five and six groups can be used as a more detailed observation perspective, and the degree of separation has not continued to improve.

### Among the four group solutions, the "high single transaction, low frequency" group deserves to be discussed separately.

| Groups in the four-group scheme | Record proportion | since last purchase | Number of purchases in half a year | Typical single transaction amount |
| --- | --- | --- | --- | --- |
| Frequent purchase type (C 2 ) | 17.1% | 13 days | 6 times | 345 |
| High single stroke, low frequency type (C 1 ) | 27.8% | 53 days | 2 times | 417 |
| Low frequency, short visible history type (C 0 ) | 26.6% | 35 days | 1 times | 202 |
| Long time unpurchased type (C 3 ) | 28.6% | 131 days | 1 times | 231 |

"High single transaction, low frequency type" accounts for **27.8%**, and the typical half-year consumption amount is **987**. This reminds us: high consumption amount can come from "buying frequently" or "buying more each time". Whether the two types of customers are suitable for the same maintenance rhythm requires further research.

The six-group plan also found a pair of groups worthy of comparison: both groups bought about **3 times** in half a year, and the amounts were similar (**1,027 / 1,024**), but the time since the last purchase was **14 days / 55 days** respectively. Their consumption scale in the past was similar, but their current purchasing stages are different. This is a difference that is easily overlooked when stratified only by total consumption.

## 3. What features were newly discovered?

The original features tell us "how much was bought". This time I added "Have there been any changes recently, and how often have I purchased it in the limited history?" These features are calculated from existing fields and are new expressions; real order intervals, category preferences, and return preferences still require transaction details.

| New features | Popular meaning and algorithm | How to understand |
| --- | --- | --- |
| Changes in buying pace | The number of purchases in the last 30 days − the average number of purchases per month in the previous 60 days | For example, in the last 30 days, 3 times were purchased, and in the previous 60 days, a total of 2 times were purchased, and the change is + 2. Only calculated for people who have at least 90 days of visible history. |
| Proportion of recent purchases | Number of purchases in the last 90 days ÷ Number of purchases in the last 180 days | Close to 1: purchases within half a year are concentrated in the recent period; for 0: no purchases were made in the last 90 days. Customers with short history may naturally have higher prices. |
| Number of monthly purchases converted based on visible history | Number of purchases in half a year ÷ Number of visible historical days × 30; limited number of days 30 — 180 days | For the same purchase 2 times, the purchase frequency should be treated differently if it occurs for two months and for half a year. |
| Is history long enough? | Is there at least 90 days of visible purchase history? | Prompt whether there is enough observation time for changes in rhythm to avoid taking the lack of history as no purchase. |

In addition, two markers have been added to facilitate finding people: **Recent Acceleration** = The rhythm change has increased by at least 1 times/month; **Temporary interruption after repeated purchases** = There has been a visible history of 90 days, and purchases have been made at least 2 times in the past 90 days, but no purchases have been made in the last 30 days.

### Do new features really make clustering better? Actually tried it

After adding the above 4 item to the original 8 item, compare the 3 — 6 group. The new feature scheme selected **3 group**. It singles out a group of **20.1%** whose first visible purchase was typically only **48 days** ago, with **98.5%** having less than 90 days of visible history. Explain that the new features allow the model to emphasize "what observation stage the customer is in".

Measured by the newly added 12 features, the new solution scores **0.3145**. But the "ruler" of measurement has changed, and this number cannot be directly compared with the original score. Reusing the original 8 features to measure the new classification method for the same 3 group, the score is **0.2056**, and the original solution is **0.2242**. Therefore, the gain at this stage is that a new observation dimension has been found, and there is no evidence that the original model should be replaced with a new solution.

## 4. Will these characteristics still hold true over time?

Use the original model to group customers from 2011, to 6 to 8 months, and then see if they actually purchase again after 30 days. The following results are only used for post-mortem inspection, and future purchase records will not be included in the grouping.

| group | Actual repurchase ratio during training period | Actual repurchase ratio during validation period |
| --- | --- | --- |
| Frequent buyers | 47.9% | 47.8% |
| Infrequent buyers | 18.9% | 19.2% |
| Long inactive | 14.1% | 14.3% |

![Actual repurchase comparison of each group in the next period of time](figures/cluster_followup_story.png)

The repurchase ratio of the frequent buyers group in the three verification months is **47.4% — 48.5%**, and the repurchase ratio of the group that has not purchased for a long time is **12.1% — 16.3%**. This difference exists in the three verification months, indicating that the profiles have subsequent behavioral correspondence. It also shows that "I haven't bought it for a long time" does not mean that I will never buy it again.

### A counter-intuitive discovery: recent acceleration may not necessarily continue to accelerate

In **Frequently Visited and Frequently Buyed**, only records with a visible history of 90 days are compared:

| Validation period subgroup | Number of customer records per month | Actual repurchase ratio after 30 days |
| --- | --- | --- |
| The number of recent purchases has increased significantly | 769 | 40.8% |
| did not reach the above-mentioned increase rate | 1,787 | 50.8% |

This result reminds us: there is a rhythm to buying, and the recent concentrated buying may not continue into the next month. Whether it is related to concentrated replenishment, seasonality or customer composition, the current data cannot be confirmed. "Changes in purchasing rhythm" can be submitted to the stage five model for testing, but cannot be written directly as "the more active you are, the easier it is to repurchase".

### Another detail: People who haven't bought yet may still have the basis to buy.

In the **low-frequency purchase type**, there are **190 records** of "temporary interruption after repeated purchases", and the repurchase ratio 30 days after the validation period is **32.1%**; the remaining records in the same group that have a history of 90 days are **18.1%** ( 1,550 records).

This tag includes both "purchased more than once in the past" and "recently paused," so the higher repurchase rate may come from an existing purchase base. It is suitable for distinguishing customer status, but it cannot be used to interpret "pause" as the reason for repurchase.

The above are exploratory observations. The same customer can contribute records for multiple months and cannot be used as an independent sample for ordinary significance testing; the "prominent characteristics" here refer to observable differences, and there is no claim of statistical significance or effective marketing measures. The testing period is still left for final evaluation.

## 5. What reference does simple sorting set for subsequent prediction models?

Assume that you can only view 20% customers every month: RFM At the same time, look at "whether you bought recently, how many times you bought, and how much you spent" as a simple sorting reference for subsequent prediction models.

| Sorting method | A total of records selected during the validation period | which will be purchased again after 30 days | Repurchase ratio within the list |
| --- | --- | --- | --- |
| Recent, times and amount ( RFM ) | 1,631 | 910 | 55.8% |

Subsequent complex models should be compared with this simple approach to determine whether the added complexity is worth it. This is a look back at the historical list and does not mean that marketing campaigns resulted in these purchases.

## 6. What can you say when reporting?

> We first use three groups to capture the overall structure: frequent buyers, low-frequency buyers, and long-term non-buyers. Then we refine the grouping and find that high consumption may come from high-frequency purchases or higher single amounts; people with similar consumption may have different levels of recent activity. After further constructing the buying rhythm and visible historical characteristics, it is found that the recent acceleration does not necessarily extend into the next month. Therefore, the value of customer segmentation is to clarify these differences and raise testable questions for subsequent repurchase predictions.

## Data files and reproduction

- All qualified snapshots **54,040**; the training period is used to discover groupings, and the validation period is used for subsequent behavioral control.
- Run `python stage4/kmeans.py` in the project root directory to regenerate analysis and reports; run `python stage4/kmeans.py --check` to check existing results.
- `data/` holds the result table, `figures/` holds the two pictures quoted in the text, and `model/` holds the loadable model.

| File | Content and purpose |
| --- | --- |
| [Customer results summary table](data/customer_results.csv) | One row per customer per month: original features, new features, clustering, center distance, two sortings, and top 20% tags. |
| [Group profile summary](data/cluster_profiles.csv) | Profiles of each group in the original and newly added feature schemes; filter by feature_set, k, split. |
| [Comparison of grouping schemes](data/model_comparison.csv) | Comparison of k= 3 — 6 features of the original 8 term and the newly added 12 term, totaling 8 solutions. |
| [Sort Baseline Performance](data/baseline_metrics.csv) | Month-by-month and summary sorting performance during training and validation periods. |
| [Basis for behavioral discovery](data/behavior_evidence.json) | Same-day consumption share, purchase rhythm subgroups, and monthly repurchase comparisons are combined into one record. |
| [Model package](model/segmentation_bundle.joblib) | The main model, RFM cut point, and new feature exploration model share one file. |
| [Run record](run_manifest.json) | Enter hashes, versions, configurations, date partitioning, and preprocessing statistics. |

In the profile table, `feature_set=original_8` is the original plan, `enhanced_12` is the new feature exploration plan; `is_main_model=True` is the main model used in the text. `split` distinguishes between training, verification and testing; `snapshot_count` is the number of monthly customer records, `snapshot_share` is the proportion, `_median` is the median, and `_mean` is the mean. The actual repurchase ratio during the test period is left blank.

<details>
<summary>Instructions for using methods and models (click to expand)</summary>

**Training boundaries.** Following the upstream time division, filling, transformation and scaling are only fitted during the training period; customer ID, date and future labels are not entered for grouping. Whether the original transaction and the last period of observation are complete still need to be corroborated by the transaction details of stages two and three.

**Group method.** The original 8 feature is filled with the median of the training period, and the column with a skewness greater than 1 is log 1 p, and then the scale is unified. Four k use the same batch of 4,000 training records to compare the degree of separation; random seeds 42, and each scheme tries 20 initial centers. First, the minimum group proportion is required to reach 5%, and then the solution with the highest degree of separation is selected. The new feature scheme allows purchase rhythm changes to be negative, using a logarithmic transformation that preserves the sign, and is fitted independently.

**Sort definition.** The recent purchase score is the inverse of the number of days since last purchase. RFM is scored according to the quintile cut-off point of the training period, R reverse, F/M forward, and the 1 - 5 points are added together; the same value is in the same bracket. Each date is preceded by 20% (the number of people is rounded up), and the same points are sorted in ascending order according to the customer ID text. The summary list performance is based on the total of the selected records month by month; the score is the basis for sorting.

**Load model.** Run in the project root directory:

```python
import joblib
bundle = joblib.load('stage4/outputs/stage4/model/segmentation_bundle.joblib')
X = snapshots[bundle['features']]
clusters = bundle['pipeline'].predict(X)
distances = bundle['pipeline'].transform(X)
# The exploration model is located in bundle['exploratory'], and the input is first generated using stage 4.kmeans.derive_features.
```

The center distance of the main model is located in the feature space after unified scale, which can be used in stage five; rolling verification should be refitted during each fold training period. The exploration models in the model package use the importable transformation functions in stage 4.transforms. Dependency versions are recorded in run_manifest.json.

</details>
