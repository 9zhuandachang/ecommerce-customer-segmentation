# 购买条件下的消费额预测：开发期结果

问题是“如果这名客户未来 30 天购买，会花多少钱”。
评估时只对实际发生购买的客户计算误差；实际购买状态没有进入预测特征。
这个数值不能作为所有客户的未来消费额或替代无条件预测的购买者误差。
1–5 月及已看过的 6–8 月仅作开发分析；9–11 月未用于模型比较。

| 期间 | 方法 | 购买者 MAE | 购买者金额偏差 |
| --- | --- | ---: | ---: |
| rolling_jan_may | historic_monthly_rate | 396.56 | -19.96% |
| rolling_jan_may | average_prior_invoice | 360.23 | -28.97% |
| rolling_jan_may | conditional_gamma_mean | 355.84 | +5.65% |
| rolling_jan_may | conditional_l1_median | 334.51 | -5.20% |
| known_jun_aug | historic_monthly_rate | 427.12 | -39.71% |
| known_jun_aug | average_prior_invoice | 410.44 | -38.24% |
| known_jun_aug | conditional_gamma_mean | 358.63 | -7.17% |
| known_jun_aug | conditional_l1_median | 352.18 | -18.08% |
| rolling_jan_aug | historic_monthly_rate | 408.46 | -28.02% |
| rolling_jan_aug | average_prior_invoice | 379.78 | -32.75% |
| rolling_jan_aug | conditional_gamma_mean | 356.93 | +0.42% |
| rolling_jan_aug | conditional_l1_median | 341.39 | -10.45% |

历史每月消费速率和上次 180 天平均订单金额都是只用快照前数据的简单对照。
L1 模型预测条件中位数，适合最小化购买者绝对误差；其总金额通常偏低。
Gamma 模型预测条件均值，金额偏差更适合作为期望金额的组成部分。
逐月和逐客户输出及输入哈希见同目录。
