# 订单金额变化特征：购买后金额研发结果

本实验只检验“若未来购买，会花多少”。1–5 月用于开发；已查看的 6–8 月仅作方向核对；9–11 月未读取。不能把条件金额当成全体客户的预计消费额。

| 期间 | 方法 | 购买者 MAE | 购买者金额偏差 | 购买者数 |
| --- | --- | ---: | ---: | ---: |
| rolling_jan_may | historic_monthly_rate | 396.56 | -19.96% | 3,414 |
| rolling_jan_may | average_prior_invoice | 360.23 | -28.97% | 3,414 |
| rolling_jan_may | conditional_l1_median | 334.51 | -5.20% | 3,414 |
| rolling_jan_may | conditional_l1_order_dynamics | 328.56 | -6.18% | 3,414 |
| known_jun_aug | historic_monthly_rate | 427.12 | -39.71% | 2,177 |
| known_jun_aug | average_prior_invoice | 410.44 | -38.24% | 2,177 |
| known_jun_aug | conditional_l1_median | 352.18 | -18.08% | 2,177 |
| known_jun_aug | conditional_l1_order_dynamics | 352.23 | -18.00% | 2,177 |

## 事前固定的研发推进条件

- `buyer_mae_reduction_at_least_5pct`：未满足
- `at_least_four_of_five_months_not_worse`：满足
- `absolute_buyer_amount_bias_not_larger`：未满足

值得进入新的独立检验：**否**。这不改变原无条件金额强化目标，也不能作为系统接入依据。
逐月、逐客户、训练标签截止、特征逐分核对及哈希见同目录。
