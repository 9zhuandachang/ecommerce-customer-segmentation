# 复购概率 × 购买后金额：开发期实验

分类器复用阶段五选定的 A 组设置，并按阶段六方法只用成熟历史校准；条件金额仍用 20 个截点前特征。1–5 月曾用于复购模型参数选择，6–8 月也已查看；均非独立金额测试。9–11 月未参与本实验。

| 期间 | 方法 | 全体 MAE | 购买者 MAE | Top20 消费覆盖 | 总额偏差 | Top20 群体偏差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | extended_two_stage | 175.13 | 412.63 | 72.17% | +25.28% | +12.60% |
| rolling_jan_may | stage6_probability_times_gamma | 163.29 | 431.91 | 71.62% | +6.44% | -0.00% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | extended_two_stage | 188.85 | 439.10 | 73.19% | -8.32% | -14.54% |
| known_jun_aug | stage6_probability_times_gamma | 189.76 | 452.17 | 71.94% | -11.26% | -14.86% |

| 期间 | 购买 AP | 购买 Brier | 预测购买人数偏差 |
| --- | ---: | ---: | ---: |
| rolling_jan_may | 0.4996 | 0.1388 | +6.08% |
| known_jun_aug | 0.5751 | 0.1606 | -3.44% |

## 事前固定的 1–5 月推进条件

- `buyer_mae_reduction_at_least_5pct`：未满足
- `overall_mae_increase_at_most_2pct`：满足
- `top20_capture_not_lower`：满足
- `aggregate_bias_abs_not_larger`：满足

是否值得进入新的独立检验：**否**。开发期结果只能用于研究；不据此接入仪表板。
逐月结果、开发期逐客户预测、训练与校准标签截止及输入哈希见同目录。
