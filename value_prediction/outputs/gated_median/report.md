# 概率门控金额中位数：开发期结果

阈值 0.5 和所有输入模型均在本次结果前固定。1–5 月和 6–8 月都已查看，仅供研发判断；9–11 月未参与。该点预测不是期望消费额。

| 期间 | 方法 | 全体 MAE | 购买者 MAE | Top20 覆盖 | 总额偏差 | 非零预测人数 | Top20 可完整区分 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | 16,435 | 是 |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | 16,435 | 是 |
| rolling_jan_may | stage6_probability_times_gamma | 163.29 | 431.91 | 71.62% | +6.44% | 16,435 | 是 |
| rolling_jan_may | probability_gated_median | 129.05 | 517.75 | 57.77% | -36.21% | 1,424 | 否 |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | 8,146 | 是 |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | 8,146 | 是 |
| known_jun_aug | stage6_probability_times_gamma | 189.76 | 452.17 | 71.94% | -11.26% | 8,146 | 是 |
| known_jun_aug | probability_gated_median | 158.05 | 524.35 | 61.25% | -46.30% | 932 | 否 |

## 事前固定的研发推进条件

- `buyer_mae_reduction_at_least_5pct`：未满足
- `overall_mae_increase_at_most_2pct`：满足
- `top20_capture_not_lower`：未满足
- `aggregate_bias_abs_not_larger`：未满足
- `all_monthly_top20_distinguishable`：未满足

可推进新数据检验：**否**。原强化目标仍需单独检验，当前结果不构成独立验收。
逐客户预测、逐月指标与源文件哈希见同目录。
