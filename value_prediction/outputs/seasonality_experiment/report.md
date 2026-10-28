# 日历月份候选：按时间向前研究

事先固定的唯一差别是快照月份的正弦和余弦；其余模型设置与原 20 特征 Tweedie 相同。2011 年 6–11 月均已查看，只用于开发确认或回顾诊断。

| 期间 | 方法 | 全体 MAE | 购买者 MAE | Top20 消费覆盖 | 总额偏差 | Top20 群体偏差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | calendar_tweedie | 166.95 | 439.67 | 73.16% | +13.98% | +8.48% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | calendar_tweedie | 208.43 | 410.84 | 73.49% | +21.61% | +9.56% |
| viewed_sep_nov | historic_180d | 263.47 | 529.85 | 67.28% | -20.77% | -16.92% |
| viewed_sep_nov | extended_tweedie | 252.58 | 535.01 | 69.47% | -26.16% | -24.09% |
| viewed_sep_nov | calendar_tweedie | 267.48 | 502.95 | 69.10% | -8.83% | -13.61% |

## 预测分数事先选出的前 1% 客户：金额尺度诊断

| 期间 | 方法 | 名单客户月 | 预测金额/实际金额 | 实际消费覆盖 |
| --- | --- | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 167 | 146.84% | 29.91% |
| rolling_jan_may | extended_tweedie | 167 | 102.26% | 27.50% |
| rolling_jan_may | calendar_tweedie | 167 | 110.70% | 27.65% |
| known_jun_aug | historic_180d | 83 | 98.97% | 30.62% |
| known_jun_aug | extended_tweedie | 83 | 79.10% | 32.90% |
| known_jun_aug | calendar_tweedie | 83 | 87.01% | 32.87% |
| viewed_sep_nov | historic_180d | 90 | 73.90% | 31.29% |
| viewed_sep_nov | extended_tweedie | 90 | 73.88% | 31.56% |
| viewed_sep_nov | calendar_tweedie | 90 | 77.47% | 31.94% |

该名单仅按预测分数选择；前 1% 指标是诊断项，不用于事后选择日历模型或调整推进门槛。

## 仅 1–5 月的预设研发推进门槛

- `buyer_mae_reduction_at_least_5pct`：未满足
- `overall_mae_increase_at_most_2pct`：满足
- `top20_capture_not_lower`：满足
- `aggregate_bias_abs_not_larger`：满足

全部满足也只说明值得带到新的固定测试；这些已查看月份不能提供同商家独立验收。逐月、客户级预测、训练标签截止与文件哈希见同目录。
