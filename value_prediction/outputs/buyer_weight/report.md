# 已查看开发期：购买者双倍训练权重的金额模型

事前方案见 `buyer_weight_protocol.md`。只给已成熟、未来 30 天确实购买的历史训练行 2 倍权重；当前预测时不知道未来购买状态。
这不是新的独立测试，也不改变原强化目标。

| 期间 | 方法 | 全体 MAE | 购买者 MAE | Top20 消费覆盖 | 总额偏差 | Top20 群体偏差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | buyer_weight_2x_tweedie | 207.14 | 387.25 | 71.48% | +61.54% | +31.66% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | buyer_weight_2x_tweedie | 214.63 | 402.47 | 73.16% | +22.67% | +3.72% |

## 运行前锁定的开发期推进门槛

- `buyer_mae_reduction_at_least_5pct`：满足
- `overall_mae_increase_at_most_2pct`：未满足
- `top20_capture_not_lower`：未满足
- `aggregate_bias_abs_not_larger`：未满足

## 事后误差分解（解释取舍，不用于调权重）

| 期间 | 方法 | 零消费客户占比 | 零消费客户 MAE | 购买者 MAE |
| --- | --- | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 79.23% | 143.36 | 396.56 |
| rolling_jan_may | extended_tweedie | 79.23% | 106.64 | 420.92 |
| rolling_jan_may | buyer_weight_2x_tweedie | 79.23% | 159.92 | 387.25 |
| known_jun_aug | historic_180d | 73.28% | 122.16 | 427.12 |
| known_jun_aug | extended_tweedie | 73.28% | 96.92 | 440.47 |
| known_jun_aug | buyer_weight_2x_tweedie | 73.28% | 146.12 | 402.47 |

零消费客户的实际金额为 0，因此该组 MAE 就是平均预测金额。双倍权重减少了购买者误差，却显著抬高零消费客户预测，说明两项目标目前存在冲突。
只有运行前四项推进条件全部满足才建议进入新数据的固定测试；任何结果均不能作为原商家的独立验收。
逐月指标、客户级预测、训练标签截止和源哈希见同目录。
