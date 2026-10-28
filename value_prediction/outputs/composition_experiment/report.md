# 快照前购买构成特征：开发期固定候选

方案在模型运行前固定于 `composition_protocol.md`。新特征来自每个快照前 180 天交易，仅用已成熟历史标签训练。2011 年 6–8 月已查看，以下均非独立验收。

| 期间 | 方法 | 全体 MAE | 购买者 MAE | Top20 消费覆盖 | 总额偏差 | Top20 群体偏差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | composition_tweedie | 172.30 | 415.14 | 72.06% | +22.17% | +10.96% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | composition_tweedie | 188.96 | 439.06 | 73.25% | -6.54% | -12.12% |

## 运行前锁定的开发期推进门槛

- `buyer_mae_reduction_at_least_5pct`：未满足
- `overall_mae_increase_at_most_2pct`：满足
- `top20_capture_not_lower`：满足
- `aggregate_bias_abs_not_larger`：未满足

只有四项都满足才建议进入新的外部固定测试。不根据 6–8 月结果改变三项新特征或判定门槛。
逐月、客户级结果、训练截止和来源哈希见同目录。
