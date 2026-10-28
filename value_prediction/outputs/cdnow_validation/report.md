# CDNOW 跨商家消费额预测检验

按 `cdnow_protocol.md` 事先固定的规则，使用完整的真实交易来源。原始文件仅放系统临时目录，仓库仅留聚合结果。
金额为美元，客户只来自 1997 年第一季度首次购买者；商品 ID 不可用，商品多样性固定为 1。此结果不能代替原商家的新月份独立验收。

源交易 69,659 行、客户 23,570 人；排除零金额交易 80 行后，合格客户 23,502 人，客户月快照 109,629 条。零金额行金额之和为 0，交易金额逐分核对通过。
最后三个月可行性（每月 ≥1,000 快照、≥100 未来购买者）：满足。

| 截点 | 分区 | 方法 | 客户数 | 购买者 | MAE | 购买者 MAE | Top20 覆盖 | 总额偏差 | Top20 群体偏差 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1997-10-01 | development | historic_180d | 7,249 | 1,318 | 16.28 | 34.40 | 54.73% | +51.31% | +67.74% |
| 1997-10-01 | development | recent_90d | 7,249 | 1,318 | 15.27 | 36.98 | 53.07% | +35.68% | +85.80% |
| 1997-10-01 | development | extended_tweedie | 7,249 | 1,318 | 12.76 | 40.31 | 56.11% | -16.04% | -11.73% |
| 1997-10-01 | development | extended_tweedie_lag_calibrated | 7,249 | 1,318 | 12.76 | 40.31 | 56.11% | -16.04% | -11.73% |
| 1997-11-01 | development | historic_180d | 6,734 | 1,469 | 18.16 | 41.81 | 53.34% | +9.04% | +23.57% |
| 1997-11-01 | development | recent_90d | 6,734 | 1,469 | 16.80 | 43.67 | 51.96% | -5.61% | +30.64% |
| 1997-11-01 | development | extended_tweedie | 6,734 | 1,469 | 15.95 | 45.86 | 56.53% | -24.96% | -20.24% |
| 1997-11-01 | development | extended_tweedie_lag_calibrated | 6,734 | 1,469 | 16.34 | 45.04 | 56.53% | -17.79% | -12.62% |
| 1997-12-01 | development | historic_180d | 6,612 | 1,340 | 17.77 | 38.47 | 50.04% | +36.71% | +65.65% |
| 1997-12-01 | development | recent_90d | 6,612 | 1,340 | 17.04 | 40.00 | 48.37% | +30.87% | +90.44% |
| 1997-12-01 | development | extended_tweedie | 6,612 | 1,340 | 16.11 | 39.67 | 51.40% | +12.92% | +21.95% |
| 1997-12-01 | development | extended_tweedie_lag_calibrated | 6,612 | 1,340 | 17.06 | 39.05 | 51.40% | +27.96% | +38.19% |
| 1998-01-01 | development | historic_180d | 6,343 | 1,099 | 16.39 | 34.76 | 58.75% | +68.28% | +71.58% |
| 1998-01-01 | development | recent_90d | 6,343 | 1,099 | 16.69 | 37.29 | 53.40% | +71.10% | +120.09% |
| 1998-01-01 | development | extended_tweedie | 6,343 | 1,099 | 14.35 | 35.30 | 60.06% | +30.86% | +13.16% |
| 1998-01-01 | development | extended_tweedie_lag_calibrated | 6,343 | 1,099 | 14.74 | 34.81 | 60.06% | +38.47% | +19.73% |
| 1998-02-01 | development | historic_180d | 5,961 | 1,209 | 17.01 | 37.01 | 53.64% | +35.00% | +50.71% |
| 1998-02-01 | development | recent_90d | 5,961 | 1,209 | 17.72 | 38.29 | 50.63% | +43.45% | +93.39% |
| 1998-02-01 | development | extended_tweedie | 5,961 | 1,209 | 14.03 | 40.54 | 56.19% | -15.20% | -9.81% |
| 1998-02-01 | development | extended_tweedie_lag_calibrated | 5,961 | 1,209 | 13.99 | 40.63 | 56.19% | -16.12% | -10.79% |
| 1998-03-01 | development | historic_180d | 5,937 | 1,430 | 17.95 | 37.92 | 53.01% | +8.91% | +23.32% |
| 1998-03-01 | development | recent_90d | 5,937 | 1,430 | 17.90 | 41.00 | 48.98% | +1.72% | +45.48% |
| 1998-03-01 | development | extended_tweedie | 5,937 | 1,430 | 15.29 | 43.09 | 55.49% | -36.01% | -29.01% |
| 1998-03-01 | development | extended_tweedie_lag_calibrated | 5,937 | 1,430 | 15.26 | 43.20 | 55.49% | -36.70% | -29.79% |
| 1998-04-01 | locked_test | historic_180d | 6,055 | 1,074 | 16.08 | 32.82 | 57.46% | +74.48% | +83.05% |
| 1998-04-01 | locked_test | recent_90d | 6,055 | 1,074 | 15.47 | 34.41 | 54.37% | +64.20% | +111.86% |
| 1998-04-01 | locked_test | extended_tweedie | 6,055 | 1,074 | 14.21 | 32.59 | 59.91% | +43.62% | +34.52% |
| 1998-04-01 | locked_test | extended_tweedie_lag_calibrated | 6,055 | 1,074 | 14.67 | 32.35 | 59.91% | +52.09% | +42.44% |
| 1998-05-01 | locked_test | historic_180d | 5,900 | 1,136 | 15.86 | 31.68 | 55.65% | +59.44% | +72.68% |
| 1998-05-01 | locked_test | recent_90d | 5,900 | 1,136 | 15.36 | 33.26 | 52.13% | +50.81% | +101.27% |
| 1998-05-01 | locked_test | extended_tweedie | 5,900 | 1,136 | 12.63 | 35.81 | 58.00% | -3.84% | -2.37% |
| 1998-05-01 | locked_test | extended_tweedie_lag_calibrated | 5,900 | 1,136 | 12.80 | 35.41 | 58.00% | +0.36% | +1.89% |
| 1998-06-01 | locked_test | historic_180d | 5,529 | 1,089 | 16.75 | 37.15 | 55.31% | +33.68% | +44.54% |
| 1998-06-01 | locked_test | recent_90d | 5,529 | 1,089 | 16.50 | 38.20 | 54.71% | +31.09% | +65.17% |
| 1998-06-01 | locked_test | extended_tweedie | 5,529 | 1,089 | 13.79 | 40.16 | 58.58% | -13.35% | -12.24% |
| 1998-06-01 | locked_test | extended_tweedie_lag_calibrated | 5,529 | 1,089 | 13.87 | 39.92 | 58.58% | -11.39% | -10.25% |

## 最后三个月合并

| 方法 | MAE | 购买者 MAE | Top20 覆盖 | 总额偏差 | Top20 群体偏差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| historic_180d | 16.22 | 33.86 | 56.09% | +54.97% | +66.12% |
| recent_90d | 15.76 | 35.27 | 53.75% | +47.98% | +91.53% |
| extended_tweedie | 13.54 | 36.20 | 58.81% | +7.64% | +5.90% |
| extended_tweedie_lag_calibrated | 13.79 | 35.90 | 58.81% | +12.38% | +10.53% |

## 事先固定的强化目标

- `extended_tweedie`：未全部满足。mae_10pct_better=是；buyer_mae_5pct_better=否；top20_capture_3pp_higher=否；aggregate_bias_within_5pct=否；top20_group_bias_within_10pct=是；all_months_mae_not_worse=是；all_months_top20_not_worse=是
- `extended_tweedie_lag_calibrated`：未全部满足。mae_10pct_better=是；buyer_mae_5pct_better=否；top20_capture_3pp_higher=否；aggregate_bias_within_5pct=否；top20_group_bias_within_10pct=否；all_months_mae_not_worse=是；all_months_top20_not_worse=是

各月训练标签截止、原始金额核对及来源哈希见同目录文件。任何跨商家通过都只属于方法可迁移性证据，不构成原商家金额模型上线许可。
