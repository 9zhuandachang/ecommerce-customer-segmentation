# 消费额预测开发期向前验证

本报告只使用 2011 年 1–8 月开发期，9–11 月测试行未参与训练或评估。
每月重新训练时，仅使用未来 30 天标签窗口已经结束的更早客户快照。
1–5 月用于滚动开发比较；6–8 月是此前已看过的开发确认期。

| 期间 | 方法 | MAE | 购买者 MAE | Top20 消费额覆盖 | 总额偏差 | Top20 群体偏差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | historic_180d | 195.95 | 396.56 | 71.05% | +64.51% | +62.52% |
| rolling_jan_may | recent_90d | 185.38 | 423.56 | 68.43% | +44.44% | +63.37% |
| rolling_jan_may | base_tweedie | 176.46 | 430.37 | 70.28% | +19.37% | +7.74% |
| rolling_jan_may | extended_tweedie | 171.92 | 420.92 | 71.55% | +19.91% | +8.93% |
| rolling_jan_may | extended_tweedie_lag_calibrated | 166.75 | 433.68 | 71.55% | +8.62% | -1.18% |
| rolling_jan_may | extended_tweedie_history_available | 155.36 | 438.13 | 73.21% | +1.58% | -0.76% |
| rolling_jan_may | extended_tweedie_previous_year | 156.36 | 443.76 | 72.48% | +1.48% | +0.38% |
| rolling_jan_may | extended_tweedie_previous_year_recent_weighted | 155.24 | 450.35 | 72.89% | -1.86% | -2.78% |
| rolling_jan_may | extended_poisson | 186.77 | 410.36 | 72.51% | +36.94% | +16.66% |
| rolling_jan_may | extended_l2 | 191.20 | 423.15 | 71.89% | +45.30% | +32.82% |
| rolling_jan_may | extended_two_stage | 175.13 | 412.63 | 72.17% | +25.28% | +12.60% |
| rolling_jan_may | extended_two_stage_previous_year | 158.84 | 434.66 | 72.59% | +5.27% | +2.24% |
| known_jun_aug | historic_180d | 203.66 | 427.12 | 71.38% | +8.15% | +6.43% |
| known_jun_aug | recent_90d | 209.43 | 433.88 | 67.36% | +15.88% | +26.25% |
| known_jun_aug | base_tweedie | 190.52 | 458.33 | 72.09% | -11.02% | -15.22% |
| known_jun_aug | extended_tweedie | 188.73 | 440.47 | 73.19% | -7.12% | -12.66% |
| known_jun_aug | extended_tweedie_lag_calibrated | 191.26 | 436.37 | 73.19% | -2.35% | -8.14% |
| known_jun_aug | extended_tweedie_history_available | 181.82 | 450.15 | 72.77% | -15.15% | -15.64% |
| known_jun_aug | extended_tweedie_previous_year | 177.92 | 442.39 | 73.33% | -17.33% | -18.49% |
| known_jun_aug | extended_tweedie_previous_year_recent_weighted | 179.57 | 448.44 | 73.14% | -15.89% | -16.09% |
| known_jun_aug | extended_poisson | 199.53 | 447.96 | 72.94% | +3.73% | -3.72% |
| known_jun_aug | extended_l2 | 199.43 | 447.14 | 72.19% | +2.87% | -2.63% |
| known_jun_aug | extended_two_stage | 188.85 | 439.10 | 73.19% | -8.32% | -14.54% |
| known_jun_aug | extended_two_stage_previous_year | 179.90 | 446.94 | 73.47% | -17.93% | -20.14% |
| rolling_jan_aug | historic_180d | 198.51 | 408.46 | 71.19% | +41.51% | +39.57% |
| rolling_jan_aug | recent_90d | 193.35 | 427.58 | 67.99% | +32.79% | +48.37% |
| rolling_jan_aug | base_tweedie | 181.12 | 441.26 | 71.02% | +6.97% | -1.77% |
| rolling_jan_aug | extended_tweedie | 177.50 | 428.53 | 72.22% | +8.88% | +0.00% |
| rolling_jan_aug | extended_tweedie_lag_calibrated | 174.87 | 434.72 | 72.22% | +4.15% | -4.06% |
| rolling_jan_aug | extended_tweedie_history_available | 164.13 | 442.81 | 73.03% | -5.25% | -6.81% |
| rolling_jan_aug | extended_tweedie_previous_year | 163.50 | 443.22 | 72.82% | -6.19% | -7.38% |
| rolling_jan_aug | extended_tweedie_previous_year_recent_weighted | 163.30 | 449.60 | 72.99% | -7.58% | -8.22% |
| rolling_jan_aug | extended_poisson | 191.00 | 425.00 | 72.68% | +23.39% | +8.32% |
| rolling_jan_aug | extended_l2 | 193.93 | 432.49 | 72.01% | +27.99% | +18.32% |
| rolling_jan_aug | extended_two_stage | 179.68 | 422.94 | 72.59% | +11.57% | +1.44% |
| rolling_jan_aug | extended_two_stage_previous_year | 165.82 | 439.44 | 72.95% | -4.19% | -6.96% |

与同期间历史金额基线相比：

- rolling_jan_may / recent_90d：MAE -5.40%；Top20 覆盖 -2.62 个百分点。
- rolling_jan_may / base_tweedie：MAE -9.95%；Top20 覆盖 -0.77 个百分点。
- rolling_jan_may / extended_tweedie：MAE -12.26%；Top20 覆盖 +0.50 个百分点。
- rolling_jan_may / extended_tweedie_lag_calibrated：MAE -14.91%；Top20 覆盖 +0.50 个百分点。
- rolling_jan_may / extended_tweedie_history_available：MAE -20.72%；Top20 覆盖 +2.15 个百分点。
- rolling_jan_may / extended_tweedie_previous_year：MAE -20.21%；Top20 覆盖 +1.42 个百分点。
- rolling_jan_may / extended_tweedie_previous_year_recent_weighted：MAE -20.78%；Top20 覆盖 +1.84 个百分点。
- rolling_jan_may / extended_poisson：MAE -4.69%；Top20 覆盖 +1.46 个百分点。
- rolling_jan_may / extended_l2：MAE -2.43%；Top20 覆盖 +0.84 个百分点。
- rolling_jan_may / extended_two_stage：MAE -10.63%；Top20 覆盖 +1.11 个百分点。
- rolling_jan_may / extended_two_stage_previous_year：MAE -18.94%；Top20 覆盖 +1.53 个百分点。
- known_jun_aug / recent_90d：MAE +2.83%；Top20 覆盖 -4.02 个百分点。
- known_jun_aug / base_tweedie：MAE -6.45%；Top20 覆盖 +0.71 个百分点。
- known_jun_aug / extended_tweedie：MAE -7.33%；Top20 覆盖 +1.82 个百分点。
- known_jun_aug / extended_tweedie_lag_calibrated：MAE -6.09%；Top20 覆盖 +1.82 个百分点。
- known_jun_aug / extended_tweedie_history_available：MAE -10.72%；Top20 覆盖 +1.39 个百分点。
- known_jun_aug / extended_tweedie_previous_year：MAE -12.64%；Top20 覆盖 +1.95 个百分点。
- known_jun_aug / extended_tweedie_previous_year_recent_weighted：MAE -11.83%；Top20 覆盖 +1.76 个百分点。
- known_jun_aug / extended_poisson：MAE -2.03%；Top20 覆盖 +1.56 个百分点。
- known_jun_aug / extended_l2：MAE -2.08%；Top20 覆盖 +0.81 个百分点。
- known_jun_aug / extended_two_stage：MAE -7.27%；Top20 覆盖 +1.81 个百分点。
- known_jun_aug / extended_two_stage_previous_year：MAE -11.67%；Top20 覆盖 +2.09 个百分点。
- rolling_jan_aug / recent_90d：MAE -2.60%；Top20 覆盖 -3.19 个百分点。
- rolling_jan_aug / base_tweedie：MAE -8.76%；Top20 覆盖 -0.17 个百分点。
- rolling_jan_aug / extended_tweedie：MAE -10.59%；Top20 覆盖 +1.04 个百分点。
- rolling_jan_aug / extended_tweedie_lag_calibrated：MAE -11.91%；Top20 覆盖 +1.04 个百分点。
- rolling_jan_aug / extended_tweedie_history_available：MAE -17.32%；Top20 覆盖 +1.84 个百分点。
- rolling_jan_aug / extended_tweedie_previous_year：MAE -17.63%；Top20 覆盖 +1.64 个百分点。
- rolling_jan_aug / extended_tweedie_previous_year_recent_weighted：MAE -17.73%；Top20 覆盖 +1.81 个百分点。
- rolling_jan_aug / extended_poisson：MAE -3.78%；Top20 覆盖 +1.50 个百分点。
- rolling_jan_aug / extended_l2：MAE -2.31%；Top20 覆盖 +0.83 个百分点。
- rolling_jan_aug / extended_two_stage：MAE -9.49%；Top20 覆盖 +1.40 个百分点。
- rolling_jan_aug / extended_two_stage_previous_year：MAE -16.47%；Top20 覆盖 +1.76 个百分点。

逐月结果、每条客户预测、训练截止证明和输入哈希见同目录。这里的结果仅供下一轮设计，不能作为最终独立测试。
