# 未来 30 天金额预测区间：开发期检查

固定方案：`interval_protocol.md`。无条件金额采用去年同期直接模型；“若购买”金额采用条件中位数。区间只利用当时已结束的过去月份预测残差。
名义覆盖率 90%，同时比较原金额绝对残差与 `log1p` 金额残差。6–8 月此前已查看，以下是开发确认结果而非独立验收。

| 期间 | 目标 | 形式 | 客户组 | 经验覆盖 | 高于上界 | 平均宽度 | 中位宽度 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| rolling_feb_may | unconditional | absolute | all | 91.12% | 7.04% | 408.30 | 377.58 |
| rolling_feb_may | unconditional | absolute | predicted_top10 | 56.60% | 25.00% | 630.55 | 631.06 |
| rolling_feb_may | unconditional | log1p | all | 95.08% | 0.02% | 23478.23 | 7284.58 |
| rolling_feb_may | unconditional | log1p | predicted_top10 | 68.48% | 0.00% | 142994.46 | 70688.96 |
| rolling_feb_may | unconditional | absolute_grouped | all | 91.31% | 8.21% | 450.89 | 272.14 |
| rolling_feb_may | unconditional | absolute_grouped | predicted_top10 | 91.87% | 5.60% | 2000.01 | 2091.72 |
| rolling_feb_may | unconditional | log1p_grouped | all | 96.62% | 0.02% | 53803.80 | 6051.88 |
| rolling_feb_may | unconditional | log1p_grouped | predicted_top10 | 96.17% | 0.00% | 461394.58 | 228083.64 |
| rolling_feb_may | if_purchase | absolute | all | 91.45% | 6.16% | 1051.92 | 1025.06 |
| rolling_feb_may | if_purchase | absolute | predicted_top10 | 66.41% | 20.58% | 1323.94 | 1255.35 |
| rolling_feb_may | if_purchase | log1p | all | 91.35% | 3.35% | 1487.27 | 920.48 |
| rolling_feb_may | if_purchase | log1p | predicted_top10 | 84.66% | 3.88% | 4270.47 | 2214.11 |
| rolling_feb_may | if_purchase | absolute_grouped | all | 89.85% | 8.69% | 1130.68 | 680.13 |
| rolling_feb_may | if_purchase | absolute_grouped | predicted_top10 | 90.49% | 5.63% | 3358.74 | 3093.71 |
| rolling_feb_may | if_purchase | log1p_grouped | all | 91.24% | 3.49% | 1721.92 | 833.67 |
| rolling_feb_may | if_purchase | log1p_grouped | predicted_top10 | 91.26% | 1.94% | 5918.27 | 3075.40 |
| known_jun_aug | unconditional | absolute | all | 89.52% | 7.61% | 421.41 | 381.63 |
| known_jun_aug | unconditional | absolute | predicted_top10 | 44.61% | 26.72% | 644.52 | 651.74 |
| known_jun_aug | unconditional | log1p | all | 87.33% | 0.00% | 20247.95 | 7561.69 |
| known_jun_aug | unconditional | log1p | predicted_top10 | 69.12% | 0.00% | 117079.51 | 54901.04 |
| known_jun_aug | unconditional | absolute_grouped | all | 88.26% | 9.18% | 447.00 | 283.86 |
| known_jun_aug | unconditional | absolute_grouped | predicted_top10 | 89.71% | 6.74% | 1798.61 | 1682.17 |
| known_jun_aug | unconditional | log1p_grouped | all | 86.46% | 0.00% | 42436.55 | 6430.88 |
| known_jun_aug | unconditional | log1p_grouped | predicted_top10 | 84.68% | 0.00% | 351042.93 | 164487.11 |
| known_jun_aug | if_purchase | absolute | all | 90.49% | 7.53% | 939.91 | 932.18 |
| known_jun_aug | if_purchase | absolute | predicted_top10 | 62.24% | 26.79% | 1141.88 | 1140.51 |
| known_jun_aug | if_purchase | log1p | all | 90.49% | 3.77% | 1309.55 | 856.57 |
| known_jun_aug | if_purchase | log1p | predicted_top10 | 86.22% | 6.12% | 3764.68 | 1968.49 |
| known_jun_aug | if_purchase | absolute_grouped | all | 90.58% | 8.13% | 1083.23 | 671.63 |
| known_jun_aug | if_purchase | absolute_grouped | predicted_top10 | 88.52% | 8.67% | 3189.74 | 2958.97 |
| known_jun_aug | if_purchase | log1p_grouped | all | 90.26% | 3.58% | 1461.77 | 794.58 |
| known_jun_aug | if_purchase | log1p_grouped | predicted_top10 | 92.60% | 2.30% | 4843.24 | 2586.61 |

## 逐月覆盖率

| 截点 | 目标 | 形式 | 全体覆盖 | 预测前 10% 覆盖 |
| --- | --- | --- | ---: | ---: |
| 2011-02-01 | unconditional | absolute | 92.80% | 62.69% |
| 2011-02-01 | unconditional | log1p | 97.22% | 72.24% |
| 2011-02-01 | unconditional | absolute_grouped | 93.76% | 94.03% |
| 2011-02-01 | unconditional | log1p_grouped | 99.79% | 97.91% |
| 2011-02-01 | if_purchase | absolute | 92.44% | 68.07% |
| 2011-02-01 | if_purchase | log1p | 92.60% | 86.55% |
| 2011-02-01 | if_purchase | absolute_grouped | 89.76% | 91.60% |
| 2011-02-01 | if_purchase | log1p_grouped | 91.50% | 90.76% |
| 2011-03-01 | unconditional | absolute | 92.84% | 64.67% |
| 2011-03-01 | unconditional | log1p | 97.36% | 73.65% |
| 2011-03-01 | unconditional | absolute_grouped | 93.32% | 96.11% |
| 2011-03-01 | unconditional | log1p_grouped | 99.82% | 98.20% |
| 2011-03-01 | if_purchase | absolute | 92.85% | 70.45% |
| 2011-03-01 | if_purchase | log1p | 91.42% | 88.64% |
| 2011-03-01 | if_purchase | absolute_grouped | 90.56% | 90.91% |
| 2011-03-01 | if_purchase | log1p_grouped | 91.42% | 94.70% |
| 2011-04-01 | unconditional | absolute | 91.36% | 53.94% |
| 2011-04-01 | unconditional | log1p | 93.91% | 59.09% |
| 2011-04-01 | unconditional | absolute_grouped | 92.06% | 89.39% |
| 2011-04-01 | unconditional | log1p_grouped | 95.12% | 95.45% |
| 2011-04-01 | if_purchase | absolute | 90.94% | 66.41% |
| 2011-04-01 | if_purchase | log1p | 89.58% | 78.12% |
| 2011-04-01 | if_purchase | absolute_grouped | 90.94% | 89.84% |
| 2011-04-01 | if_purchase | log1p_grouped | 90.18% | 87.50% |
| 2011-05-01 | unconditional | absolute | 87.13% | 43.93% |
| 2011-05-01 | unconditional | log1p | 91.50% | 68.85% |
| 2011-05-01 | unconditional | absolute_grouped | 85.63% | 87.54% |
| 2011-05-01 | unconditional | log1p_grouped | 91.27% | 92.79% |
| 2011-05-01 | if_purchase | absolute | 89.90% | 61.03% |
| 2011-05-01 | if_purchase | log1p | 91.75% | 85.29% |
| 2011-05-01 | if_purchase | absolute_grouped | 88.42% | 89.71% |
| 2011-05-01 | if_purchase | log1p_grouped | 91.75% | 91.91% |
| 2011-06-01 | unconditional | absolute | 88.71% | 39.85% |
| 2011-06-01 | unconditional | log1p | 86.61% | 68.42% |
| 2011-06-01 | unconditional | absolute_grouped | 85.74% | 89.47% |
| 2011-06-01 | unconditional | log1p_grouped | 85.29% | 81.20% |
| 2011-06-01 | if_purchase | absolute | 91.81% | 67.97% |
| 2011-06-01 | if_purchase | log1p | 91.01% | 88.28% |
| 2011-06-01 | if_purchase | absolute_grouped | 91.95% | 92.97% |
| 2011-06-01 | if_purchase | log1p_grouped | 89.80% | 95.31% |
| 2011-07-01 | unconditional | absolute | 90.12% | 48.35% |
| 2011-07-01 | unconditional | log1p | 87.11% | 71.06% |
| 2011-07-01 | unconditional | absolute_grouped | 88.54% | 90.48% |
| 2011-07-01 | unconditional | log1p_grouped | 86.85% | 89.74% |
| 2011-07-01 | if_purchase | absolute | 90.24% | 60.47% |
| 2011-07-01 | if_purchase | log1p | 90.24% | 87.60% |
| 2011-07-01 | if_purchase | absolute_grouped | 90.66% | 86.82% |
| 2011-07-01 | if_purchase | log1p_grouped | 90.38% | 94.57% |
| 2011-08-01 | unconditional | absolute | 89.70% | 45.49% |
| 2011-08-01 | unconditional | log1p | 88.25% | 67.87% |
| 2011-08-01 | unconditional | absolute_grouped | 90.42% | 89.17% |
| 2011-08-01 | unconditional | log1p_grouped | 87.20% | 83.03% |
| 2011-08-01 | if_purchase | absolute | 89.37% | 58.52% |
| 2011-08-01 | if_purchase | log1p | 90.21% | 82.96% |
| 2011-08-01 | if_purchase | absolute_grouped | 89.09% | 85.93% |
| 2011-08-01 | if_purchase | log1p_grouped | 90.63% | 88.15% |

区间宽度与覆盖率是同一组已查看月份的经验结果。客户跨月重复、金额长尾和季节变化意味着 90% 名义值不保证未来覆盖。区间不代替原单点金额模型的强化验收，也未接入仪表板。
