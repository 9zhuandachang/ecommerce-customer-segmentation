# 高额未来消费的截点前信号：事后审计

事先口径见 `tail_signal_protocol.md`。高额组由真实未来消费定义，只能用于诊断；所有排名分数和特征本身均为截点前可得。所有月份此前已查看，不是独立验收。

| 期间 | 购买快照 | 实际前 1% 人数 | 该组实际消费份额 | 模型预测/该组实际 | 该组进入预测 Top20 | 历史 180 天金额中位百分位 | 未来/旧月均中位倍数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | 3,414 | 37 | 19.26% | 41.32% | 97.30% | 99.8% | 1.40× |
| known_jun_aug | 2,177 | 24 | 25.42% | 39.23% | 100.00% | 99.7% | 1.97× |
| viewed_sep_nov | 3,025 | 32 | 25.39% | 43.53% | 96.88% | 99.7% | 2.25× |

历史金额百分位在同月全部 eligible 客户中计算。“未来/旧月均”中的未来结果只用于事后波动审计，不能作为预测时的输入。

## 截点前特征：高额组与其他实际购买者

| 期间 | 特征 | 高额组 | 其他购买者 |
| --- | --- | ---: | ---: |
| rolling_jan_may | monetary_value_180d | 39112.42 | 1162.98 |
| rolling_jan_may | max_invoice_value_180d | 14662.58 | 470.10 |
| rolling_jan_may | monetary_value_90d | 17085.54 | 502.86 |
| rolling_jan_may | invoice_frequency_180d | 17.00 | 3.00 |
| rolling_jan_may | recency_days | 14.00 | 37.00 |
| rolling_jan_may | bulk_spend_share_180d | 98.6% | 47.6% |
| rolling_jan_may | median_unit_price_180d | 1.69 | 1.95 |
| rolling_jan_may | non_uk_180d | 35.1% | 9.3% |
| rolling_jan_may | prior_year_available | 100.0% | 100.0% |
| known_jun_aug | monetary_value_180d | 34820.21 | 960.90 |
| known_jun_aug | max_invoice_value_180d | 8804.95 | 411.87 |
| known_jun_aug | monetary_value_90d | 19124.97 | 527.53 |
| known_jun_aug | invoice_frequency_180d | 15.50 | 3.00 |
| known_jun_aug | recency_days | 11.50 | 28.00 |
| known_jun_aug | bulk_spend_share_180d | 99.0% | 46.2% |
| known_jun_aug | median_unit_price_180d | 1.98 | 1.95 |
| known_jun_aug | non_uk_180d | 37.5% | 9.3% |
| known_jun_aug | prior_year_available | 100.0% | 100.0% |
| viewed_sep_nov | monetary_value_180d | 35737.47 | 880.93 |
| viewed_sep_nov | max_invoice_value_180d | 5433.57 | 411.60 |
| viewed_sep_nov | monetary_value_90d | 21415.03 | 475.03 |
| viewed_sep_nov | invoice_frequency_180d | 20.50 | 3.00 |
| viewed_sep_nov | recency_days | 7.50 | 32.00 |
| viewed_sep_nov | bulk_spend_share_180d | 96.0% | 45.5% |
| viewed_sep_nov | median_unit_price_180d | 1.77 | 1.95 |
| viewed_sep_nov | non_uk_180d | 28.1% | 9.9% |
| viewed_sep_nov | prior_year_available | 100.0% | 100.0% |

## 两阶段金额分解（首轮结果后的解释性补充）

仅 1–8 月有已保存的两阶段样本外预测；以下均按事后实际高额组计算。

| 期间 | 实际高额组人数 | 平均购买概率 | 假定会买时的金额预测/实际 | 概率×金额预测/实际 | 直接模型预测/实际 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rolling_jan_may | 37 | 74.67% | 59.29% | 46.22% | 41.32% |
| known_jun_aug | 24 | 81.90% | 46.45% | 39.88% | 39.23% |

“假定会买时”的金额预测仍来自截点前模型，这里仅用真实结果选事后审计组。条件金额也显著偏低时，低估不能只归因于购买概率折减；但按实际极值选组本身也会造成向下的选择偏差。

## 按预测分数事先选出的前 1% 客户

| 期间 | 名单客户月 | 命中实际极高消费组 | 名单预测金额/实际金额 | 名单实际消费份额 |
| --- | ---: | ---: | ---: | ---: |
| rolling_jan_may | 167 | 81.08% | 102.26% | 27.50% |
| known_jun_aug | 83 | 91.67% | 79.10% | 32.90% |
| viewed_sep_nov | 90 | 84.38% | 73.88% | 31.56% |

这个名单只依据截点前的预测分数选择，因此它的预测/实际金额比更适合检查高分客户的校准；逐月数据见 `predicted_tail_monthly.csv`。

逐月高额组人数、波动和历史最大订单百分位见 `monthly_tail_metrics.csv`。所有特征的分组汇总见 `feature_signals.csv`。这些差异不能证明某特征能因果解释未来高额购买，也不能据此在已看过月份调参。
