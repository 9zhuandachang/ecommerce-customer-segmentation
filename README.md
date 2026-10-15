# 电商客户分群与 30 天复购预测

本目录汇集 UCI Online Retail II 原始数据、已生成的清洗与客户快照结果，以及分群、预测和 Streamlit 仪表板的原型代码。PPT 和中文 proposal 位于上一级 `pre2` 目录；项目目标与验收条件见 [任务流程与验收清单](docs/项目任务流程与验收清单.md)，统一的业务范围、统计单位和导出口径见 [业务范围与数据口径](docs/业务范围与数据口径.md)。

## 目录

- `data/raw/`：两份原始年度 CSV，保持内容不变。
- `data/processed/customer_snapshot_pipeline/`：清洗审计、月度客户快照、日期划分、校验报告和运行记录。
- `src/data_cleaning.py`、`src/features.py`：清洗规则、客户快照与时间划分的唯一实现。
- `build_customer_snapshots.py`：调用上述模块，从原始 CSV 写出审计与快照产物。
- `tests/`：清洗、特征与标签窗口、时间划分的单元测试。
- `src/`、`app/`：分群、预测训练和仪表板的原型代码。
- `docs/`：任务清单、业务与数据口径、快照流水线说明。

## 已完成的数据流水线

在本目录运行：

```powershell
python -m pip install -r requirements.txt
python build_customer_snapshots.py `
  data/raw/online_retail_II_2009-2010.csv `
  data/raw/online_retail_II_2010-2011.csv
python -m unittest discover -s tests -v
```

默认输出到 `data/processed/customer_snapshot_pipeline/`。清洗和快照口径见 [流水线说明](docs/customer_snapshot_pipeline.md)。现有产物记录 1,067,371 条原始交易行、776,579 条合格交易行，以及 18 个截止日期的 54,040 条客户快照。

## 原型代码状态

`src/train_pipeline.py` 现在读取上述 `customer_snapshots.csv`，沿用其中的训练、验证、测试划分，再调用分群与预测原型。运行命令为 `python src/train_pipeline.py`；完成训练后，可用 `streamlit run app/dashboard.py` 查看历史截止日期的评分。

本地已用真实快照运行该原型，生成 `data/processed/` 下的训练缓存与评分文件及 `models/` 下的模型文件。这些产物连同两份原始 CSV 一并纳入仓库；`clean_transactions.csv` 通过 Git LFS 保存，克隆仓库时需安装 Git LFS 才能取得完整文件。分群、正式六组实验、评估与仪表板仍需按任务清单逐项验证。现有代码比较的是 LR、GBDT、三个软投票权重及 Dummy 基线，尚未实现提案中的“两组特征 × 三类模型”；界面和导出会将其标为原型评分，不能当作最终报告或已校准复购概率。PPT 与中文 proposal 保留为 2026 年 9 月的计划版本，进度以本目录文档为准。
