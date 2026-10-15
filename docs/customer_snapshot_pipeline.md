# 客户快照数据流水线

该流水线保留两份原始 CSV 不变，生成可追溯的数据清洗结果和月度客户快照。

清洗规则实现位于 `src/data_cleaning.py`，客户特征、标签和时间划分实现位于 `src/features.py`；`build_customer_snapshots.py` 负责调用并写出产物。后续训练直接读取 `customer_snapshots.csv`，不再重复清洗原始数据。

## 清洗规则

- 完全重复行仅保留第一次出现的记录。
- 缺失 Customer ID 的记录不进入客户级建模。
- 取消订单、数量或价格非正的记录不作为合格购买。
- 邮费、手续费、折扣、人工调整、测试、样品和礼品券等非商品编码不进入购买特征。
- 每条被排除记录保留来源文件、原始行号和一个主要排除原因。

## 快照规则

- 每月 1 日生成快照。
- 特征窗口为 `[t-180天, t)`，严格不使用截止时点及之后的数据。
- 客户必须在特征窗口内至少有一次合格购买。
- 标签窗口为 `[t, t+30天)`；窗口内至少有一次合格购买时，`repurchase_30d=1`。
- 训练、验证和测试按时间顺序划分。跨越下一分区起点的标签窗口会被标记为 purge，不用于建模。

## 运行

```powershell
python build_customer_snapshots.py `
  data/raw/online_retail_II_2009-2010.csv `
  data/raw/online_retail_II_2010-2011.csv `
  --output-dir data/processed/customer_snapshot_pipeline
```

主要输出：

- `clean_transactions.csv`：合格交易明细。
- `excluded_transactions.csv`：被排除记录及原因。
- `cleaning_audit.csv`：各规则命中数与主要排除数。
- `cutoff_schedule.csv`：快照、特征、标签和数据分区边界。
- `customer_snapshots.csv`：客户特征及未来 30 天复购标签。
- `validation_report.csv`：行数对账、快照唯一性和防泄漏检查。
- `run_manifest.json`：源文件哈希、配置、日期范围和输出规模。

## 测试

```powershell
python -m unittest discover -s tests -v
```
