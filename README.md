# 电商客户分析

基于 UCI Online Retail II 购买记录，完成客户分群、未来30天复购评分、消费潜力排序，以及购买习惯画像与商品推荐。用户上传购买明细、选择分析日期，系统生成四项分析，支持筛选和导出名单。

先阅读[项目学习指南](docs/项目学习指南.md)，也可下载[HTML阅读版](docs/项目学习指南.html)在浏览器打开。字段和操作见[上传数据使用说明](docs/上传数据使用说明.md)。

## 当前使用的方案

| 功能 | 当前方案 | 页面提供什么 |
| --- | --- | --- |
| 客户分群 | K-means，4群 | 群体分布、购买特点、群内客户名单。 |
| 未来30天复购 | 14项历史特征＋XGBoost | 复购倾向评分、客户优先顺序。 |
| 高消费潜力 | 20项历史特征＋两步LightGBM | 消费潜力排名、前20%关注名单。 |
| 画像与商品推荐 | D完整组34项特征＋LightGBM分类 | 购买习惯画像、每位客户10件推荐商品。 |

商品推荐默认采用 **D完整组＋LightGBM分类**：综合商品购买记录、客户画像与EASE商品关联，为候选商品逐件打分，再选出Top10。个人重复购买保留为离线实验对照，各方法比较见学习指南和对应报告。

## 快速启动

已在 Python 3.11 环境验证。保存的模型依赖训练时的库版本，请使用 `requirements.txt`。数据和较大产物通过 Git LFS 保存，克隆后需要取得实际文件内容。

```powershell
git clone https://github.com/9zhuandachang/ecommerce-customer-segmentation.git
Set-Location ecommerce-customer-segmentation
git lfs install
git lfs pull
python -m pip install -r requirements.txt
python -m streamlit run app/dashboard.py
```

服务启动后，浏览器打开 [http://localhost:8501](http://localhost:8501)。已在项目根目录时，只需执行上面最后一条启动命令。

1. 在侧栏“数据导入”上传 `data/demo/2011-12-01/purchase_history.csv`。
2. 分析日期选择 **2011-12-01**，点击“生成分析”。
3. 在四个分析页面查看、筛选并导出结果。

系统支持 **2011-09-01至2011-12-01之间任意一天，包含首尾**。演示明细包含759,615条真实合格交易，最晚为2011-11-30；选择12月1日会生成3,444位客户的快照，前20%消费潜力客户为689位。系统按照分析日期提取此前的购买记录，调用保存好的模型。

客户总数在标题旁展示，匹配和显示数量位于表格上方；消费中位数位于群体对比区。推荐页的客户列表可按近180天购买次数降序排列；从10项购买习惯指标中，将有实际数值的内容展示在摘要卡和“画像详情”中。

更换明细或日期时，返回“数据导入”重新生成分析。新分析成功前会保留原结果。已有旧规则分析的会话会提示重新上传并生成，以应用新的推荐模型。代码更新后若出现旧模块导入错误，在运行服务的终端按 `Ctrl+C`，再执行启动命令；样式更新可按 `Ctrl+F5` 刷新页面。

## 项目目录

| 位置 | 用途 |
| --- | --- |
| `app/dashboard.py` | 唯一业务页面入口。 |
| `app/upload_inference.py` | 校验上传明细、生成客户特征、调用四项分析。 |
| `app/business_views.py`、`app/dashboard.css` | 业务表格、导出字段和页面样式。 |
| `app/*_data.py` | 当前及历史实验结果的读取与核对适配器。 |
| `data/raw/` | 两份原始年度CSV。 |
| `data/processed/customer_snapshot_pipeline/` | 合格交易、排除记录、审计、54,040条客户月快照及时间划分。 |
| `data/demo/2011-12-01/` | 真实上传示例、来源记录及演示压缩包。 |
| `src/`、`build_customer_snapshots.py` | 共用清洗、特征与基础模型构建代码，数据流水线入口。 |
| `stage4/`、`stage5/`、`stage6/` | 分群对比、复购对比及模型集成。 |
| `value_prediction/` | 消费潜力排序与金额相关实验。 |
| `recommendation/` | 推荐算法、购买画像、商品级打分比较。 |
| `tests/` | 数据窗口、模型复现、名单与页面回归检查。 |
| `docs/` | 学习指南、业务口径、任务流程和数据说明。 |

## 数据与模型重建入口

仓库已包含可运行的结果，查看页面可直接使用。需要重做实验时，在项目根目录按顺序执行相关入口。

```powershell
# 清洗、客户月快照和审计
python -B build_customer_snapshots.py data/raw/online_retail_II_2009-2010.csv data/raw/online_retail_II_2010-2011.csv

# 三种分群算法、2—6群比较，选择K-means四群
python -B -m stage4.compare_segmentation compare
python -B -m stage4.compare_segmentation select --algorithm kmeans --k 4

# 四类复购模型、三组特征比较，然后集成分群与复购模型
python -B -m stage5.compare_repurchase
python -B -m stage6.integrate_reanalysis

# 消费额标签、20项历史特征、消费潜力排序比较
python -B -m value_prediction.labels
python -B -m value_prediction.temporal_features
python -B -m value_prediction.ranking_comparison

# 推荐算法、画像混合和商品级打分比较
python -B -m recommendation.experiment
python -B -m recommendation.profile_experiment
python -B -m recommendation.item_experiment

# 重新生成真实上传示例
python -B -m app.prepare_upload_demo
```

各实验保存了来源哈希和冻结配置。部分入口发现已有结果时会进行完整性核对；参数及时间窗口见对应方案文件。

## 结果与比较报告

| 内容 | 报告或模型位置 |
| --- | --- |
| 分群算法与群数比较 | [分群方案](stage4/segmentation_comparison_protocol.md)、`stage4/outputs/segmentation_comparison/` |
| 复购模型对比 | [复购方案](stage5/repurchase_comparison_protocol.md)、`stage5/outputs/repurchase_comparison/` |
| 当前分群与复购模型集成 | [统一回放报告](stage6/outputs/reanalysis_v2/report.md)、`stage6/outputs/reanalysis_v2/model/integrated_bundle.joblib` |
| 消费潜力排序对比 | [排序报告](value_prediction/outputs/ranking_comparison_v1/report.md)、`value_prediction/outputs/ranking_comparison_v1/ranking_bundle.joblib` |
| 商品推荐算法对比 | [推荐报告](recommendation/outputs/v1/report.md)、`recommendation/outputs/v1/selected_model.joblib` |
| 画像混合与商品级模型对比 | [画像比较](recommendation/outputs/profile_v2/report.md)、[商品级比较](recommendation/outputs/item_v3_1/report.md) |

训练快照为2010年6月—2011年5月，验证为2011年6—8月，测试期复核为2011年9—11月。按所选方案计算，复购AP为0.6201、前20%名单精确率66.22%；消费潜力名单月均消费覆盖率70.23%；默认D完整组推荐NDCG@10为0.3001、Recall@10为17.79%、Hit@10为76.93%。具体统计单位、计算方法和月份口径见学习指南。

## 最终报告

| 部分 | 中文版 | English |
| --- | --- | --- |
| Part 1 使用的工具与技术 | [下载文档](docs/final_report/Part1_工具与技术_中文版.docx) | [Tools and Techniques](docs/final_report/Part1_Tools_and_Techniques_EN.docx) |
| Part 2 系统设计与模型 | [下载文档](docs/final_report/Part2_系统设计与模型_中文版.docx) | [System Design and Models](docs/final_report/Part2_System_Design_and_Models_EN.docx) |
| Part 3 系统效果与评价 | [下载文档](docs/final_report/Part3_系统效果与评价_中文版.docx) | [System Performance](docs/final_report/Part3_System_Performance_EN.docx) |

流程图：[中文版](docs/images/project_workflow_v3.png) · [English](docs/images/project_workflow_en.png)。可编辑SVG与图片保存在同一目录。

## 验证与仓库整理

```powershell
python -B -m unittest discover -s tests -v
```

已清理早期独立原型入口及其模型和缓存、退出导航的旧页面、重复演示快照、旧流程图和空占位文件。当前数据、模型、结果报告及其校验记录保留。

旧三群、早期复购、金额预测和推荐迭代仍保留在对应模块及 `outputs/` 中；它们用于实验对照、复现和现有测试。`src/modelling.py` 仍被这些实验引用。各阶段共用的数据处理代码继续位于 `src/data_cleaning.py` 与 `src/features.py`。
