# 商品推荐演示：客户12348

在PowerShell运行：

```powershell
cd "D:\桌面\AIS课件与workshop\pre2\main"
python -B -m recommendation.experiment
python -m streamlit run app/dashboard.py
```

已有完整产物时，第一个命令核对来源和输出，不重新调参。浏览器打开本地仪表板：

1. 进入 **Product recommendations**。
2. 推荐日期选择 **2011-09-01**，客户选择 **12348**。
3. 默认方案为 **个人重复购买**：查看10件商品和“过去180天买过”标记，下载CSV。
4. 在“查看推荐方案”切换 **ItemKNN、隐式反馈ALS、EASE**，比较同一客户的不同名单。
5. 展开比较表查看五种方法；切换日期、客户后仍各有10件商品。

这个例子的离线答案：该客户下一次合格购买发生于 **2011-09-25 13:13**，订单内有 **23077、23078** 两种商品。默认推荐分别将它们排在第 **1、3** 位，Recall@10为100%，NDCG@10约0.9197。这是核对用的单个例子；总体效果使用全部测试客户计算，不能用这个成功案例代表总体准确度。

命令行核对：

```powershell
Import-Csv .\recommendation\outputs\v1\test_customer_metrics.csv |
  Where-Object { $_.snapshot_date -eq '2011-09-01' -and $_.customer_id -eq '12348' -and $_.method -eq 'personal_repeat' } |
  Select-Object snapshot_date, customer_id, recommended_items, ndcg_10, recall_10, hit_10
```

未来订单和命中结果仅存离线评价文件，页面名单及下载不包含这些答案。全部模型、规则及不利结果见 [比较报告](outputs/v1/report.md)。
