# CST-MIL

## 新版：六方法 SCCD 比较与可续跑全量实验

新版入口为 `cst-bench`，包含 LDA、BTM、GSDMM、NMF、BERTopic 与 CST-MIL 的统一适配。
CST-MIL 增加可变数量主题、消息软分配、主题片段及原文证据；讨论主题由语料学习，
会话风险标签仍为 `CB / Non-CB`，并非预设三个风险类别。

本地 SCCD 共 **677 个完整会话**，不能按 3000 会话启动。小实验为 300 会话
（180/60/60），全量为 677 会话（406/135/136）。`--limit` 是累计推理会话数，
首次运行仍先拟合完整训练区；扩大 limit 会复用已保存模型并跳过完成会话。

使用 [新版操作指南](benchmark/操作指南.md) 中的六条独立命令，由你逐方法启动全量训练。
300 会话比较与六方法 5 会话验证完成后，查看实际
[比较报告](benchmark/comparison/pilot300/analysis-report.md)、
`benchmark/selection.json` 和 `benchmark/verification/`；结果尚未生成时不代表已通过验收。
所有新版实验使用 `uv run --frozen --group benchmark ...`，以保留锁定的基线依赖。

资源位置：JDK/原生工具在本目录下 `benchmark/tools/`；BERTopic 权重和嵌入缓存实际在
仓库根目录的 `../benchmark/models/paraphrase-multilingual-MiniLM-L12-v2/` 与
`../benchmark/cache/bertopic_embeddings/`。恢复实验时必须一并保留这两个仓库根目录资源，
不能只保留 `cst_mil/`。

## 旧版 `run_001` 实验记录（保留）

下面的复现命令、8 主题 MiniBatchNMF 描述和数值均属于原 `run_001`，不能作为新版六方法比较结果。

CST-MIL（Conversation Structure and Topic-aware Multiple-Instance Learning）是一个面向 SCCD 微博会话的 CPU 轻量级风险检测原型。它结合字符 n-gram、弱监督多实例学习、MiniBatchNMF 隐主题和回复结构，在不使用大语言模型、预训练语言模型、深度神经网络或 GPU 的条件下输出会话风险及证据评论。

## 研究边界

- 任务：SCCD 会话级 `CB / Non-CB` 二分类。
- 输入：微博原帖及按时间排序的评论；`to_id` 仅用于恢复回复结构。
- 训练监督：只使用会话标签。评论标签仅用于测试集证据定位评估。
- 不使用：用户身份、互动热度、细粒度评论标签、严重程度、转发数据。
- 本项目是一次 300 会话、固定种子 42 的可行性实验，不构成对五种基线的性能优越性证明。

## 数据许可提示

数据来自 [STAIR-BUPT/SCCD](https://github.com/STAIR-BUPT/SCCD) 的固定提交 `cf4015b802cabd651885211dc382152c1a270c31`。截至本实验记录时，该仓库没有清晰的独立 LICENSE 文件。因此，下载的数据只用于本地研究验证，不应随本项目重新分发；公开派生数据前应向原作者确认授权。

## 复现

```powershell
uv sync --frozen
uv run cst-mil download-data
uv run cst-mil prepare --seed 42 --sessions 300
uv run cst-mil train --config configs/pilot.yaml
uv run cst-mil evaluate --run artifacts/run_001
```

完整的一次性运行也可使用：

```powershell
uv run cst-mil run-pilot --config configs/pilot.yaml
```

正式产物写入 `artifacts/run_001/`。同一目录已存在完成标记时，`run-pilot` 默认拒绝覆盖，避免把第二次训练混入唯一正式运行。

## 方法概览

1. 字符 2–5 gram 哈希特征捕获短文本、错别字和变体表达。
2. 三轮自训练 MIL 从正会话中选择最高风险的 20% 评论实例。
3. 词级 TF-IDF 与 8 主题 MiniBatchNMF 提供隐主题和主题变化特征。
4. 会话级 Logistic Regression 汇总实例风险、主题与回复结构，输出风险概率。
5. Top-3 评论作为证据；主题高权重词提供可解释线索。

## 结果口径

只报告这一个固定划分上的描述性指标和对固定测试预测的分层 bootstrap 区间。没有重复训练、没有基线运行，也不执行显著性优越比较。

## 已完成的固定试验

`artifacts/run_001/COMPLETED.json` 记录正式训练次数为 1。60 个平衡测试会话的主要结果为：Accuracy 81.67%、macro-F1 81.62%、CB Recall 76.67%、ROC-AUC 90.11%、平均精确率 AP 90.43%。模型训练 4.646 秒，测试推理平均 12.05 毫秒/会话，模型文件约 3.58 MiB。

评论证据定位明显更弱：评论 AP 29.25%，Top-3 在含危险评论的会话中命中至少一条危险评论的比例为 56.82%。因此，当前方法只能作为会话筛查可行性原型，不能把 Top-3 候选当作可靠的单句定案证据。

详细指标、条件性 bootstrap 区间、图表、错误案例和统计边界见：

- `artifacts/run_001/metrics.json`
- `artifacts/run_001/analysis/analysis-report.md`
- `artifacts/run_001/analysis/stats-appendix.md`
- `artifacts/run_001/analysis/figure-catalog.md`
