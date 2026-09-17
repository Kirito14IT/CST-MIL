# 短文本隐主题发现基线代码与复现调研

## 新版 SCCD 可执行适配入口

五个 baseline 已通过 `../cst_mil/src/cst_mil/bench/` 接入共同的 SCCD 会话窗口、
训练/验证/测试划分、风险层和断点协议。LDA-C、BTM、GSDMM 使用本目录原生代码的适配版；
NMF 明确采用 scikit-learn 第三方实现；BERTopic 使用固定版本及本地多语言句向量模型。
JDK 和原生工具位于 `../cst_mil/benchmark/tools/`。BERTopic 权重实际位于仓库根目录的
`../benchmark/models/paraphrase-multilingual-MiniLM-L12-v2/`，嵌入缓存位于
`../benchmark/cache/bertopic_embeddings/`；这两个资源目录都需要保留，不能只复制 Python 项目目录。

运行方法、样本单位及六条独立全量命令见
[SCCD 操作指南](../cst_mil/benchmark/操作指南.md)。SCCD 只有 677 个完整会话；
300 会话比较与最多三次追加 CST-MIL 修订的实际结果生成后见
[新版比较报告](../cst_mil/benchmark/comparison/pilot300/analysis-report.md)。
报告和验证记录未生成时，不能将代码适配完成等同于实验完成。

以下保留原始代码收集与论文调研记录，其中“尚未安装/运行”的状态属于当时的历史阶段。

## 1. 本目录包含什么

本目录按“原作者优先”保存 LDA、BTM、GSDMM、BERTopic 的作者代码，并为未找到原作者官方仓库的 NMF 提供单独说明。

本阶段仅下载源码并固定版本，没有安装依赖、下载模型、下载数据集或运行训练。

| 方法 | 本地目录 | 代码身份 | 固定版本 | 许可证与注意事项 |
|---|---|---|---|---|
| LDA | [`LDA/`](./LDA/) | David Blei 团队的 LDA-C | `0c46575e89092683010db077f713f8aa3b3594a2` | [`license.txt`](./LDA/license.txt) 是 LGPL-2.1 文本，但 [`readme.txt`](./LDA/readme.txt) 和源码头写 GPL-2.0-or-later，许可证声明存在冲突 |
| BTM | [`BTM/`](./BTM/) | 第一作者 Xiaohui Yan 的实现 | `66cc9b475afec81f3e74bb393b874b3fe5d5a148` | Apache-2.0；C++、Python、Bash；作者说明仅在 Linux 测试，Windows 建议 Cygwin |
| GSDMM | [`GSDMM/`](./GSDMM/) | 论文作者 Jianhua Yin 的 Java 实现 | `1b344d9de28faee43ec0df0c3cf7ea57737fe14e` | 仓库没有 LICENSE；可公开查看和下载，但不能据此确认修改、再分发授权 |
| NMF | [`NMF/`](./NMF/) | 未找到 Lee–Seung 原作者维护的官方仓库 | 无 | 本目录只有[来源与替代实现说明](./NMF/README.md) |
| BERTopic | [`BERTopic/`](./BERTopic/) | Maarten Grootendorst 官方实现 | tag `v0.17.4`，commit `75f2910562f0e372beee29acbbd2a2835ba72cf2` | MIT；Python `>=3.10` |

> 注意：BERTopic 远端同时存在同名分支和标签 `v0.17.4`。本目录固定的是 **tag** 对应的提交 `75f2910`，不是同名分支的提交。

### GSDMM 的第三方 Python 实现

任务要求文档提到的 [`rwalk/gsdmm`](https://github.com/rwalk/gsdmm) 是 Ryan Walker 的第三方 Python 复现，不是论文作者代码，因此本次没有下载。它采用 MIT 许可证，运行方便，但当前实现要求文档传入去重后的 token 集合，忽略同一文档内重复词；这与作者 Java 实现和原论文对词频的处理并不完全等价。

## 2. 五种方法分别做什么

这些算法本身不绑定某个数据集；下表描述的是输入和核心机制。

| 方法 | 典型输入 | 核心机制 | 短文本上的主要特点 |
|---|---|---|---|
| LDA | 文档-词频计数矩阵 | 每篇文档由多个潜在主题混合生成 | 单条文本过短时，文档内部词共现不足 |
| BTM | 分词后的短文本集合 | 在整个语料层面建模词对 biterm | 聚合全局词对，专门缓解短文本稀疏性 |
| GSDMM | 每篇短文本的 token 与词频 | 每篇短文档只属于一个主题/簇，使用 Gibbs Sampling | 适合单主题短文本，可自动让空簇消失 |
| NMF | 非负 TF-IDF 矩阵 | 将文档-词矩阵分解成文档-主题和主题-词矩阵 | 快、直观，但结果受向量化与初始化影响明显 |
| BERTopic | 原始文本及其句向量 | Sentence-Transformers → UMAP → HDBSCAN → c-TF-IDF | 语义能力强，但嵌入和聚类成本更高，并会产生离群主题 `-1` |

BERTopic 的基础流程不依赖生成式大语言模型。GPU 可以加速文本嵌入，但不是必需条件；基础 c-TF-IDF 主题词来自原语料，不会天然“生成幻觉”。只有额外接入生成式模型为主题自动命名或总结时，才需要单独控制生成幻觉。

## 3. 原论文和代表性实验分别使用什么数据

### LDA

原论文：[Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)，JMLR 2003。

- C. elegans 科学摘要：5,225 篇，28,414 个不同词项，用于困惑度实验。
- TREC Associated Press：16,333 篇新闻、23,075 个词项，用于困惑度实验。
- Reuters-21578 子集：约 8,000 篇、15,818 个词项，用于分类实验。
- EachMovie：用于协同过滤，不是主题发现语料。
- 本地 LDA-C 的 [`example/ap.tgz`](./LDA/example/ap.tgz) 只包含 2,246 篇 AP 示例文档，不等于原论文的 16,333 篇实验子集。

### BTM

原论文：[A Biterm Topic Model for Short Texts](https://xiaohuiyan.github.io/paper/BTM-WWW13.pdf)，WWW 2013。

- Tweets2011：TREC 2011 Microblog 原始集合约 1,600 万条 tweet；论文清洗后使用 4,230,578 条。
- Baidu Zhidao Question：论文清洗后 189,080 条、35 类；本地仓库包含 [`data/baiduQA.txt`](./BTM/data/baiduQA.txt)。
- 20 Newsgroups：18,828 篇、20 类，用作正常长度文本验证。
- 仓库 [`sample-data/`](./BTM/sample-data/) 仅用于演示命令，作者明确说明它不是论文实验数据。

BTM 主要报告主题一致性、Jensen–Shannon 距离、分类准确率、Purity、NMI、ARI、耗时和内存。原论文没有用 held-out perplexity 横向评价 BTM。

### GSDMM

原论文：[A Dirichlet Multinomial Mixture Model-based Approach for Short Text Clustering](https://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf)，KDD 2014。

- Google News 2013-11-27 快照：11,109 条新闻、152 个事件簇。
- 基于同一快照构造 TitleSet、SnippetSet、TitleSnippetSet 三种长度版本；本地仓库对应 [`data/T`](./GSDMM/data/T)、[`data/S`](./GSDMM/data/S) 和 [`data/TS`](./GSDMM/data/TS)。
- TweetSet：从 TREC 2011/2012 Microblog 的高度相关 tweet 构造，2,472 条、89 个簇；本地为 [`data/Tweet`](./GSDMM/data/Tweet)。

论文使用 Homogeneity、Completeness、NMI、ARI、AMI，并报告 20 次运行的均值和标准差。作者仓库也包含 20 Newsgroups，但它不是原论文四组主实验数据之一。

### NMF

原论文及数据说明见 [`NMF/README.md`](./NMF/README.md)。原始工作使用 CBCL 人脸和 Grolier Encyclopedia；它们不适合作为当前中文短文本任务的统一公开基准。

### BERTopic

原论文：[BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794)，2022。

- 20 Newsgroups：论文使用 OCTIS 预处理后的 16,309 篇、20 类。
- BBC News：2,225 篇新闻。
- Trump tweets：44,253 条，覆盖 2009–2021 年。
- 动态主题实验另使用 2006–2015 年联合国大会一般性辩论发言。

论文主要使用 NPMI topic coherence、topic diversity 和运行时间；动态实验按时间片重复计算主题质量。

## 4. 有没有一个公开数据集可以统一评估

### 结论

可以选择公开语料让五种方法运行并进行统一比较，但目前没有发现一个数据集同时满足以下全部条件：

- 中文；
- 真实群聊或多轮聊天；
- 同时含黄、赌、毒、暴力、诈骗与正常内容；
- 具有会话级人工标签；
- 数据许可证清晰且允许公开复现。

因此应使用“通用算法验证 + 中文危险领域验证”两条轨道，不应让 LCCC-base 单独承担危险识别评测。

### 通用算法验证

| 数据集 | 适合做什么 | 局限 |
|---|---|---|
| [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) | 五种方法共同冒烟测试；有 20 个类别，可计算 NMI/ARI/Purity | 英文且多数文本较长，不是聊天；不同发布入口的许可说明需要单独核实 |
| [SearchSnippets / STTM](https://github.com/qiang2100/STTM/tree/master/dataset) | 12,295 条、8 类、平均约 14.4 个词，较贴近短文本主题模型 | STTM 仓库没有为该数据集提供清晰的独立 LICENSE，不宜重新打包分发 |
| [UCI News Aggregator](https://archive.ics.uci.edu/dataset/359/news%2Baggregator) | 422,937 条新闻标题，4 类并带事件 ID；CC BY 4.0，许可最清楚 | 英文新闻标题，不是聊天或危险内容 |

若只做一次最容易解释的算法冒烟测试，优先用 20 Newsgroups；若后续成果需要连数据一起公开，优先考虑 UCI News Aggregator 的分层子集。

### 中文危险领域验证

#### ChiFraud

- 代码与数据：[xuemingxxx/ChiFraud](https://github.com/xuemingxxx/ChiFraud)
- 论文：[CHIFRAUD: A Long-term Web Text Dataset for Chinese Fraud Detection](https://aclanthology.org/2025.coling-main.398/)
- 包含 59,106 条欺诈/非法推广文本和 352,328 条正常文本，覆盖赌博、色情交易、假证件、违禁药品等十类。
- 与“先聚焦黄赌毒”的方向最接近，但它是网页文本，不是群聊；仓库没有明确 LICENSE，因此可用于本地学术研究前的可行性测试，但公开再分发前需要联系作者确认权限。

#### ChineseHarm-Bench

- 官方仓库：[zjunlp/ChineseHarm-bench](https://github.com/zjunlp/ChineseHarm-bench)
- 数据集卡：[Hugging Face](https://huggingface.co/datasets/zjunlp/ChineseHarm-bench)
- 正式 `bench.json` 含 6,000 条，六类各 1,000 条：博彩、低俗色情、谩骂引战、欺诈、黑产广告、不违规。
- 许可证为 CC BY-NC 4.0，适合非商业研究；没有独立毒品类别，也不是多轮会话。

建议先在 ChiFraud 或 ChineseHarm-Bench 上验证“中文短文本危险主题”能力，但必须把结论限定为单条文本或网页文本，不能据此宣称已经解决真实群聊上下文识别。

## 5. LCCC-base 是什么

LCCC 是 **Large-scale Cleaned Chinese Conversation corpus**，由清华大学 CoAI 团队发布：

- 官方仓库：[thu-coai/CDial-GPT](https://github.com/thu-coai/CDial-GPT)
- 原论文：[A Large-Scale Chinese Short-Text Conversation Dataset](https://arxiv.org/abs/2008.03946)
- Hugging Face 加载方式：`load_dataset("lccc", "base")`

LCCC-base 的原始对话来自微博评论树，经过比 LCCC-large 更严格的规则和分类器清洗。官方统计约为：

- 682 万段对话；
- 约 2,007 万条 utterance；
- 单条 utterance 平均约 6.79–8.32 个分词词语；
- 每条记录本质上是一组按顺序排列的对话语句。

它不适合作为当前项目的危险内容主评测集，原因是：

1. 没有主题标签，也没有危险/正常标签，不能直接计算 NMI、ARI、Purity 或危险识别准确率。
2. 清洗过程主动过滤敏感词、脏话、表情和部分不当内容，危险正样本反而会被削弱。
3. 微博评论路径不等于真实即时通讯群聊；公开字段没有完整群成员、稳定账号、回复关系和群角色结构。
4. 将 LCCC 当正常样本、把另一个平台的数据当危险样本，会让模型学到平台和文体差异，造成虚假的高分。

合理用途包括：测试中文对话读取和分词流程、进行无监督主题展示、提供开放域对话背景语料。若用于正常对照，危险和正常数据还应来自同平台、同时间范围并经过同样清洗。

## 6. 真正复现前先做什么

### 第一步：统一任务定义

当前材料存在范围差异：[`任务要求.docx`](../任务要求.docx) 将危险主题定义为“暴力、诈骗”，而 [`待办事项.md`](../待办事项.md) 又提出先聚焦“黄赌毒”。正式实验前应先固定一套主标签，例如：

```text
normal / gambling / pornography / drugs
```

或者：

```text
normal / violence / telecom_fraud
```

同一张主结果表不能在实验中途更换标签定义。

### 第二步：固定文档单位

需要预先决定一个样本是：

- 一整段会话；或
- 固定时间窗口；或
- 固定消息条数的滑动窗口。

五种方法必须使用同一批样本边界。不能让 LDA/NMF 输入整段会话，却让 BTM/GSDMM 输入单条消息。

### 第三步：建立统一数据清单

至少保存：`doc_id`、原始语句序列、拼接后的文档、分词结果、标签、数据划分、数据来源。真实聊天数据还应先脱敏并确认采集与使用授权。

### 第四步：统一预处理但保留方法所需输入

- 使用固定版本的中文分词器、停用词表、同义/暗语词典和最小词频规则。
- LDA 输入词频计数；NMF 输入 TF-IDF；BTM/GSDMM 输入相同的 token 序列；BERTopic 输入相同的原始文档，并固定中文或多语言句向量模型。
- 预处理产物一次生成并保存，避免每个方法各自清洗导致比较失真。

### 第五步：先小规模验证，再跑完整实验

1. 先取 500–2,000 条公开数据验证数据读取、训练、推理和结果格式。
2. 有标签实验将主比较的主题数 `K` 固定为真值类别数；GSDMM/BERTopic 自动发现的簇数另行报告。
3. 每个方法至少运行 5 个随机种子，报告均值和标准差。
4. 统一报告 NPMI 或 `c_v`、topic diversity、NMI、ARI、Purity、运行时间和峰值内存。
5. BERTopic额外报告 `-1` 离群样本比例和有效覆盖率。
6. 不用 perplexity 横向比较全部五种方法，因为 NMF、BTM 和 BERTopic 没有与 LDA 完全一致的文档似然定义。

## 7. 环境准备建议

原作者代码跨越 C、C++、Java 和现代 Python，建议使用 WSL2/Ubuntu 管理传统代码：

| 方法 | 最小环境方向 | 说明 |
|---|---|---|
| LDA | `gcc`、`make`、标准数学库 | 根目录运行 `make`；输入是稀疏词频格式 |
| BTM | `g++`、`make`、`bash`、`bc`、`wc`、Python 3 | 作者只在 Linux 测试；Windows README 建议 Cygwin |
| GSDMM | JDK、Eclipse 或命令行 `javac/java` | 仓库带 `org.json` jar；论文参数与代码默认参数需要重新核对 |
| NMF | 后续建议 Python/scikit-learn | 当前未下载第三方实现 |
| BERTopic | Python `>=3.10`、sentence-transformers、UMAP、HDBSCAN、scikit-learn | CPU 可运行；中文需要 jieba 等分词器与中文/多语言嵌入模型 |

下一阶段应先分别完成四个作者仓库的最小示例运行，再开发统一数据适配与评测脚本；不要直接在完整中文数据上同时排查环境、预处理和算法问题。
