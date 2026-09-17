## Latent Dirichlet allocation

## SCCD 统一实验入口（本项目适配）

原始 C 变分 EM 算法通过 `cst_mil/src/cst_mil/bench/native_topics.py` 的
`NativeTopicModel("lda", config, work_dir)` 接入 SCCD。适配内容包括 Windows 编译、
显式随机种子、独立推理文件、长路径缓冲和冻结模型推理；默认主题容量 32、种子 42、
EM 上限 100、文档变分迭代上限 20，训练期估计 alpha。文档迭代预算固定，
不沿用旧代码在 EM 下降时无限倍增变分预算的行为。

从项目 Python 目录单独启动或续跑：

```powershell
cd D:\github\intern\cst_mil
uv run --frozen --group benchmark cst-bench run --suite full677 --method lda --limit all --resume
```

样本单位为完整会话；词表和主题参数只在训练区拟合。空文本/全未知词的主题分布为零。
运行数据、划分、版本与结果说明见 [SCCD 操作指南](../../cst_mil/benchmark/操作指南.md)。

## Original implementation

This is a C implementation of variational EM for latent Dirichlet allocation (LDA), a topic model for text or other discrete data. LDA allows you to analyze of corpus, and extract the topics that combined to form its documents. For example, click [here](https://github.com/Blei-Lab/lda-c/blob/master/example/ap-topics.pdf) to see the topics estimated from a small corpus of Associated Press documents. LDA is fully described in [Blei et al. (2003)](http://www.cs.columbia.edu/~blei/papers/BleiNgJordan2003.pdf).

This code contains:

* an implementation of variational inference for the per-document topic proportions and per-word topic assignments
* a variational EM procedure for estimating the topics and exchangeable Dirichlet hyperparameter

## Readme

View the [readme.txt](https://github.com/Blei-Lab/lda-c/blob/master/readme.txt) and fork or clone the repository.

## Sample data

2246 documents from the Associated Press **[download](https://github.com/Blei-Lab/lda-c/blob/master/example/ap.tgz)**.

Top 20 words from 100 topics estimated from the AP corpus **[pdf](https://github.com/Blei-Lab/lda-c/blob/master/example/ap-topics.pdf)**.

## Bug fixes and updates

To learn about bug-fixes, updates, and discuss LDA and related techniques, please join the topic-models mailing list, topic-models [at] lists.cs.princeton.edu.

To join, click [here](https://lists.cs.princeton.edu/mailman/listinfo/topic-models).

## Other implementations on the web

There are several other implementations of LDA on the web:
* [R package
](http://cran.r-project.org/web/packages/lda/)
* [The Mallet Toolkit from UMass](http://mallet.cs.umass.edu/)
* [Gregor Heinrich's LDA-J](http://www.arbylon.net/projects/)
* [Multinomial PCA](http://cosco.hiit.fi/search/MPCA/)
