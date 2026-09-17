# NMF 原作者代码说明

## SCCD 当前可执行适配

本项目已增加 `cst_mil/src/cst_mil/bench/native_topics.py::NMFTopicModel`，使用
scikit-learn 第三方 NMF 实现，不将其标为原论文作者源码。默认 TF-IDF 输入、主题容量 32、
`init="nndsvda"`、`solver="mu"`、`max_iter=300`、`tol=1e-4`、种子 42。
输出先用主题因子的 L1 尺度校正 NMF 权重，再按行归一化；空文本/全未知词为零分布。
小到不能初始化 32 个因子的工程烟雾数据显式记录有效因子数，并补零保持输出宽度一致。

```powershell
cd D:\github\intern\cst_mil
uv run --frozen --group benchmark cst-bench run --suite full677 --method nmf --limit all --resume
```

依赖版本由 Python 项目的锁文件固定，实际 sklearn 版本写入每次运行元数据。
完整协议见 [SCCD 操作指南](../../cst_mil/benchmark/操作指南.md)。

## 原作者代码存档范围

本目录没有放置可执行源码，因为在论文页面和作者公开主页中，未找到可以核实为 Daniel D. Lee、H. Sebastian Seung 原作者维护的 NMF 官方代码仓库。

这不等于“NMF 没有开源实现”。NMF 已有很多成熟实现，例如 scikit-learn 的 [`sklearn.decomposition.NMF`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)。本项目当前采用“原作者优先”策略，因此暂不把第三方实现下载到这里，也不把它们标为原论文代码。

## 原始论文

1. Daniel D. Lee, H. Sebastian Seung. [Learning the parts of objects by non-negative matrix factorization](https://www.nature.com/articles/44565). *Nature*, 1999.
2. Daniel D. Lee, H. Sebastian Seung. [Algorithms for Non-negative Matrix Factorization](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html). *NeurIPS*, 2000.

## 原论文数据

- CBCL 人脸数据：2,429 张 `19 × 19` 灰度人脸图像，实验分解得到 49 个基向量。这不是文本主题发现数据。
- Grolier Encyclopedia：30,991 篇百科文章；去除 430 个常见词后保留 15,276 个词项，实验分解为 200 个语义特征。

Grolier Encyclopedia 属于商业百科语料，未找到与原论文实验完全一致、许可证清晰且可自由重新分发的数据版本，因此不适合作为当前项目的公开复现基准。

## 后续可选实现

若后续目标从“保存原作者代码”转为“统一 Python 复现”，建议使用固定版本的 scikit-learn，并显式设置：

```python
from sklearn.decomposition import NMF

model = NMF(
    n_components=K,
    init="nndsvda",
    solver="mu",
    beta_loss="frobenius",
    random_state=SEED,
)
```

其中 `solver="mu"` 比默认坐标下降求解器更接近 Lee–Seung 论文的乘法更新思路。文本主题发现时，输入通常使用 TF-IDF 非负矩阵。
