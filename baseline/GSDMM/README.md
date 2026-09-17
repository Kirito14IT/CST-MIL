# GSDMM

## SCCD 统一实验入口（本项目适配）

`src/main/Benchmark.java` 为原 Java GSDMM 增加训练、统计量保存和新文档冻结后验推理。
`Model` 仍使用 collapsed Gibbs，固定种子 42，主题容量 32、100 次迭代、alpha=0.1、
beta=0.1；后验改为等价对数计算以避免长窗口下溢，完整保留重复词项的词频与 `+j` 项。
输入统一使用 UTF-8；空文本/全未知词输出零分布。便携 JDK 在
`cst_mil/benchmark/tools/`，版本及校验记录见其中 `jdk-provenance.json`。

Python 适配类位于 `cst_mil/src/cst_mil/bench/native_topics.py`。

```powershell
cd D:\github\intern\cst_mil
uv run --frozen --group benchmark cst-bench run --suite full677 --method gsdmm --limit all --resume
```

模型保存时包含原生统计量及实际编译类，恢复推理不会更新训练计数。
共同协议及逐方法命令见 [SCCD 操作指南](../../cst_mil/benchmark/操作指南.md)。

## Original implementation
The datasets are in format of JSON like follows:   
   {"text": "centrepoint winter white gala london", "cluster": 65}   
   {"text": "mourinho seek killer instinct", "cluster": 96}   
   {"text": "roundup golden globe won seduced johansson voice", "cluster": 72}   
   {"text": "travel disruption mount storm cold air sweep south florida", "cluster": 140}   
   {"text": "wes welker blame costly turnover", "cluster": 89}   
         	......   
	   
The output of GSDMM are D (the number of documents in the dataset) lines. Each line contains the estimated cluster for that document.
