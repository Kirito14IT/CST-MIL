# CST-MIL Python package

[简体中文首页](../README.md) · [English](../README.en.md)

本目录包含最新语义方案、便携部署适配、传统主题基线适配器与历史研究入口。运行演示、准备模型及安装环境请使用根目录README；发布模型采用五来源联合监督，不能沿用早期弱监督CST-MIL的实验成绩。

安装后从仓库根目录查看入口：

```powershell
python -m cst_mil.deployment --help
python -m cst_mil.research_followup.full_joint_comparison --help
python -m cst_mil.research_followup.semantic_patent_followup --help
cst-bench --help
```

历史研究入口需要原始数据、来源审计和冻结划分等前置文件；这些不随代码重新分发。`--help`不等同于完整研究数据已准备，详见根目录的“研究代码与复现边界”。
