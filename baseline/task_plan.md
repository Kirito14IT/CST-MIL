# Task Plan: 原作者基线代码下载与数据集复现调研

## Goal
将四个可核实的原作者基线仓库固定到指定版本，补充 NMF 无官方代码说明，并交付可核验的数据集与复现指南。

## Phases
- [x] Phase 1: 检查工作区、来源与下载前提
- [x] Phase 2: 下载并固定 LDA、BTM、GSDMM、BERTopic 源码
- [x] Phase 3: 编写 NMF 说明与 baseline 总说明
- [x] Phase 4: 验证来源、版本、许可证、目录与文档

## Key Questions
1. 四个作者仓库能否下载并准确固定到计划版本？
2. 每个方法的官方身份、许可证和运行限制是否被准确记录？
3. 数据集建议是否区分原论文数据、通用基准和项目领域数据？

## Decisions Made
- 代码来源采用用户确认的“原作者优先”策略。
- GSDMM 下载 jackyin12/GSDMM；rwalk/gsdmm 仅作为第三方替代列入说明。
- NMF 不下载第三方代码，只创建原论文代码缺失说明。
- 本阶段不安装依赖、不下载数据集或模型、不运行训练。

## Errors Encountered
- BTM 前两次 `git clone` 直连 GitHub 443 超时；改为仅对当前 Git 命令使用本机 `127.0.0.1:7897` 代理后成功，不修改全局配置。
- 第一次仓库汇总核验使用 `foreach (...) { ... } | Format-List` 触发 PowerShell 空管道解析错误；改用数组收集结果后再格式化。
- LDA 首次直连 clone 只输出开始信息但未形成目标目录；改用已验证的单命令临时代理重新下载并再次核对 HEAD。
- BERTopic 同时存在同名分支和标签 `v0.17.4`；`git clone --branch` 取到了分支提交 `a578f82`，已显式抓取 `refs/tags/v0.17.4` 并切换到计划要求的标签提交 `75f2910`。
- 最终核验脚本中的 `"$name: ..."` 触发 PowerShell 变量名后冒号解析错误；改用 `-f` 格式化字符串重跑。

## Status
**Complete** - 四个仓库、NMF 说明、总说明及全部核验均已完成。
