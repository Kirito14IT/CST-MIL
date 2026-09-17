# Notes: 基线来源与复现调研

## 计划固定的源码

- LDA: https://github.com/blei-lab/lda-c @ `0c46575e89092683010db077f713f8aa3b3594a2`
- BTM: https://github.com/xiaohuiyan/BTM @ `66cc9b475afec81f3e74bb393b874b3fe5d5a148`
- GSDMM: https://github.com/jackyin12/GSDMM @ `1b344d9de28faee43ec0df0c3cf7ea57737fe14e`
- BERTopic: https://github.com/MaartenGr/BERTopic tag `v0.17.4` @ `75f2910562f0e372beee29acbbd2a2835ba72cf2`

## 下载结果

- LDA、BTM、GSDMM 均处于 detached HEAD，并与计划提交一致。
- BERTopic 远端同时存在同名分支和标签 `v0.17.4`；已显式切到标签提交 `75f2910`，不是同名分支提交 `a578f82`。
- 下载时仅对命令临时使用本机代理 `127.0.0.1:7897`，没有更改全局 Git 配置。

## 重要边界

- 原论文数据与仓库示例数据必须分开说明。
- “公开可访问”不自动等于“具有明确开放许可证”。
- LCCC-base 不含危险主题标签，不能单独用于危险识别定量评估。
- 本次交付不包含依赖、模型权重和数据集。

## 本地源码核验要点

- LDA 的 `license.txt` 是 LGPL-2.1 文本，但 `readme.txt` 与 Makefile 源码头声明 GPL-2.0-or-later，必须保留冲突说明。
- BTM 自带 Apache-2.0 许可证，README 明确仅在 Linux 测试并建议 Windows 使用 Cygwin。
- GSDMM 根目录不存在 LICENSE；只能称作者公开实现，不能声称许可证明确。
- BERTopic tag `v0.17.4` 的 `pyproject.toml` 要求 Python `>=3.10`，基础依赖包括 sentence-transformers、UMAP、HDBSCAN 与 scikit-learn。
- NMF 原作者论文有公开页面，但未找到可核实的作者维护代码仓库；后续 Python 复现可选 scikit-learn `solver="mu"`。

## 最终验证

- 四个仓库的 origin、HEAD 与计划一致，`git status --porcelain` 均为空。
- BTM 与 BERTopic 许可证文件存在；GSDMM 根目录无 LICENSE/COPYING；LDA 的 LGPL/GPL 表述冲突已由本地文本确认。
- `README.md`、`NMF/README.md`、`notes.md`、`task_plan.md` 均通过 UTF-8 读回检查。
- README 中 18 个本地链接全部存在，17 个外部链接均返回 HTTP 200。
- 全局 Git 代理与 `HTTP_PROXY` / `HTTPS_PROXY` 环境变量仍为空；下载使用的代理仅作用于单次 Git 命令。
