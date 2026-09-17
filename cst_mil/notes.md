# Notes: CST-MIL 与 SCCD

## Evidence already established

- `任务进度报告.md` 当前为空文件；现有 `baseline/` 是固定版本源码归档，没有 SCCD 训练结果。
- SCCD 原始发布包含 677 个会话：354 个 CB、323 个 Non-CB；`comments.csv` 包含 38,999 条评论。
- 原始评论存在孤立 `post_id`、重复 `comment_id` 与一组冲突标签，读取器必须显式隔离并记录。
- 固定划分在 `post_id` 层面完全互斥，但 raw user/text 审计发现 train/test 有 7 个原帖作者、93 个评论用户重叠，并有 35 种长度至少 5 的完全相同评论文本跨集合；首轮不使用 user ID，仍需把风格与近重复风险列为限制。
- SCCD 是微博原帖—评论树，不是微信/QQ 群聊；研究结论限于网络欺凌、辱骂和言语暴力。
- SCCD 官方仓库未提供清晰 LICENSE，因此本地实验可以记录来源，但不重新分发数据。

## Reporting boundaries

- LCCC-base 对普通中文对话处理有价值，但不适合作为危险会话评测集。
- 五种原始基线不能直接输出会话级风险标签；本轮只提出后续统一适配方案，不声称已经通过 SCCD 实验证明它们性能较差。
- “高稀疏性”可用消息长度统计支持；“强噪声性”仅部分满足；“快速演化性”在静态 SCCD 上不能被实证。字符哈希与 SGD 估计器具备后续 `partial_fit` 的技术基础，但本轮没有实现或验证完整在线更新流水线。

## Protected baseline snapshot

- `任务要求.docx`: SHA256 `5D5E57482D5F7F90F552F543F5FD4B12B35589417D4D814D32AADBEE9E7273FA`, 22,507 bytes.
- `任务要求 - 副本.docx`: SHA256 `0118C5866B613D01B89D14BB577181CE5C6EB82B7D24FBF9E07FC6A2D2A0AD3B`, 23,453 bytes.
- `baseline/`: 324 files, deterministic tree SHA256 `4B4B3875F0B4876F66325CB12CDC87EF445C3AA7EBD0044ED2826B707773AF1A`.
- `任务进度报告.md`: implementation start size 0 bytes.

## Sources

- SCCD repository: https://github.com/STAIR-BUPT/SCCD
- SCCD paper: https://aclanthology.org/2025.coling-main.639/
- LCCC/CDial-GPT repository: https://github.com/thu-coai/CDial-GPT
- LCCC paper: https://arxiv.org/abs/2008.03946

## Formal run_001 outcome

- Exactly one formal fit completed; test sessions: 60 (30 CB / 30 Non-CB).
- Accuracy 0.8167, macro-F1 0.8162, CB recall 0.7667, ROC-AUC 0.9011, AP 0.9043.
- Confusion matrix: TN 26, FP 4, FN 7, TP 23.
- Conditional test bootstrap: macro-F1 [0.7147, 0.9161], CB recall [0.6000, 0.9000], AP [0.8313, 0.9690].
- Evidence localization remains weak: comment AP 0.2925, Hit@3 0.5682, mean Recall@3 0.0512.
- Training 4.646 s; inference 12.05 ms/session; model 3.58 MiB.
