"""Generate disclosure and tidy metric tables directly from sealed results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def build_report(output: Path) -> Path:
    def read(name: str):
        return json.loads((output / name).read_text(encoding="utf-8"))

    manifest = read("prepared/manifest.json")
    development = read("metrics.json")
    testing = read("metrics_test.json")
    summary = read("training_summary.json")
    gate = read("performance_gate.json")
    frozen = read("freeze.json")
    selection = read("qa/selection.json")
    rows = []
    for split, metrics in [("dev", development), ("test", testing)]:
        for source, category_metrics in metrics["messages"][split].items():
            if source == "source_macro":
                continue
            for category, value in category_metrics.items():
                if not value["n_observed"]:
                    continue
                rows.append(
                    {
                        "unit": "message",
                        "split": split,
                        "source": source,
                        "category": category,
                        "n": value["n_observed"],
                        "annotation_coverage": value["annotation_coverage"],
                        "AP": value["AP"],
                        "AUC": value["roc_auc"],
                        "macro_F1": value["macro_f1"],
                        "precision": value["precision"],
                        "recall": value["recall"],
                        "threshold": value["threshold"],
                    }
                )
        value = metrics["conversations"][split]
        rows.append(
            {
                "unit": "conversation",
                "split": split,
                "source": "sccd",
                "category": "CB",
                "n": value["n_observed"],
                "annotation_coverage": 1,
                "AP": value["AP"],
                "AUC": value["roc_auc"],
                "macro_F1": value["macro_f1"],
                "precision": value["precision"],
                "recall": value["recall"],
                "threshold": value["threshold"],
            }
        )
    with (output / "metrics_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# CST 联合监督语义版 v1.1：训练、核验与部署说明",
        "",
        "日期：2026-10-05（Asia/Shanghai）。这是独立的新监督与语义变体，"
        "旧 CST-r0 与各基线实验均保留。v1.1修正v1分块编码遗漏原生起止标记的问题；"
        "旧v1已标无效但证据完整保留。",
        (
            "部署性能门槛通过；默认模型的切换仍须以最终系统验收记录为准。"
            if gate["passed"]
            else "**当前为未通过部署门槛的候选版本：默认演示系统仍使用旧CST-MIL full677 r0。** "
            "候选在独立测试环境接入，不自动改正式部署清单，不用测试成绩绕过验证门槛。"
        ),
        "",
        "## 模型与来源",
        "",
        "演示系统只加载一个选定模型包。包内共享冻结多语言编码器，含三个风险输出层、SCCD会话聚合层及独立动态主题解释器。编码器没有微调；本轮没有付费API调用。",
        "",
        "| 来源 | 训练记录 | 验证记录 | 测试记录 | 可监督类别 |",
        "|---|---:|---:|---:|---|",
    ]
    labels = {
        "sccd": "攻击／冒犯（另有会话CB标签）",
        "conda": "来源定义的攻击／毒性",
        "toxicn": "攻击／冒犯、仇恨／歧视（转述掩码未知）",
        "scamspot": "垃圾／诈骗合并",
        "hatebr": "冒犯",
    }
    for source, counts in manifest["counts"].items():
        lines.append(
            f"| {source} | {counts['train']['records']} | {counts['dev']['records']} | "
            f"{counts['test']['records']} | {labels[source]} |"
        )
    lines += [
        "",
        "SCCD会话划分406／135／136。训练侧剔除了1,224条跨分区重复评论（SCCD1,222、ScamSpot2），284个SCCD训练会话的评论集合因此缩小，会话ID均保留。",
        "67个训练冲突组件按类别置未知；103个验证／测试重复组件保留并报告评估依赖。"
        "ToxiCN reporting不自动改负类；其训练55／验证12／测试13条在攻击与仇恨任务上未知。",
        "任何来源的正常标签仅用于其实际标注类别，不作为全部风险类别的阴性。仇恨任务的已标注评估来自中文ToxiCN；金融垃圾／诈骗以ScamSpot语言分布为准，不能称为所有语言均验证。",
        "",
        "## 实测结果",
        "",
        "| 单位 | 分区 | 来源 | 类别 | 已标注n | AP | AUC | macro-F1 | 精确率 | 召回率 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["unit"],
                    row["split"],
                    row["source"],
                    row["category"],
                    str(row["n"]),
                    *[
                        f"{row[name]:.4f}" if row[name] is not None else "未定义"
                        for name in ("AP", "AUC", "macro_F1", "precision", "recall")
                    ],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "主表只统计对应observed_mask为真的标签；聚合pooled以记录数为权，source_macro指标另见原始metrics文件，不用大来源掩盖其他来源成绩。",
        f"验证部署性能门槛：{'通过' if gate['passed'] else '未通过'}。"
        "阈值仅由验证macro-F1选择；修订版测试在本版配置冻结后一次评估，不按测试成绩换配置。原错误v1已打开过同一测试集；本次为独立于成绩的编码协议纠错复算，不能称首次未触碰测试。",
        "AP反映完整排序，改变分类阈值不能提高AP。来源macro平均AP作为补充，"
        "不能在看到结果后用它替代预设pooled门槛；单条消息与会话指标也不能互相替代。",
        "",
        "## 主题与分数口径",
        "",
        f"动态讨论主题使用逐消息语义聚类，距离阈值为{frozen['topic_distance']}；独立于风险分类特征。名称从完整原文词组抽取，keyword_supports保存消息ID、字符位置和摘录。",
        "模型辅助双评审覆盖30个真实开发会话与12个编写挑战，属于开发质检，没有人工讨论主题金标准，不报告为真实主题准确率。",
        f"距离选择只使用30个真实开发会话的双评审三维均值：{selection['mean_real_dev_ratings']}；"
        "12个编写挑战用于功能核验，不计入距离选择。字面支持不能证明概括性与分组正确。",
        "词组提取沿用旧候选开发阶段的Unicode泛词、否定／疑问边界规则；本版首先修正编码输入协议。"
        "新向量初评仍发现“非常／一部”这类功能词名称，冻结前加入通用中文词性过滤，"
        "保留revision_01原始QA与双评审、revision_02中间输出与部分复核；"
        "补齐功能词子类与情态词后，revision_03变化候选重新复核。风险层和向量不变，不重训、"
        "不读本版测试调参。30个真实开发会话和12个编写挑战清单始终保持相同。",
        "明确劝阻攻击的语言模式只约束解释分组，不覆盖分类分数；规则不能证明已识别所有讽刺、引用或隐性立场。孤立消息和不稳定分组允许不命名，仍保留风险证据。",
        "复核的逐例问题和漏组情况保留于两份评审记录；编写的校园挑战可能"
        "未形成主题。词组边界和功能词及情态词过滤规则不是命名实体识别器或立场理解模型，不能保证主题完整和语义正确。",
        "最高消息风险线索强度=max(所有消息、类别分数)，不是综合风险概率，也不是跨类别严重程度。SCCD会话评分另列；校园与其他领域仅为迁移参考。",
        "",
        "## 本地成本",
        "",
        f"训练墙钟时间：{summary['training_seconds']:.2f}秒；"
        f"训练峰值RSS：{summary['peak_process_rss_bytes'] / 1024**2:.2f}MiB。",
        "本版正式训练期间不并行运行其他训练或模型推理；墙钟时间仍包括资产验证、编码、拟合和保存，不视为算法核心纯计算时间。",
        f"新checkpoint：{(output / 'model.joblib').stat().st_size / 1024**2:.2f}MiB；"
        f"共享编码器资产：{summary['encoder_asset_bytes'] / 1024**2:.2f}MiB。"
        "需要CPU版PyTorch与sentence-transformers，无GPU要求；本地计算及硬件成本不等于零。",
    ]
    cost_path = output / "cost.json"
    if cost_path.exists():
        cost = read("cost.json")
        lines += [
            f"模型加载：{cost['model_load_seconds']:.3f}秒；进程首次推理：{cost['first_inference_seconds']:.3f}秒；驻留中位数/P95：{cost['warm_median_seconds']:.3f}/{cost['warm_p95_seconds']:.3f}秒。",
            f"计时覆盖42个开发/挑战会话、3次重复，推理峰值RSS：{cost['peak_process_rss_mib']:.2f}MiB；不冒充完整136测试耗时。",
            "安装依赖的版本与文件大小详见cost.json的deployment_dependencies；"
            "它不是独立模型参数大小，也不是最小发行包的估计。操作系统文件缓存未重置，首次推理不等于机器完全冷启动。",
        ]
    lines += [
        "",
        "## 复现与启动",
        "",
        "在 D:\\github\\intern 运行（固定CPU四线程）：",
        "",
        "```powershell",
        '$py = "D:\\github\\intern\\cst_mil\\.venv\\Scripts\\python.exe"',
        "& $py -B -X utf8 -m cst_mil.unified_semantic_v1_1.cli prepare",
        "& $py -B -X utf8 -m cst_mil.unified_semantic_v1_1.cli train --resume",
        "& $py -B -X utf8 -m cst_mil.unified_semantic_v1_1.cli qa",
        "# freeze仅在双评审完成、主题硬性检查通过且本版测试尚未打开时运行。",
        "# 部署性能门槛未通过仍可保存候选；不得自动替换默认。已完成版本不得重新冻结。",
        "& $py -B -X utf8 -m cst_mil.unified_semantic_v1_1.cli test",
        "```",
        "",
        "演示系统启动：",
        "",
        "```powershell",
        "cd D:\\github\\intern\\WebApp",
        "npm start",
        "```",
        "",
        "浏览器 http://127.0.0.1:3210。启动只加载部署清单中的一个版本，不重新训练。遇旧模型服务时按原窗口Ctrl+C关闭后再启动；不自动杀用户进程。",
        "候选预览可显式覆盖清单；默认是否已切换以deployment_status.json和系统验收记录为准：",
        "",
        "```powershell",
        "$env:CAMPUS_DEPLOYMENT_FILE = 'D:/github/intern/cst_mil/benchmark/research_followup/"
        "20261005/unified_semantic_v1_1/deployment.candidate.json'",
        "npm start",
        "# 恢复默认（关闭本次启动窗口后）：",
        "Remove-Item Env:CAMPUS_DEPLOYMENT_FILE",
        "```",
        "旧历史按schema1保留原分数、主题、模型版本与人工备注；新分析按schema2保存实际worker身份。",
        "恢复入口校验代码/数据/编码器/输出指纹；测试绑定freezeSHA和模型SHA。部署清单是Node与Python的共同真源，任一资产不匹配明确拒绝加载。",
        "",
        "所有完整预测、样本权重、OOF名单、排除清单、双评审和费用记录均在本目录；API调用次数为0。",
        "",
        "## 后续改进的依据",
        "",
        "当前不应仅换阈值或挑选测试最好配置。攻击类训练按来源等权，而验证pooled主要由SCCD"
        "评论构成；这是一项需另行检验的目标权重差异，不是已经证明的失败原因。"
        "若消息任务门槛仍未通过，应预先固定加入父回复语境的对照；"
        "仇恨监督仅来自ToxiCN，宜补充经过人工确认的语境和立场标签。",
        "讨论主题另需真实校园评论的人工分组及名称复核。当前编写挑战不能替代真实域金标，"
        "也不能用两个模型辅助审阅的分数宣称达到人工准确率。新增训练或变更协议应另建版本，"
        "保留本次未通过门槛的完整结果。",
    ]
    path = output / "结果与部署说明.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
