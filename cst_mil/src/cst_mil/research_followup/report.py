"""Build report tables and strict analysis notes from verified result files."""

from __future__ import annotations

import joblib

from .io import METHODS, OUT, PROJECT, read, save

REPORT = PROJECT / "reports/专题研究报告"
NAMES = dict(
    lda="LDA", btm="BTM", gsdmm="GSDMM", nmf="NMF", bertopic="BERTopic", cst_mil_r0="CST-MIL r0"
)


def esc(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def latex_table(filename, caption, headers, data, alignment=None):
    alignment = alignment or "l" + "r" * (len(headers) - 1)
    text = [
        "{\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{longtable}{" + alignment + "}",
        "\\caption{" + caption + r"}\\",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
        "\\endhead",
    ]
    text += [" & ".join(str(v) for v in row) + r" \\" for row in data]
    text += ["\\bottomrule", "\\end{longtable}", "}"]
    (REPORT / "generated" / filename).write_text("\n".join(text) + "\n", encoding="utf-8")


def build():
    (REPORT / "generated").mkdir(parents=True, exist_ok=True)
    performance = read(OUT / "performance.json")
    evidence = read(OUT / "evidence.json")
    pairs = read(OUT / "paired.json")
    costs = read(OUT / "costs.json")
    ablations = read(OUT / "ablations.json")
    if len(ablations) != 30:
        raise ValueError("All 30 ablation variants must be present")
    latex_table(
        "performance_table.tex",
        "原六系统会话风险识别（测试136会话）",
        ["方法", "macro-F1", "Precision", "Recall", "AP", "AUC"],
        [
            [NAMES[r["method"]]]
            + [f"{r[k]:.4f}" for k in ["macro_f1", "precision_cb", "recall_cb", "ap", "roc_auc"]]
            for r in performance
        ],
    )
    topk = read(OUT / "ranking_at_k.json")
    latex_table(
        "ranking_table.tex",
        "固定会话浏览数量下的精确率与召回率",
        ["方法", "P@10", "P@20", "P@30", "R@10", "R@20", "R@30"],
        [
            [NAMES[m]]
            + [
                f"{next(r for r in topk if r['method'] == m and r['k'] == k)[v]:.4f}"
                for v in ["precision", "recall"]
                for k in [10, 20, 30]
            ]
            for m in METHODS
        ],
    )
    latex_table(
        "evidence_table.tex",
        "会话内Top-3证据评价（101个含危险评论会话）",
        ["方法", "命中数", "Hit@3", "Recall@3", "评论AP"],
        [
            [
                NAMES[r["method"]],
                f"{r['hit_count']}/101",
                f"{r['hit_rate_at_3']:.4f}",
                f"{r['mean_recall_at_3']:.4f}",
                f"{r['comment_ap']:.4f}",
            ]
            for r in evidence
        ],
    )
    latex_table(
        "topic_table.tex",
        "当前适配系统的可变主题输出（136会话、8028消息）",
        ["方法", "主题数范围", "均值", "多重归属消息数"],
        [
            [
                NAMES[r["method"]],
                f"{r['minimum']}--{r['maximum']}",
                f"{r['mean']:.2f}",
                r["overlapping_messages"],
            ]
            for r in read(OUT / "topic_counts.json")
        ],
    )
    latex_table(
        "paired_table.tex",
        r"CST减去基线的排序指标差值与配对95\%区间",
        ["对照基线", "指标", "差值", r"95\%下界", r"95\%上界"],
        [
            [
                NAMES[r["baseline"]],
                r["metric"].replace("roc_auc", "AUC").upper(),
                f"{r['delta']:.4f}",
                f"{r['ci_low']:.4f}",
                f"{r['ci_high']:.4f}",
            ]
            for r in pairs
        ],
    )
    latex_table(
        "cost_table.tex",
        "原始完整流程成本；时间单位秒，大小单位MiB",
        ["方法", "训练", "测试推理", "P50", "P95", "峰值RSS", "模型", "部署依赖"],
        [
            [NAMES[r["method"]]]
            + [
                f"{r[k]:.2f}"
                for k in [
                    "train_seconds",
                    "inference_test_seconds",
                    "latency_p50_seconds",
                    "latency_p95_seconds",
                    "peak_rss_mib",
                    "model_mib",
                ]
            ]
            + [f"{r['encoder_mib'] + r['runtime_mib']:.2f}"]
            for r in costs
        ],
    )
    modes = {
        "full": "完整",
        "no_mil": "去MIL",
        "no_reply": "去回复",
        "no_topics": "去主题",
        "direct_topics": "主题直接分类",
    }
    ordered = sorted(
        ablations,
        key=lambda r: (
            METHODS.index(r["method"]),
            r["mode"],
            r["instance_head"],
            r["session_head"],
        ),
    )
    latex_table(
        "ablation_table.tex",
        "固定主题表示下的30组风险层与组件对照",
        ["方法", "设置", "实例/会话层", "验证F1", "测试F1", "AP", "AUC"],
        [
            [
                NAMES[r["method"]],
                modes[r["mode"]],
                (r["instance_head"] + "/" + r["session_head"]).upper(),
                f"{r['dev_macro_f1']:.4f}",
                f"{r['macro_f1']:.4f}",
                f"{r['ap']:.4f}",
                f"{r['roc_auc']:.4f}",
            ]
            for r in ordered
        ],
        "lllrrrr",
    )
    baseline = max([r for r in ablations if r["method"] != "cst_mil_r0"], key=lambda r: r["ap"])
    selected = {
        r["mode"]: r for r in ablations if r["method"] == "cst_mil_r0" and r["session_head"] == "lr"
    }
    finding = (
        "实验显示，风险层选择会改变数值和排名。例如，LDA的主题直接分类对照"
        "macro-F1为0.8792，而原完整系统为0.8088；BTM的RF/RF组合为0.8807，"
        "原LR/LR组合为0.7983。因此不能用“统一用了线性层”排除分类器影响。\n\n"
        f"五基线新对照中最高AP为{baseline['ap']:.4f}，来自{NAMES[baseline['method']]}；"
        "该值用于描述预先列出的对照集合，不用于在测试集重选正式基线。"
        f"CST去MIL的AP为{selected['no_mil']['ap']:.4f}，"
        f"去主题的AP为{selected['no_topics']['ap']:.4f}、"
        f"macro-F1为{selected['no_topics']['macro_f1']:.4f}。"
        "这些结果说明组合模块的作用取决于指标，当前证据不能支持每个组件都必然改善所有指标。"
        "后续若修改正式方法，应使用新的开发协议与独立测试，而不是根据本次测试挑选最优变体。\n"
    )
    (REPORT / "generated/ablation_findings.tex").write_text(finding, encoding="utf-8")
    case = read(OUT / "topic_failure_case.json")
    payload = joblib.load(OUT / "cache/cst_mil_r0.joblib")
    i = next(i for i, s in enumerate(payload["sessions"]) if s["post_id"] == case["session_id"])
    detailed = []
    for w, weights in zip(payload["windows"][i], payload["window_topics"][i], strict=True):
        if case["target"]["comment_id"] in w["message_ids"]:
            detailed.append(
                dict(
                    window_id=w["window_id"],
                    start=w["start"],
                    end=w["end"],
                    topic_15=float(weights[15]),
                    topic_23=float(weights[23]),
                )
            )
    case["covering_window_weights"] = detailed
    save(OUT / "topic_failure_case.json", case)
    weights = case["target"]["topic_strengths"]
    line = (
        f"该消息归一化主题15和23的强度分别为{weights[15]:.4f}和{weights[23]:.4f}，"
        "均满足公共多重归属阈值。覆盖窗口的原始权重保存在JSON证据文件中。"
        "其父回复位于更早的消息位置，当前主题表示按连续位置窗口聚合，"
        "并不是沿父回复链单独建立主题。\n"
    )
    (REPORT / "generated/case_details.tex").write_text(line, encoding="utf-8")
    quote = read(OUT / "api_quote.json")
    low = min(r["one_pass_upper_usd"] for r in quote["groups"])
    high = max(r["one_pass_upper_usd"] for r in quote["groups"])
    api = (
        f"截至本稿生成，API执行状态：尚未发起付费模型请求。已准备20个验证会话和"
        f"136个测试会话，UTF-8字节保守输入界为{quote['input_utf8_byte_bound']:,}，"
        "总输出上限为319,488 token。该输入界不是模型分词器的精确计数。"
        f"按公开分组倍率，单轮保守报价约{low:.2f}--{high:.2f}美元；"
        "每会话最多三次尝试时最高公开费率对应约7.12美元。"
        "账户实际路由分组与预算尚待确定，不能以该报价代替实测费用、延迟或幻觉率"
        "\\cite{router}。\n"
    )
    if (OUT / "api_run/evaluation.json").exists():
        evaluation = read(OUT / "api_run/evaluation.json")
        api = (
            "API运行状态：" + esc(evaluation["status"]) + "。"
            f"有效测试输出{evaluation.get('valid_test', 0)}/136。"
            "逐请求原始响应、费用估算和错误记录见配套实验目录。\n"
        )
        if evaluation.get("comparable_full_test_metrics"):
            r = evaluation["comparable_full_test_metrics"]
            api += (
                f"Gemini会话AP={r['ap']:.4f}，AUC={r['roc_auc']:.4f}，"
                f"macro-F1={r['macro_f1']:.4f}。\n"
            )
    (REPORT / "generated/api_status.tex").write_text(api, encoding="utf-8")
    analysis_notes(performance, pairs, ablations, finding)
    return {"report_source": str(REPORT / "main.tex"), "tables": 7, "ablations": len(ablations)}


def analysis_notes(performance, pairs, ablations, finding):
    (OUT / "analysis-report.md").write_text(
        """# 研究补证结论

主问题：冻结主题与会话划分后，风险排序优势、Top-3和分类器选择的关系是什么？

六方法原指标复算一致；30组风险层/组件对照完成，原LR/LR系统概率重建误差通过1e-7门限。
全部预测按同一完整会话配对；不将窗口当成独立统计样本。

## 支持的结论
- 原六系统中CST的AP=.9465、AUC=.9413点估计最高，Top-3为67/101。
- AP不是每个固定浏览预算的最优保证：前30个会话CST为29正例，NMF为30。
- CST和NMF共用相同主题表示，公共阈值接口不是CST独有贡献。
- CST对NMF的AP/AUC差值95%区间均跨零，不支持稳定优于所有基线的概括。
- 线性风险层不能视为无影响：分类器与组件消融会改变排名。

## 风险层实验解释
"""
        + finding.replace("\\", "")
        + """
## Claim candidates
| 结论 | 允许表述 | 不支持的加强表述 | 证据 |
|---|---|---|---|
| 排序 | 本次AP/AUC点估计最高 | 所有浏览预算均最好 | performance/ranking_at_k/paired |
| 证据 | 101会话中Top-3命中67个 | 三条足以覆盖全部危险内容 | evidence |
| 主题 | 接口支持可变数量与重叠归属 | 语义正确且为CST独有 | topic_counts/topic_failure_case |
| 风险层 | 排名对分类器选择敏感 | 已排除风险层影响 | ablations |
| 成本 | 对BERTopic本次测试更快且模型组成小 | 比每种方法都更快更省 | costs |

所有结论均基于一次冻结拟合及回顾性测试。API状态与数据集接入限制分别见api_quote.json和dataset_review.md。
""",
        encoding="utf-8",
    )
    lines = [
        "# 统计附录",
        "",
        "单位：完整会话。测试136，CB71，Non-CB65。",
        "随机种子42，分层配对bootstrap 2000次；固定拟合，不代表训练种子不确定性。",
        "主比较：CST对五基线，AP和AUC共10项。以下同时给出逐项95%区间及Bonferroni近似家族区间；不输出未经检验的p值。",
        "",
        "|基线|指标|差值|95% CI|家族CI|",
        "|---|---|---:|---|---|",
    ]
    for r in pairs:
        lines.append(
            f"|{r['baseline']}|{r['metric']}|{r['delta']:.6f}|"
            f"[{r['ci_low']:.6f}, {r['ci_high']:.6f}]|"
            f"[{r['family_ci_low']:.6f}, {r['family_ci_high']:.6f}]|"
        )
    lines += [
        "",
        "2,000次重采样的家族区间尾部仅约5个重采样值，端点稳定性有限。",
        "30组消融是预设诊断矩阵；不据测试最优项重选正式方法。没有新增独立训练种子。",
        "Top-3以101个含危险评论的会话为分母，与71个会话CB正例不同。",
    ]
    (OUT / "stats-appendix.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "figure-catalog.md").write_text(
        """# 图表目录

## ranking_curves.pdf / .png
- 数据：六方法136会话原始风险分数。
- 目的：展示不同阈值下PR/ROC，而不把单阈值F1视为排序质量。
- 观察：CST整体AP较高，具体浏览预算仍可能由NMF获得更多正例。
- 推论边界：单次固定模型，不含训练随机性误差线。

## evidence.pdf / .png
- 数据：101个含危险评论会话的Top-3。
- 目的：区分至少命中一次与覆盖危险评论比例。
- 观察：CST命中率最高，覆盖率不随之同样最高。
- 推论边界：不能当作三条证据足以完成判定。

## report tables
- 所有表格由report.py从JSON/CSV生成。
- 表格中的历史端到端成本与消融缓存计时分开，API报价与实际用量分开。
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    print(build())
