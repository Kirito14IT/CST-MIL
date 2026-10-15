"""Full result tables, length control and publication exports from saved predictions."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from .core import METHODS, OUT, PROJECT, digest, read, read_jsonl, save, verify_sources, write_jsonl

NAMES = {
    "lda": "LDA",
    "btm": "BTM",
    "gsdmm": "GSDMM",
    "nmf": "NMF",
    "bertopic": "BERTopic",
    "cst_mil_r0": "CST",
    "length_only": "Length only",
}


def _metrics(rows, threshold):
    from cst_mil.research_followup.round2.execution import evaluate

    if not rows:
        return dict(n_samples=0, ap=None, auc=None)
    result = evaluate(rows, threshold)
    if len({r["label"] for r in rows}) < 2:
        result.update(ap=None, auc=None)
    return result


def _csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(out=OUT):
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from cst_mil.metrics import select_threshold

    from .data import load_dataset

    out = Path(out)
    dest = out / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    master = read(out / "prepared/manifest.json")
    recovery_file = out / "recovered_runs.json"
    recovered = read(recovery_file)["runs"] if recovery_file.exists() else {}
    metrics, strata, lengths, limitations, integrity = [], [], [], [], []
    for dataset in master["datasets"]:
        manifest, sessions, messages = load_dataset(dataset, out)
        msg = {m["post_id"]: m for m in messages}
        raw_lengths = {s["post_id"]: len(msg[s["post_id"]]["raw_text"]) for s in sessions}
        train = [s for s in sessions if s["split"] == "train"]
        dev = [s for s in sessions if s["split"] == "dev"]
        test = [s for s in sessions if s["split"] == "test"]
        def features(ss, lengths=raw_lengths):
            return np.log1p([lengths[s["post_id"]] for s in ss]).reshape(-1, 1)
        baseline = make_pipeline(
            StandardScaler(),
            LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42),
        )
        baseline.fit(features(train), [s["label"] for s in train])
        threshold, _ = select_threshold(
            [s["label"] for s in dev], baseline.predict_proba(features(dev))[:, 1]
        )
        probs = baseline.predict_proba(features(test))[:, 1]
        baseline_rows = [
            dict(post_id=s["post_id"], label=s["label"], risk_probability=float(p))
            for s, p in zip(test, probs, strict=True)
        ]
        joblib.dump(baseline, dest / f"{dataset}_length_only.joblib")
        write_jsonl(dest / f"{dataset}_length_predictions.jsonl", baseline_rows)
        cuts = np.quantile([raw_lengths[s["post_id"]] for s in train], [1 / 3, 2 / 3]).tolist()
        save(
            dest / f"{dataset}_length_baseline.json",
            dict(
                threshold=threshold,
                feature="log1p(raw Unicode character count including spaces/punctuation)",
                train_only_fit=True,
                dev_only_threshold=True,
                tertile_cuts_train_only=cuts,
            ),
        )
        for split in ("train", "dev", "test"):
            for label in (0, 1):
                values = [
                    raw_lengths[s["post_id"]]
                    for s in sessions
                    if s["split"] == split and s["label"] == label
                ]
                lengths.append(
                    dict(
                        dataset=dataset,
                        split=split,
                        label=label,
                        n=len(values),
                        median=float(np.median(values)),
                        p95=float(np.quantile(values, 0.95)),
                        fraction_le30=float(np.mean(np.array(values) <= 30)),
                    )
                )
        for method in (*METHODS, "length_only"):
            recovery = recovered.get(f"{dataset}/{method}")
            if method == "length_only":
                rows = baseline_rows
                current_threshold = threshold
                costs = dict(
                    training_seconds=None,
                    test_seconds=None,
                    median_seconds=None,
                    p95_seconds=None,
                    loading_seconds=None,
                    peak_rss_mib=None,
                    checkpoint_mib=None,
                    encoder_mib=None,
                    java_mib=None,
                )
            else:
                folder = Path(recovery["path"]) if recovery else out / "runs" / dataset / method
                summary = read(folder / "summary.json")
                if recovery:
                    original_failure = Path(recovery["original_failure"])
                    if digest(original_failure) != recovery["original_failure_sha256"]:
                        raise ValueError("Original numerical failure evidence changed")
                    failed = read(original_failure)["binding"]
                    if failed["config"] != summary["config"]:
                        raise ValueError("Numerical recovery changed the configuration")
                    binding = read(folder / "binding.json")
                    if failed["cohort_fingerprint"] != binding["cohort_fingerprint"]:
                        raise ValueError("Numerical recovery changed the cohort")
                verify_sources(summary["sources"])
                verify_sources(summary["artifact_hashes"])
                verify_sources(summary["training"]["artifact_hashes"])
                rows = read_jsonl(folder / "test_predictions.jsonl")
                current_threshold = summary["threshold"]
                expected = {s["post_id"]: s["label"] for s in test}
                actual = {r["post_id"]: r["label"] for r in rows}
                if expected != actual or len(rows) != len(expected):
                    raise ValueError("Incomplete or mismatched test predictions")
                recomputed = _metrics(rows, current_threshold)
                if recomputed != summary["test"]["metrics"]:
                    raise ValueError("Metric recomputation differs")
                integrity.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        predictions=len(rows),
                        metrics_recomputed=True,
                    )
                )
                dep = summary["dependencies"]
                costs = dict(
                    training_seconds=summary["training"]["training_seconds"],
                    test_seconds=summary["test"]["latency"]["total_seconds"],
                    median_seconds=summary["test"]["latency"]["median_seconds"],
                    p95_seconds=summary["test"]["latency"]["p95_seconds"],
                    loading_seconds=summary["loading"]["seconds"],
                    peak_rss_mib=max(
                        summary["training"]["peak_rss_bytes"],
                        summary["loading"]["peak_rss_bytes"],
                        summary["test"]["peak_rss_bytes"],
                    )
                    / 2**20,
                    checkpoint_mib=dep["model_bytes"] / 2**20,
                    encoder_mib=dep["encoder_bytes"] / 2**20,
                    java_mib=dep["java_runtime_bytes"] / 2**20,
                )
            metric = _metrics(rows, current_threshold)
            metrics.append(
                dict(
                    dataset=dataset,
                    method=method,
                    n_test=len(rows),
                    threshold=current_threshold,
                    ap=metric["ap"],
                    auc=metric["auc"],
                    macro_f1=metric["macro_f1"],
                    precision=metric["precision"],
                    recall=metric["recall"],
                    run_revision=recovery["revision"] if recovery else "original",
                    original_failed_attempt=bool(recovery),
                    **costs,
                )
            )
            bins = {
                "le30": lambda n: n <= 30,
                "gt30": lambda n: n > 30,
                "short_train_tertile": lambda n, bounds=cuts: n <= bounds[0],
                "mid_train_tertile": lambda n, bounds=cuts: bounds[0] < n <= bounds[1],
                "long_train_tertile": lambda n, bounds=cuts: n > bounds[1],
            }
            for name, check in bins.items():
                selected = [r for r in rows if check(raw_lengths[r["post_id"]])]
                m = _metrics(selected, current_threshold)
                strata.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        stratum=name,
                        n=len(selected),
                        n_positive=sum(r["label"] for r in selected),
                        ap=m.get("ap"),
                        auc=m.get("auc"),
                        macro_f1=m.get("macro_f1"),
                        precision=m.get("precision"),
                        recall=m.get("recall"),
                    )
                )
        limitations.append(
            dict(
                dataset=dataset,
                source=manifest.get("source", manifest.get("sources")),
                unit="single_comment",
                no_message_top3_gold=True,
            )
        )
    _csv(dest / "metrics_costs.csv", metrics)
    _csv(dest / "length_strata.csv", strata)
    _csv(dest / "length_distribution.csv", lengths)
    save(
        dest / "verification.json",
        dict(
            method_runs=len(integrity),
            rows=integrity,
            complete_predictions=True,
            recomputed_metrics=True,
            limitations=limitations,
        ),
    )
    _figures(dest, metrics, lengths)
    report = _text(metrics, master, strata)
    (dest / "analysis.md").write_text(report, encoding="utf-8")
    report_dir = PROJECT / "reports/专题研究报告/短文本扩展_20261004"
    report_dir.mkdir(parents=True, exist_ok=True)
    _latex(report_dir, dest, metrics, report)
    save(
        dest / "delivery_manifest.json",
        {
            str(p.resolve()): digest(p)
            for p in [*dest.glob("*"), *report_dir.glob("*")]
            if p.is_file() and p.name != "delivery_manifest.json"
        },
    )
    return dest


def _figures(dest, metrics, lengths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, NullFormatter

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    datasets = list(dict.fromkeys(r["dataset"] for r in metrics))
    if any(r["median_seconds"] <= 0 for r in metrics if r["method"] != "length_only"):
        raise ValueError("Log latency requires strictly positive measurements")
    colors = ["#999999", "#4477AA", "#66CCEE", "#228833", "#CCBB44", "#165B78", "#AA3377"]
    markers = ["o", "v", "^", "s", "P", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.4))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.29, top=0.85, wspace=0.40)
    x = np.arange(len(datasets))
    width = 0.11
    for i, method in enumerate((*METHODS, "length_only")):
        chosen = [
            next(r for r in metrics if r["dataset"] == d and r["method"] == method)
            for d in datasets
        ]
        axes[0].bar(
            x + (i - 3) * width,
            [r["ap"] for r in chosen],
            width,
            label=NAMES[method],
            color=colors[i],
        )
        if method != "length_only":
            axes[1].scatter(
                [r["median_seconds"] for r in chosen],
                [r["ap"] for r in chosen],
                label=NAMES[method],
                color=colors[i],
                s=22,
                marker=markers[i],
            )
    display_names = {"conda": "CONDA", "toxicn": "ToxiCN", "scamspot": "ScamSpot", "hatebr": "HateBR"}
    axes[0].set_xticks(x, [display_names[d] for d in datasets], rotation=15,
                      ha="right", rotation_mode="anchor")
    axes[0].set_ylabel("Average precision")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("a  Risk ranking and length control", loc="left", fontsize=9)
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("Median inference latency (s, log scale)")
    axes[1].set_ylabel("Average precision")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("b  Quality and CPU response time", loc="left", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, fontsize=7, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    import os

    fig.canvas.draw()
    audit_scripts = os.environ.get("CST_FIGURE_AUDIT_SCRIPTS")
    if audit_scripts:
        import sys

        sys.path.insert(0, str(Path(audit_scripts).resolve(strict=True)))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig,
            json_out=dest / "figure_shorttext_comparison.alignment.json",
            overlay_svg=dest / "figure_shorttext_comparison.alignment.svg",
            strict=True,
        )
    else:
        save(dest / "figure_shorttext_comparison.alignment.json", {
            "status": "not_requested",
            "reason": "Optional figure audit tools not configured; set CST_FIGURE_AUDIT_SCRIPTS.",
        })
    fig.savefig(dest / "figure_shorttext_comparison.pdf", dpi=600)
    fig.savefig(dest / "figure_shorttext_comparison.svg", dpi=600)
    fig.savefig(dest / "figure_shorttext_comparison.png", dpi=600)
    fig.savefig(dest / "figure_shorttext_comparison.tiff", dpi=600)
    plt.close(fig)
    save(
        dest / "figure_contract.json",
        dict(
            width_mm=180,
            font_min_pt=7,
            quantitative_source="metrics_costs.csv",
            svg_editable_text=True,
            latency="complete one-comment inference in serial CPU process; reload cost separate",
            no_confidence_claim="single fixed seed, no significant-superiority assertion",
        ),
    )


def _text(metrics, master, strata):
    lines = [
        "# 短文本扩展实测分析",
        "",
        "以下为固定种子42、最多4线程、平衡单评论子集的比较。主题模型只在训练集拟合，分类阈值只由验证集选择。AP 衡量正类在排序前端的集中程度。单评论结果不验证多轮回复结构、MIL定位或消息Top-3。",
        "",
    ]
    for dataset in master["datasets"]:
        current = [r for r in metrics if r["dataset"] == dataset]
        cst = next(r for r in current if r["method"] == "cst_mil_r0")
        length = next(r for r in current if r["method"] == "length_only")
        others = [r for r in current if r["method"] in METHODS and r["method"] != "cst_mil_r0"]
        better = [NAMES[r["method"]] for r in others if cst["ap"] > r["ap"]]
        faster = [NAMES[r["method"]] for r in others if cst["median_seconds"] < r["median_seconds"]]
        lines.append(
            f"{dataset}：CST的AP为{cst['ap']:.4f}，AUC为{cst['auc']:.4f}，macro-F1为{cst['macro_f1']:.4f}，单条推理中位耗时{cst['median_seconds']:.4f}秒。AP超过的比较方法为{'、'.join(better) or '无'}；响应更快的比较方法为{'、'.join(faster) or '无'}。长度基线AP为{length['ap']:.4f}。这些差异来自本次固定子集，不能据此断言所有风险类型的普遍优势。"
        )
        lines.append("")
    lines += [
        "ScamSpot正类合并垃圾与诈骗评论，不能称为纯诈骗识别；正负长度差异须结合长度基线和分层表解读。赌博候选若未通过标注协议核验，已按固定计划替换为HateBR，来源审计保留原因。保留原始长度，不把全部样本宣传为30字符以内。",
        "",
        "完整成绩与成本见metrics_costs.csv；长度分层见length_strata.csv。峰值内存是监测到的进程与子进程RSS和；模型加载有缓存影响，未清空操作系统缓存。编码器和Java为外部部署依赖，不把检查点大小称为完整部署大小。",
        "LDA、BTM和GSDMM的原生适配器按输入启动子进程并载入模型，时间包含这一实现过程，不能解释为数学算法固有的速度下界。共享Windows主机的外部进程未受控，各阶段机器状态已保存。",
    ]
    cst_toxi = next(r for r in metrics if r["dataset"] == "toxicn" and r["method"] == "cst_mil_r0")
    length_toxi = next(r for r in metrics if r["dataset"] == "toxicn" and r["method"] == "length_only")
    if length_toxi["ap"] >= cst_toxi["ap"]:
        lines.append(
            f"长度控制：ToxiCN长度基线AP为{length_toxi['ap']:.4f}，高于CST的{cst_toxi['ap']:.4f}；本次中文评论子集仍存在明显长度信号，不能仅凭六方法排名宣称获得稳定语义识别能力。"
        )
    scam_short = next(r for r in strata if r["dataset"] == "scamspot" and r["method"] == "cst_mil_r0" and r["stratum"] == "le30")
    lines.append(
        f"ScamSpot测试集30字符以内仅{scam_short['n']}条，其中风险正类{scam_short['n_positive']}条；这一分层不足以支持极短诈骗评论的稳定结论。"
    )
    for group in ("short_train_tertile", "mid_train_tertile", "long_train_tertile"):
        current = next(r for r in strata if r["dataset"] == "scamspot" and r["method"] == "cst_mil_r0" and r["stratum"] == group)
        length = next(r for r in strata if r["dataset"] == "scamspot" and r["method"] == "length_only" and r["stratum"] == group)
        lines.append(
            f"ScamSpot训练长度三分位分层{group}（n={current['n']}，正类={current['n_positive']}）：CST AP={current['ap']:.4f}，长度基线AP={length['ap']:.4f}。这提供长度之外预测信息的描述性证据，但分层内仍有长度差异，也不构成语义因果证明。"
        )
    recovered = [r for r in metrics if r.get("original_failed_attempt")]
    if recovered:
        affected = "、".join(f"{r['dataset']}/{NAMES[r['method']]}" for r in recovered)
        lines.append(
            f"数值恢复：{affected}的原始MU拟合因中间比值溢出产生NaN，失败记录保留。独立nmf_stable_v1按相同数学更新保留零锁定，正项乘积溢出风险时使用对数求值；目标函数、solver、初始化、种子、主题数、迭代上限、容差及数据划分均未改变。修复仅按训练数据的有限性与数学等价诊断确定，未依据测试成绩选择。本轮为24个预设配置的成功模型，另保留2次技术失败，合计26次六方法正式拟合尝试，未增加参数搜索；4个长度基线及训练数据诊断不计入这一口径。表中为恢复后成功运行的成本，原失败尝试没有精确计时，不并入该值。"
        )
    return "\n".join(lines) + "\n"


def _latex(folder, dest, metrics, report):
    def fmt(v):
        return "--" if v is None else f"{v:.4f}"

    tables = []
    for dataset in dict.fromkeys(r["dataset"] for r in metrics):
        tables += [
            r"\subsection{" + dataset + "}",
            r"\begin{tabular}{lrrrrr}\toprule",
            r"方法 & AP & AUC & macro-F1 & 精确率 & 召回率\\\midrule",
        ]
        for r in metrics:
            if r["dataset"] == dataset:
                tables.append(
                    NAMES[r["method"]]
                    + " & "
                    + " & ".join(
                        fmt(r[k]) for k in ("ap", "auc", "macro_f1", "precision", "recall")
                    )
                    + r"\\"
                )
        tables += [r"\bottomrule\end{tabular}"]
    costs = [
        r"\begin{longtable}{llrrrrr}\toprule",
        r"数据 & 方法 & 训练秒 & 测试秒 & 中位秒 & RSS MiB & 模型MiB\\\midrule\endfirsthead",
        r"\toprule 数据 & 方法 & 训练秒 & 测试秒 & 中位秒 & RSS MiB & 模型MiB\\\midrule\endhead",
    ]
    for r in metrics:
        if r["method"] == "length_only":
            continue
        costs.append(
            r["dataset"]
            + " & "
            + NAMES[r["method"]]
            + " & "
            + " & ".join(
                fmt(r[k])
                for k in (
                    "training_seconds",
                    "test_seconds",
                    "median_seconds",
                    "peak_rss_mib",
                    "checkpoint_mib",
                )
            )
            + r"\\"
        )
    costs += [r"\bottomrule\end{longtable}"]
    body = "\n\n".join(line for line in report.splitlines() if line and not line.startswith("#"))
    tex = (
        r"""\documentclass[UTF8,a4paper,11pt]{ctexart}
\usepackage[margin=22mm]{geometry}
\usepackage{booktabs,longtable,graphicx,hyperref}
\setCJKmainfont[BoldFont=SimHei,ItalicFont=KaiTi,SlantedFont=KaiTi]{SimSun}\setmainfont{TeX Gyre Termes}
\hypersetup{colorlinks=true,urlcolor=blue}
\title{短文本风险扩展实验补充报告}\author{CST-MIL研究项目}\date{2026年10月4日}
\begin{document}\maketitle
\section{任务边界与固定协议}
本轮新增数据分别来自真实游戏聊天和社交平台评论。30字符是参考长度，正文不截断。训练、验证和测试均为受控平衡子集，其精确率与AP不能直接解释为原始类别比例下的部署指标。单评论不具备可验证的回复结构，也没有消息级证据金标。
来源、许可、标签映射、冲突与近重复排除均保存在prepared和sources目录。CONDA按对局划分，ToxiCN保持官方测试来源；近重复组件跨分区隔离。种子42，CPU最多4线程，六方法沿用冻结参数。
\section{排序与分类结果}
"""
        + "\n".join(tables)
        + r"""
\section{计算成本}
训练时间含验证集预测和阈值选择，测试时间为保存模型逐条推理的耗时和；加载单列于结果CSV。模型MiB指检查点，不包含完整Python环境。BERTopic还需外部编码器，GSDMM还需Java运行时。全部方法使用CPU，未使用GPU。
"""
        + "\n".join(costs)
        + r"""
\section{图与简短分析}
\begin{center}\includegraphics[width=\textwidth]{"""
        + (dest / "figure_shorttext_comparison.pdf").as_posix()
        + "}\\end{center}\n\n"
        + body.replace("_", r"\_")
        + r"""
\section{来源与复现}
\begin{itemize}
\item CONDA：\url{https://github.com/usydnlp/CONDA}
\item ToxiCN：\url{https://github.com/DUT-lujunyu/ToxiCN}
\item ScamSpot：\url{https://aclanthology.org/2024.eacl-demo.9/}
\item 赌博候选：\url{https://zenodo.org/records/15424844}；备用HateBR：\url{https://github.com/franciellevargas/HateBR}
\end{itemize}
固定清单、数据指纹、完整模型与预测、长度分层与成本字段均独立保存。复现入口为
\texttt{python -m cst\_mil.research\_followup.shorttext}。数值恢复使用独立\texttt{shorttext\_nmf\_stable}模块；完成记录由\texttt{resume\_completed.py}按原始/恢复映射核验，旧失败目录保持冻结。完整命令见实验目录README。
\end{document}
"""
    )
    (folder / "supplement.tex").write_text(tex, encoding="utf-8")
    engine = Path("D:/texlive/2026/bin/windows/xelatex.exe")
    for _ in range(2):
        result = subprocess.run(
            [str(engine), "-interaction=nonstopmode", "-halt-on-error", "supplement.tex"],
            cwd=folder,
            capture_output=True,
        )
        (folder / "compile_output.txt").write_bytes(result.stdout)
        if result.returncode:
            raise RuntimeError("Supplement LaTeX compilation failed")
