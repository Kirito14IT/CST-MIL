"""Source-backed Python figures for the completed full joint experiment.

Figure contract: compare within each fixed evaluation task, then display the
paired change against a validation-selected topic baseline. Costs have a
separate quantitative-grid figure and never enter the classification ranking.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import fontManager
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

from .analysis import MAIN_TASKS, METHODS


DISPLAY = {"lda": "LDA", "btm": "BTM", "gsdmm": "GSDMM", "nmf": "NMF",
           "bertopic": "BERTopic", "cst_v1_1": "CST v1.1", "minilm_lr": "MiniLM+LR"}
TASK_DISPLAY = {"sccd.attack": "SCCD · 攻击评论", "conda.attack": "CONDA · 攻击",
                "hatebr.attack": "HateBR · 冒犯", "toxicn.attack": "ToxiCN · 冒犯",
                "toxicn.hate": "ToxiCN · 仇恨", "scamspot.spam_scam": "ScamSpot · 垃圾/诈骗",
                "sccd.conversation": "SCCD · 整段会话"}
COLORS = {m: "#AEBBC4" for m in METHODS}
COLORS.update({"cst_v1_1": "#2B7877", "minilm_lr": "#9B78A4"})


def _font() -> None:
    for name in ("msyh.ttc", "msyhbd.ttc"):
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            fontManager.addfont(str(path))
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 7, "legend.fontsize": 6.5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "figure.facecolor": "white"})


def _skill_path(value: str | Path | None) -> Path:
    path = Path(value) if value else Path(os.environ.get("NATURE_FIGURE_SKILL", str(Path.home() / ".codex/skills/nature-figure")))
    scripts = path / "scripts"
    if not (scripts / "audit_panel_alignment.py").exists():
        raise FileNotFoundError("nature-figure alignment/QA scripts unavailable; set NATURE_FIGURE_SKILL")
    return scripts


def _qa_interpreter(value: str | Path | None) -> str:
    if value:
        return str(value)
    # The model environment can omit PDF QA dependencies; use the already
    # available document environment without changing the plotting backend.
    try:
        import fitz  # noqa: F401
        return sys.executable
    except ImportError:
        candidate = Path(__file__).resolve().parents[4] / "reports/专利交底书/20261004/tools/.venv/Scripts/python.exe"
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("PyMuPDF is required for rendered collision QA; provide qa_python")


def _run_qa(scripts: Path, qa_python: str, pdf: Path, qa: Path) -> list[dict[str, Any]]:
    prefix = pdf.stem
    commands = [
        ("source", [sys.executable, str(scripts / "validate_figure.py"), str(Path(__file__).resolve()), "--json"]),
        ("font", [sys.executable, str(scripts / "audit_pdf_text.py"), str(pdf), "--min-pt", "5", "--json"]),
        ("collisions", [qa_python, str(scripts / "audit_figure_collisions.py"), str(pdf),
                        "--json-out", str(qa / f"{prefix}.collisions.json"),
                        "--overlay-pdf", str(qa / f"{prefix}.collisions.pdf")]),
    ]
    result = []
    for name, command in commands:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        (qa / f"{prefix}.{name}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        result.append({"audit": name, "returncode": completed.returncode,
                       "log": f"{prefix}.{name}.log"})
    if any(x["returncode"] != 0 for x in result):
        raise RuntimeError(f"Figure QA blocked for {prefix}: {result}")
    return result


def _save(fig: Any, base: Path, qa: Path, scripts: Path, qa_python: str,
          *, axes: list[Any], panel_ids: list[str], exemptions: list[dict] | None = None) -> list[str]:
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=panel_ids,
        row_groups=[{"id": "comparable-row", "panels": panel_ids}] if len(axes) == 2 else None,
        exemptions=exemptions or [], tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        json_out=qa / f"{base.name}.alignment.json",
        overlay_svg=qa / f"{base.name}.alignment.svg", strict=True)
    files = []
    for suffix in (".pdf", ".svg", ".png"):
        path = base.with_suffix(suffix)
        fig.savefig(path, dpi=600)
        files.append(str(path))
    plt.close(fig)
    audits = _run_qa(scripts, qa_python, base.with_suffix(".pdf"), qa)
    (qa / f"{base.name}.qa.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    return files


def _contrast(data: dict, task: str, baseline: str) -> tuple[float, float, float]:
    for row in data["contrasts"]:
        if row["task"] != task or row["metric"] != "AP":
            continue
        if row["left"] == "cst_v1_1" and row["right"] == baseline:
            return row["difference"], row["ci_low"], row["ci_high"]
        if row["right"] == "cst_v1_1" and row["left"] == baseline:
            return -row["difference"], -row["ci_high"], -row["ci_low"]
    raise ValueError(f"Missing paired CST contrast for {task}")


def _performance(data: dict, base: Path, qa: Path, scripts: Path, qa_python: str) -> list[str]:
    methods = data["methods"]
    summary = {(r["task"], r["method"]): r for r in data["summary"] if r["split"] == "test"}
    values = np.array([[summary[task, method]["AP_mean"] for method in methods] for task in MAIN_TASKS])
    if not np.isfinite(values).all():
        raise ValueError("Cannot draw missing or nonfinite task AP")
    comparisons = [_contrast(data, task, data["selected_baselines"][task]["method"]) for task in MAIN_TASKS]
    if not np.isfinite(np.array(comparisons)).all():
        raise ValueError("Cannot draw nonfinite paired intervals")
    fig = plt.figure(figsize=(7.20472440945, 4.8031496063))  # 183 x 122 mm
    a = fig.add_axes([.245, .255, .455, .525], label="a")
    b = fig.add_axes([.765, .255, .205, .525], label="b")
    cmap = LinearSegmentedColormap.from_list("AP", ["#F6F9FA", "#D9E8E8", "#89B5B3", "#2B7474"])
    a.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="none")
    a.set_xticks(range(len(methods)), [DISPLAY[m] for m in methods], rotation=45, ha="right", rotation_mode="anchor")
    labels = [f"{TASK_DISPLAY[t]}\nn={summary[t, methods[0]]['n']:,}; 阳性={summary[t, methods[0]]['prevalence']:.1%}" for t in MAIN_TASKS]
    a.set_yticks(range(len(MAIN_TASKS)), labels)
    a.tick_params(length=0, pad=5)
    for i in range(len(MAIN_TASKS)):
        for j in range(len(methods)):
            # Choose the more legible ink from the actual cell colour rather
            # than an AP cutoff; mid-tone teal can make white numbers faint.
            rgb = np.asarray(cmap(values[i, j])[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([.2126, .7152, .0722]))
            ink = "white" if 1.05 / (luminance + .05) >= (luminance + .05) / .05 else "black"
            a.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                   color=ink)
    a.set_xticks(np.arange(-.5, len(methods), 1), minor=True)
    a.set_yticks(np.arange(-.5, len(MAIN_TASKS), 1), minor=True)
    a.grid(which="minor", color="white", linewidth=1.3)
    a.tick_params(which="minor", length=0)
    for spine in a.spines.values():
        spine.set_visible(False)
    column = methods.index("cst_v1_1")
    a.add_patch(Rectangle((column - .5, -.5), 1, len(MAIN_TASKS), fill=False, ec="#264E55", lw=1.1, clip_on=False))
    a.axhline(5.5, color="#3C5965", lw=1)
    a.text(0, 1.105, "a", transform=a.transAxes, fontsize=9, fontweight="bold")
    a.text(.075, 1.105, "测试 AP · 三个种子的均值", transform=a.transAxes, fontsize=8.2, fontweight="bold")
    low = min(x[1] for x in comparisons)
    high = max(x[2] for x in comparisons)
    span = max(high - low, .08)
    b.set_xlim(min(low - .15 * span, -.015), max(high + .15 * span, .015))
    b.set_ylim(6.5, -.5)
    b.axvline(0, color="#98A8B1", lw=.8, ls=(0, (3, 3)))
    for i, (center, lo, hi) in enumerate(comparisons):
        color = "#2B7877" if center >= 0 else "#AA654D"
        # Percentile bounds need not contain the observed statistic. Draw the
        # actual interval independently rather than clipping an error length.
        b.hlines(i + .08, lo, hi, color=color, lw=1)
        b.vlines([lo, hi], i + .035, i + .125, color=color, lw=1)
        b.plot(center, i + .08, "o", color=color, markersize=3.4)
        baseline = DISPLAY[data["selected_baselines"][MAIN_TASKS[i]]["method"]]
        b.text(.98, i - .22, f"{center:+.3f} · {baseline}",
               transform=b.get_yaxis_transform(), ha="right", va="center", fontsize=6.4, color="#3D5360")
    b.axhline(5.5, color="#3C5965", lw=1)
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    b.tick_params(axis="x", labelsize=6.3)
    b.set_xlabel("CST − 验证集所选主题基线", labelpad=6, fontsize=6.7)
    b.text(0, 1.105, "b", transform=b.transAxes, fontsize=9, fontweight="bold")
    b.text(.13, 1.105, "配对 AP 差值", transform=b.transAxes, fontsize=8.2, fontweight="bold")
    fig.text(.035, .94, "五来源全量比较：v1.1 与统一监督基线", fontsize=11, fontweight="bold", color="#243C46")
    subtitle = "同一划分、标签掩码与输出层；七种方法，各训练三个种子"
    if data.get("evaluation_revision"):
        subtitle += "；固定权重原文评估修订"
    fig.text(.035, .885, subtitle, fontsize=7.4, color="#526B77")
    fig.text(.035, .105, "末行为会话，其余为消息；AP 受阳性比例影响，比较应在同一行内进行。", fontsize=6.8, color="#526B77")
    attempted = next(iter(data["bootstrap"].values()))["attempted"]
    fig.text(.035, .065, f"区间：{attempted:,} 次配对依赖簇 bootstrap 的95%区间；每次先计算各种子指标再取均值。", fontsize=6.8, color="#526B77")
    return _save(fig, base, qa, scripts, qa_python, axes=[a, b], panel_ids=["a", "b"],
        exemptions=[{"panels": ["a", "b"], "checks": ["panel-width"],
                     "reason": "Heatmap and paired-effect forest have different roles and intentional widths; shared top/bottom edges audited."}])


def _number(cost: dict, key: str) -> float:
    value = cost.get(key)
    if value is None or not np.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"Missing/nonfinite/negative cost: {cost['method']}/{key}")
    return float(value)


def _costs(data: dict, base: Path, qa: Path, scripts: Path, qa_python: str) -> list[str]:
    costs = {row["method"]: row for row in data["costs"]}
    methods = data["methods"]
    if set(methods) - costs.keys():
        raise ValueError("Cost figure requires every seed-42 method cost record")
    if any(row.get("GPU_required") is not False for row in costs.values()):
        raise ValueError("Cost records must explicitly declare GPU_required=false")
    x = np.arange(len(methods))
    colors = [COLORS[m] for m in methods]
    fig, axes = plt.subplots(2, 2, figsize=(7.20472440945, 5.70866141732))  # 183 x 145 mm
    fig.subplots_adjust(left=.095, right=.975, bottom=.19, top=.83, wspace=.28, hspace=.76)
    a, b, c, d = axes.ravel()
    cold = np.array([_number(costs[m], "training_seconds_cold") for m in methods])
    cached = np.array([_number(costs[m], "training_seconds_cached") for m in methods])
    training_log = bool(np.all(cold > 0) and np.all(cached > 0) and np.max(cold) / np.min(cached) > 30)
    training_bottom = float(min(cold.min(), cached.min()) / 2) if training_log else 0.
    a.bar(x - .16, cold - training_bottom, bottom=training_bottom,
          width=.29, fc="white", ec=colors, lw=1.1, label="含首次编码")
    a.bar(x + .16, cached - training_bottom, bottom=training_bottom,
          width=.29, color=colors, label="已有编码缓存")
    a.set_ylabel("训练时间 / 秒")
    if training_log:
        a.set_yscale("log")
        a.set_ylim(bottom=training_bottom)
        a.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}"))
    a.legend(loc="upper left", ncol=1, handlelength=1.1)
    totals, spreads = [], []
    for method in methods:
        warm = costs[method].get("warm_total_seconds")
        if isinstance(warm, list):
            if len(warm) != 3 or not np.isfinite(warm).all() or np.any(np.array(warm) < 0):
                raise ValueError("Warm inference requires three valid repetitions")
            totals.append(float(np.mean(warm)))
            spreads.append(float(np.std(warm, ddof=1)))
        elif warm is not None:
            raise ValueError("Keep three individual warm repetitions rather than a mean-only cost record")
        else:
            raise ValueError("Missing warm_total_seconds repetitions")
    inference_log = bool(np.all(np.array(totals) > 0) and max(totals) / min(totals) > 30)
    inference_bottom = float(min(totals) / 2) if inference_log else 0.
    b.bar(x, np.asarray(totals) - inference_bottom, bottom=inference_bottom,
          color=colors, width=.62, yerr=spreads, capsize=2,
          error_kw={"elinewidth": .8, "ecolor": "#435A65"})
    b.set_ylabel("完整测试风险预测 / 秒")
    if inference_log:
        b.set_yscale("log")
        b.set_ylim(bottom=inference_bottom)
        b.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:g}"))
    model = np.array([_number(costs[m], "model_bytes") for m in methods]) / 1024 ** 2
    encoder = np.array([_number(costs[m], "encoder_asset_bytes") for m in methods]) / 1024 ** 2
    dependency = np.array([_number(costs[m], "runtime_dependency_bytes") for m in methods]) / 1024 ** 2
    c.bar(x, model, color=colors, width=.62, label="模型")
    c.bar(x, encoder, bottom=model, color="#D0DEDF", width=.62, label="编码器")
    c.bar(x, dependency, bottom=model + encoder, color="#E8E2D7", width=.62, label="运行依赖")
    c.set_ylabel("部署资产 / MiB")
    c.legend(loc="upper left", ncol=3, handlelength=.9, columnspacing=.7)
    rss = [_number(costs[m], "peak_tree_rss_bytes") / 1024 ** 2 for m in methods]
    d.bar(x, rss, color=colors, width=.62)
    d.set_ylabel("峰值进程树 RSS / MiB")
    titles = ["训练：原冻结流程与缓存", "风险预测：完整测试集", "资产：计入编码器与依赖", "内存：包含子进程"]
    for letter, ax, title in zip("abcd", axes.ravel(), titles):
        ax.set_xticks(x, [DISPLAY[m] for m in methods], rotation=60, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, labelsize=6.5)
        ax.tick_params(axis="y", labelsize=6.5)
        if ax not in (a, c):
            ax.grid(axis="y", alpha=.16, lw=.6)
        ax.set_axisbelow(True)
        ax.text(0, 1.09, letter, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.text(.065, 1.09, title, transform=ax.transAxes, fontsize=7.9, fontweight="bold")
        if ax.get_yscale() == "linear":
            ax.set_ylim(0, ax.get_ylim()[1] * (1.55 if ax in (a, c) else 1.18))
        elif ax is a:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * (hi / lo) ** .32)
    fig.text(.035, .95, "同机 CPU 成本：风险预测与部署资产", fontsize=11, fontweight="bold", color="#243C46")
    fig.text(.035, .905, "本机桌面负载；种子42；最多4线程；无需GPU；不含动态主题聚类与命名", fontsize=7.4, color="#526B77")
    fig.text(.035, .06, "推理柱为驻留模型3次完整测试的均值，误差条为标准差；不把批量耗时称为单条延迟。", fontsize=6.8, color="#526B77")
    footer = ("训练柱沿用原冻结流程；原文验证修订开销、启动、编码器及依赖口径另见完整成本表。"
              if data.get("evaluation_revision") else
              "共享编码器按每套独立部署计入；缓存策略、模型加载及首次推理时间另见完整成本记录。")
    fig.text(.035, .025, footer, fontsize=6.8, color="#526B77")
    return _save(fig, base, qa, scripts, qa_python, axes=list(axes.ravel()), panel_ids=list("abcd"))


def plot(output: str | Path, *, skill_root: str | Path | None = None,
         qa_python: str | Path | None = None) -> list[str]:
    """Export verified, real-result figures; no training or manuscript edits."""
    root = Path(output)
    source = root / "analysis/analysis.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    verification = json.loads((source.parent / "verification.json").read_text(encoding="utf-8"))
    if data.get("status") != "complete" or verification.get("status") != "passed":
        raise ValueError("Analysis verification must pass before plotting")
    if "cst_v1_1" not in data["methods"]:
        raise ValueError("CST v1.1 is required in the comparison figure")
    _font()
    scripts, interpreter = _skill_path(skill_root), _qa_interpreter(qa_python)
    destination = root / "figures"
    qa = destination / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    outputs = _performance(data, destination / "performance", qa, scripts, interpreter)
    if data.get("missing_costs"):
        raise ValueError(f"Performance figure saved, cost figure blocked by missing costs: {data['missing_costs']}")
    outputs += _costs(data, destination / "costs", qa, scripts, interpreter)
    manifest = {"backend": "python", "archetype": "quantitative grid",
                "source_data": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "figure_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "outputs": outputs, "physical_width_mm": 183,
                "claim": "Within-task performance and paired uncertainty under a shared training protocol; no pooled rank or topic-accuracy claim",
                "QA": "Alignment, source, PDF font floor, and rendered collisions are required; final-size human visual inspection remains necessary"}
    (destination / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs
