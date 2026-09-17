"""Evaluation, uncertainty summaries, figures, and evidence-first reports."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Computed metrics plus the unit of analysis."""

    metrics: dict[str, float | int]
    unit: str = "session"


def _require_prediction_columns(predictions: pd.DataFrame) -> None:
    required = {"label", "prediction", "risk_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    if predictions.empty:
        raise ValueError("Prediction table is empty")


def compute_binary_metrics(predictions: pd.DataFrame) -> EvaluationResult:
    """Compute session-level binary classification metrics."""

    _require_prediction_columns(predictions)
    y_true = predictions["label"].astype(int).to_numpy()
    y_pred = predictions["prediction"].astype(int).to_numpy()
    y_prob = predictions["risk_probability"].astype(float).to_numpy()
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Binary evaluation requires both labels 0 and 1")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "n_sessions": int(len(predictions)),
        "n_non_cb": int((y_true == 0).sum()),
        "n_cb": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_cb": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_cb": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_cb": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return EvaluationResult(metrics=metrics)


def compute_evidence_metrics(evidence: pd.DataFrame, top_k: int = 3) -> dict[str, float | int]:
    """Evaluate ranked comments using comment labels that were not used for training."""

    required = {"post_id", "rank", "comment_label"}
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise ValueError(f"Evidence table is missing columns: {missing}")
    ranked = evidence.loc[evidence["rank"].astype(int) <= top_k].copy()
    labels = ranked["comment_label"]
    numeric = pd.to_numeric(labels, errors="coerce")
    ranked["is_cb_comment"] = np.where(
        numeric.notna(),
        numeric.eq(1),
        labels.astype(str).str.strip().str.upper().eq("CB"),
    )
    per_session = ranked.groupby("post_id", sort=False)["is_cb_comment"]
    if ranked.empty:
        return {
            "evidence_top_k": top_k,
            "n_positive_sessions": 0,
            "precision_at_k": 0.0,
            "hit_at_k": 0.0,
        }
    return {
        "evidence_top_k": int(top_k),
        "n_positive_sessions": int(ranked["post_id"].nunique()),
        "precision_at_k": float(ranked["is_cb_comment"].mean()),
        "hit_at_k": float(per_session.any().mean()),
    }


def stratified_bootstrap(
    predictions: pd.DataFrame,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float | int]]:
    """Bootstrap fixed test predictions within each class, without retraining."""

    _require_prediction_columns(predictions)
    rng = np.random.default_rng(seed)
    class_frames = [predictions.loc[predictions["label"].astype(int) == label] for label in (0, 1)]
    if any(frame.empty for frame in class_frames):
        raise ValueError("Stratified bootstrap requires both classes")

    metric_functions: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
        "macro_f1": lambda y, pred, prob: f1_score(y, pred, average="macro", zero_division=0),
        "recall_cb": lambda y, pred, prob: recall_score(y, pred, zero_division=0),
        "pr_auc": lambda y, pred, prob: average_precision_score(y, prob),
    }
    samples: dict[str, list[float]] = {name: [] for name in metric_functions}
    for _ in range(n_samples):
        resampled = []
        for frame in class_frames:
            indexes = rng.integers(0, len(frame), size=len(frame))
            resampled.append(frame.iloc[indexes])
        draw = pd.concat(resampled, ignore_index=True)
        y = draw["label"].astype(int).to_numpy()
        pred = draw["prediction"].astype(int).to_numpy()
        prob = draw["risk_probability"].astype(float).to_numpy()
        for name, function in metric_functions.items():
            samples[name].append(float(function(y, pred, prob)))

    intervals: dict[str, dict[str, float | int]] = {}
    point = compute_binary_metrics(predictions).metrics
    for name, values in samples.items():
        low, high = np.percentile(values, [2.5, 97.5])
        intervals[name] = {
            "point": float(point[name]),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "bootstrap_samples": int(n_samples),
            "seed": int(seed),
        }
    return intervals


def _save_figure(fig: plt.Figure, base_path: Path) -> list[str]:
    saved: list[str] = []
    for suffix, kwargs in ((".png", {"dpi": 600}), (".pdf", {}), (".svg", {})):
        path = base_path.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        saved.append(path.name)
    plt.close(fig)
    return saved


def plot_confusion_matrix(predictions: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Generate a colorblind-safe annotated confusion matrix."""

    _require_prediction_columns(predictions)
    figures_dir.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(
        predictions["label"].astype(int), predictions["prediction"].astype(int), labels=[0, 1]
    )
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )
    ax.set_xticks([0, 1], labels=["Non-CB", "CB"])
    ax.set_yticks([0, 1], labels=["Non-CB", "CB"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Sessions")
    return _save_figure(fig, figures_dir / "figure-01-confusion-matrix")


def plot_precision_recall(predictions: pd.DataFrame, figures_dir: Path) -> list[str]:
    """Plot the fixed test-set precision-recall curve."""

    _require_prediction_columns(predictions)
    figures_dir.mkdir(parents=True, exist_ok=True)
    y = predictions["label"].astype(int).to_numpy()
    probability = predictions["risk_probability"].astype(float).to_numpy()
    precision, recall, _ = precision_recall_curve(y, probability)
    ap = average_precision_score(y, probability)
    prevalence = float(y.mean())
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    ax.plot(recall, precision, color="#0072B2", linewidth=2, label=f"CST-MIL (AP={ap:.3f})")
    ax.axhline(
        prevalence,
        color="#E69F00",
        linestyle="--",
        linewidth=1.5,
        label=f"Prevalence={prevalence:.2f}",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=False)
    return _save_figure(fig, figures_dir / "figure-02-precision-recall")


def plot_method_pipeline(figures_dir: Path) -> list[str]:
    """Create a compact method flow diagram from deterministic vector shapes."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.2, "SCCD session\npost + replies", "#F0E442"),
        (2.15, "Char 2–5 grams\n3-round MIL", "#56B4E9"),
        (4.1, "TF-IDF + 8-topic\nMiniBatchNMF", "#009E73"),
        (6.05, "Risk/topic/reply\naggregation", "#CC79A7"),
        (8.0, "Linear session risk\n+ top-3 evidence", "#E69F00"),
    ]
    for x, label, color in boxes:
        patch = FancyBboxPatch(
            (x, 0.9),
            1.65,
            1.15,
            boxstyle="round,pad=0.08",
            facecolor=color,
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(patch)
        ax.text(x + 0.825, 1.475, label, ha="center", va="center", fontsize=9)
    for left in (1.85, 3.8, 5.75, 7.7):
        arrow = FancyArrowPatch(
            (left, 1.48),
            (left + 0.28, 1.48),
            arrowstyle="->",
            mutation_scale=12,
        )
        ax.add_patch(arrow)
    return _save_figure(fig, figures_dir / "figure-03-method-pipeline")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_analysis_bundle(
    run_dir: Path,
    predictions: pd.DataFrame,
    metrics: dict[str, float | int],
    bootstrap: dict[str, dict[str, float | int]],
    evidence_metrics: dict[str, float | int] | None = None,
) -> None:
    """Write the strict analysis bundle required for evidence-first reporting."""

    analysis_dir = run_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    plot_confusion_matrix(predictions, figures_dir)
    plot_precision_recall(predictions, figures_dir)
    plot_method_pipeline(figures_dir)

    evidence_metrics = evidence_metrics or {}
    metric_rows = "\n".join(
        f"| {name} | {value:.4f} |" if isinstance(value, float) else f"| {name} | {value} |"
        for name, value in metrics.items()
    )
    analysis_report = f"""# CST-MIL pilot analysis report

## Analysis question

Can a CPU-only, weakly supervised CST-MIL prototype identify CB sessions and
localize risky comments on one fixed, balanced SCCD pilot split?

## Evidence inventory

- Unit of analysis: complete `post_id` session.
- Formal training runs: 1 (seed 42).
- Test sessions: {metrics['n_sessions']} ({metrics['n_cb']} CB, {metrics['n_non_cb']} Non-CB).
- Comparison family: CST-MIL only; no baseline result is available in this run.

## Exact numeric summary

| Metric | Value |
|---|---:|
{metric_rows}

## Key findings

- The fixed test split produced macro-F1 {metrics['macro_f1']:.4f},
  CB recall {metrics['recall_cb']:.4f}, and PR-AUC {metrics['pr_auc']:.4f}.
- The uncertainty intervals are conditional on resampling this fixed test
  prediction table; they do not represent variability across model retraining
  or alternative data splits.
- Error counts are TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}.

## Claim candidates

- Claim:
  - Source evidence: `predictions.csv`, `metrics.json`, and Figures 01–02.
  - Allowed wording: CST-MIL completed one CPU-only SCCD pilot and produced
    measurable session-risk and evidence-localization outputs.
  - Forbidden stronger wording: CST-MIL significantly outperforms LDA, BTM, GSDMM, NMF, or BERTopic.
  - Uncertainty: One 300-session sample, one split, one training seed, and no
    adapted baseline result.
  - Next check: Run adapted baselines and multiple seeds on all 677 sessions.
  - Decision: keep

## Main caveats

- SCCD is a Weibo post–comment tree rather than a persistent IM group chat.
- Static SCCD cannot validate rapid concept evolution or online adaptation.
- The bootstrap interval quantifies test-sample uncertainty only.
"""
    (analysis_dir / "analysis-report.md").write_text(analysis_report, encoding="utf-8")

    interval_rows = "\n".join(
        f"| {name} | {values['point']:.4f} | {values['ci95_low']:.4f} | {values['ci95_high']:.4f} |"
        for name, values in bootstrap.items()
    )
    stats_appendix = f"""# Statistical appendix

## Design

- One fixed stratified split; one formal model fit with seed 42.
- No method contrast, p-value, effect size, or multiple-comparison procedure is applicable.
- The test set is the unit used for final prediction evaluation; each row is one complete session.

## Conditional uncertainty

Predictions were resampled 1,000 times with replacement within each true class.
Percentile 2.5% and 97.5% bounds are reported. These intervals are conditional
on the fitted model and fixed test set.

| Metric | Point | 95% low | 95% high |
|---|---:|---:|---:|
{interval_rows}

## Evidence localization

```json
{json.dumps(evidence_metrics, ensure_ascii=False, indent=2)}
```

## Blocked inference

- No variance across training seeds is available.
- No adapted baseline result is available.
- Therefore no statistical superiority or stability claim is allowed.
"""
    (analysis_dir / "stats-appendix.md").write_text(stats_appendix, encoding="utf-8")

    figure_catalog = f"""# Figure catalog

## Figure 01 — Confusion matrix

- Files: `figures/figure-01-confusion-matrix.png|pdf|svg`
- Purpose: show the four exact session-level decision outcomes.
- Data source: `predictions.csv`, n={metrics['n_sessions']} test sessions.
- Key observation: TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}.
- Interpretation: reveals whether the fixed threshold trades CB misses for false alarms.
- Caveat: one split only; no uncertainty bar is meaningful for the count matrix.

## Figure 02 — Precision-recall curve

- Files: `figures/figure-02-precision-recall.png|pdf|svg`
- Purpose: show precision across CB recall thresholds.
- Data source: `predictions.csv`, n={metrics['n_sessions']} test sessions.
- Key observation: average precision is {metrics['pr_auc']:.4f}; dashed line is CB prevalence.
- Interpretation: describes ranking quality without selecting extra thresholds on the test set.
- Caveat: the curve is conditional on one trained model and one test sample.

## Figure 03 — Method pipeline

- Files: `figures/figure-03-method-pipeline.png|pdf|svg`
- Purpose: document the deterministic data and model flow.
- Data source: `configs/pilot.yaml` and implementation modules.
- Interpretation: clarifies how character, topic, reply, and session-risk components connect.
- Caveat: architectural capability is not evidence of temporal adaptation performance.
"""
    (analysis_dir / "figure-catalog.md").write_text(figure_catalog, encoding="utf-8")
