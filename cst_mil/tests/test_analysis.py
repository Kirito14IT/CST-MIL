from __future__ import annotations

import pandas as pd
import pytest

from cst_mil.analysis import compute_binary_metrics, compute_evidence_metrics, stratified_bootstrap


def test_binary_metrics_and_bootstrap_are_reproducible() -> None:
    predictions = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "prediction": [0, 1, 1, 1],
            "risk_probability": [0.1, 0.7, 0.8, 0.9],
        }
    )
    result = compute_binary_metrics(predictions).metrics
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["recall_cb"] == pytest.approx(1.0)
    first = stratified_bootstrap(predictions, n_samples=20, seed=42)
    second = stratified_bootstrap(predictions, n_samples=20, seed=42)
    assert first == second


def test_evidence_metrics_use_ranked_comment_labels() -> None:
    evidence = pd.DataFrame(
        {
            "post_id": ["a", "a", "b", "b"],
            "rank": [1, 2, 1, 4],
            "comment_label": ["CB", "Non-CB", "Non-CB", "CB"],
        }
    )
    metrics = compute_evidence_metrics(evidence, top_k=3)
    assert metrics["n_positive_sessions"] == 2
    assert metrics["precision_at_k"] == pytest.approx(1 / 3)
    assert metrics["hit_at_k"] == pytest.approx(0.5)


def test_metrics_reject_single_class() -> None:
    predictions = pd.DataFrame(
        {"label": [1, 1], "prediction": [1, 0], "risk_probability": [0.8, 0.2]}
    )
    with pytest.raises(ValueError, match="both labels"):
        compute_binary_metrics(predictions)
