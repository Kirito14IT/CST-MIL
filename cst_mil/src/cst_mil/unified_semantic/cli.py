"""Independent single-model campaign entry; never rewrites old benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "cst_mil/benchmark/research_followup/20261005/unified_semantic_v1"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    tmp.replace(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def performance_gate(output: Path) -> dict:
    metrics = read(output / "metrics.json")
    baseline = read(ROOT / "cst_mil/benchmark/runs/full677/cst_mil_r0/training_summary.json")
    baseline_dev = next(
        value
        for value in baseline.values()
        if isinstance(value, dict) and value.get("n_samples") == 135 and "pr_auc" in value
    )
    checks = {}
    for category, value in metrics["messages"]["dev"]["pooled"].items():
        checks[category] = {
            "AP": value["AP"],
            "macro_f1": value["macro_f1"],
            "passed": value["AP"] >= 0.70 and value["macro_f1"] >= 0.65,
        }
    value = metrics["conversations"]["dev"]
    for name, new_name in [("pr_auc", "AP"), ("roc_auc", "roc_auc")]:
        checks["conversation_" + new_name] = {
            "new": value[new_name],
            "old": baseline_dev[name],
            "difference": value[new_name] - baseline_dev[name],
            "passed": value[new_name] >= baseline_dev[name] - 0.02,
        }
    result = {
        "passed": all(c["passed"] for c in checks.values()),
        "selection_split": "dev",
        "category_aggregation": "observed pooled; source-wise metrics separately disclosed",
        "checks": checks,
        "baseline_source": "full677/cst_mil_r0/training_summary.json",
    }
    write(output / "performance_gate.json", result)
    return result


def seal_caches(output: Path, *, verify: bool = False) -> dict:
    path = output / "cache_integrity.json"
    if verify:
        value = read(path)
        for name, expected in value.items():
            if sha(output / name) != expected:
                raise ValueError("Sealed embedding content changed: " + name)
        return value
    if path.exists():
        return seal_caches(output, verify=True)
    value = {
        p.relative_to(output).as_posix(): sha(p) for p in [output / "embeddings.npz"] if p.is_file()
    }
    if not value:
        raise ValueError("No completed training embedding cache to seal")
    write(path, value)
    return value


def select_topic_distance(output: Path) -> dict:
    """Choose only from fixed real development cases, never showcase/test scores."""
    cases = read(output / "qa/cases.json")["cases"]
    expected_ids = {c["id"] for c in cases}
    dev_ids = {c["id"] for c in cases if c.get("kind") == "real_dev"}
    if len(expected_ids) != 42 or len(dev_ids) != 30:
        raise ValueError("Selection requires 30 real development and 12 challenge cases")
    automatic = read(output / "qa/automatic_checks.json")
    scores: dict[str, list[float]] = {d: [] for d in ("0.25", "0.35", "0.45")}
    review_hashes = {}
    for name in ("reviewer_a", "reviewer_b"):
        path = output / f"qa/{name}.json"
        review = read(path)
        ids = [c["id"] for c in review.get("cases", [])]
        if len(ids) != 42 or set(ids) != expected_ids:
            raise ValueError("Selection review omitted or duplicated fixed cases")
        if review.get("case_manifest_sha256") != sha(output / "qa/cases.json"):
            raise ValueError("Selection review input fingerprint mismatch")
        for distance in scores:
            if review.get("candidate_sha256", {}).get(distance) != sha(
                output / f"qa/distance_{distance}.json"
            ):
                raise ValueError("Selection review candidate fingerprint mismatch")
            for case in review["cases"]:
                ratings = case.get("ratings", {}).get(distance, {})
                values = [ratings.get(k) for k in ("coherence", "name_support", "assignment")]
                if any(type(v) not in (int, float) or not 1 <= v <= 5 for v in values):
                    raise ValueError("Review dimensions must be numeric scores from 1 to 5")
                if case["id"] in dev_ids:
                    scores[distance].extend(values)
        review_hashes[name] = sha(path)
    means = {d: sum(v) / len(v) for d, v in scores.items()}
    eligible = [d for d in scores if automatic[d]["hard_checks_passed"]]
    if not eligible:
        raise ValueError("No candidate passed grounded/nonempty topic checks")
    priority = {"0.35": 0, "0.25": 1, "0.45": 2}
    selected = sorted(eligible, key=lambda d: (-means[d], priority[d]))[0]
    result = {
        "distance": float(selected),
        "mean_real_dev_ratings": means,
        "eligible_distances": eligible,
        "criterion": (
            "equal mean of two model-assisted reviews, 30 real dev cases, three dimensions"
        ),
        "tie_order": [0.35, 0.25, 0.45],
        "challenge_cases_used_for_selection": False,
        "review_sha256": review_hashes,
        "human_gold": False,
    }
    write(output / "qa/selection.json", result)
    return result


def freeze(output: Path, distance: float) -> dict:
    import joblib

    if distance not in (0.25, 0.35, 0.45):
        raise ValueError("Topic distance must be one of the prespecified development choices")
    if (output / "test_evaluation_state.json").exists():
        raise ValueError("Test already opened; no configuration refreezing")
    gate = performance_gate(output)
    automatic = read(output / "qa/automatic_checks.json")[str(distance)]
    if not automatic["hard_checks_passed"]:
        raise ValueError("Topic grounding/nonempty gate failed")
    reviews = []
    expected_ids = {c["id"] for c in read(output / "qa/cases.json")["cases"]}
    for reviewer in ("reviewer_a", "reviewer_b"):
        path = output / f"qa/{reviewer}.json"
        value = read(path)
        ids = [c["id"] for c in value.get("cases", [])]
        if len(ids) != 42 or len(set(ids)) != 42 or set(ids) != expected_ids:
            raise ValueError("Independent review must include all 42 fixed cases")
        if value.get("case_manifest_sha256") != sha(output / "qa/cases.json"):
            raise ValueError("Review is not bound to the fixed input cases")
        for candidate in ("0.25", "0.35", "0.45"):
            if value.get("candidate_sha256", {}).get(candidate) != sha(
                output / f"qa/distance_{candidate}.json"
            ):
                raise ValueError("Review candidate outputs changed")
            if any(candidate not in c.get("ratings", {}) for c in value["cases"]):
                raise ValueError("Review omitted a prespecified candidate distance")
        reviews.append({"path": str(path.relative_to(output)), "sha256": sha(path)})
    selection = select_topic_distance(output)
    if selection["distance"] != distance:
        raise ValueError("Requested distance differs from independent development selection")
    checkpoint = output / "model.joblib"
    original = output / "model_before_topic_freeze.joblib"
    if not original.exists():
        shutil.copy2(checkpoint, original)
    model = joblib.load(checkpoint)
    model.topic_distance = distance
    joblib.dump(model, checkpoint, compress=3)
    code = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
    value = {
        "model_sha256": sha(checkpoint),
        "deployment_eligible": bool(gate["passed"]),
        "deployment_gate_sha256": sha(output / "performance_gate.json"),
        "topic_distance": distance,
        "thresholds": model.thresholds,
        "conversation_threshold": model.conversation_threshold,
        "runtime_code_fingerprints": code,
        "reviews": reviews,
        "topic_selection_sha256": sha(output / "qa/selection.json"),
        "development_cases_sha256": sha(output / "qa/cases.json"),
        "selection_split": "30 real dev cases; challenges inspected; no test metrics accessed",
        "model_assisted_review_not_human_gold": True,
        "embedding_cache_integrity": seal_caches(output, verify=True),
    }
    write(output / "freeze.json", value)
    state_path = output / "run_state.json"
    if state_path.exists():
        state = read(state_path)
        state["topic_finalization"] = {
            "previous_model_sha256": sha(original),
            "freeze_sha256": sha(output / "freeze.json"),
        }
        state["output_fingerprints"]["model.joblib"] = sha(checkpoint)
        write(state_path, state)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single CST supervised semantic campaign")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    train = commands.add_parser("train")
    train.add_argument("--resume", action="store_true")
    commands.add_parser("qa")
    commands.add_parser("gate")
    commands.add_parser("select-topics")
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--distance", type=float, required=True)
    commands.add_parser("test")
    commands.add_parser("seal-caches")
    commands.add_parser("cost")
    commands.add_parser("report")
    args = parser.parse_args(argv)
    from cst_mil.bench.resources import configure_cpu

    configure_cpu(4)
    output = args.output.resolve()
    if args.command == "prepare":
        from .data import prepare_dataset

        result = prepare_dataset(output)
    elif args.command == "train":
        from .training import train

        if args.resume and (output / "cache_integrity.json").exists():
            seal_caches(output, verify=True)
        result = train(output, output / "prepared", resume=args.resume, defer_test=True)
        seal_caches(output)
    elif args.command == "qa":
        from .qa import run_topic_qa

        result = run_topic_qa(output)
    elif args.command == "gate":
        result = performance_gate(output)
    elif args.command == "select-topics":
        result = select_topic_distance(output)
    elif args.command == "freeze":
        result = freeze(output, args.distance)
    elif args.command == "seal-caches":
        result = seal_caches(output)
    elif args.command == "cost":
        from .cost import measure

        result = measure(output)
    elif args.command == "report":
        from .report import build_report

        result = {"report_path": str(build_report(output))}
    else:
        from .training import evaluate_test

        seal_caches(output, verify=True)
        result = evaluate_test(output, output / "prepared", output / "freeze.json")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
