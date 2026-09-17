"""Bounded predeclared pilot sequence; held-out evaluation follows frozen selection."""

from __future__ import annotations

import copy
import subprocess
import sys

from .common import read_json, write_json
from .dataset import BENCHMARK, PROJECT
from .evaluation import replacement_gate, stopping_gate
from .runner import DEFAULT_CONFIG, METHODS, functional_checks_passed, run_path, utc_now


def _invoke(method: str, revision: str | None, stage: str):
    log = BENCHMARK / "logs" / f"pilot300_{method}_{revision or 'baseline'}_{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        "-u",
        "-m",
        "cst_mil.bench.cli",
        "run",
        "--suite",
        "pilot300",
        "--method",
        method,
        "--resume",
        "--stage",
        stage,
    ]
    if revision:
        args += ["--revision", revision]
    print(f"PILOT {method} {revision or ''} {stage}: {log}", flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\nSTART {utc_now()}\n")
        handle.flush()
        result = subprocess.run(
            args,
            cwd=PROJECT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    if result.returncode:
        raise RuntimeError(f"Pilot {method}/{revision} {stage} failed; see {log}")
    path = run_path("pilot300", method, revision)
    if stage == "train":
        dev = read_json(path / "dev_metrics.json")
        print(
            f"DEV {method}/{revision}: F1={dev['risk']['macro_f1']:.4f}, "
            f"NPMI={dev['topics']['npmi']:.4f}, coverage={dev['topics']['coverage']:.4f}",
            flush=True,
        )


def run_pilot_campaign() -> dict:
    if not functional_checks_passed():
        raise ValueError("Run and record functional tests before formal pilot")
    selection_path = BENCHMARK / "selection.json"
    ledger_path = BENCHMARK / "campaign.json"
    ledger = (
        read_json(ledger_path)
        if ledger_path.exists()
        else {
            "started_at": utc_now(),
            "seed": 42,
            "formal_sessions": 300,
            "max_additional_revisions": 3,
            "selection_split": "dev",
            "history": [],
        }
    )
    if not selection_path.exists():
        references = []
        for method in METHODS[:-1]:
            _invoke(method, None, "train")
            path = run_path("pilot300", method)
            dev = read_json(path / "dev_metrics.json")
            timing = read_json(path / "training_summary.json").get("timings", {})
            references.append((method, dev, timing.get("dev_inference_seconds", float("inf"))))
        references.sort(
            key=lambda item: (
                -item[1]["risk"]["macro_f1"],
                -item[1]["topics"]["npmi"],
                item[2],
                item[0],
            )
        )
        baseline, baseline_metrics, _ = references[0]
        ledger["baseline_reference"] = baseline
        _invoke("cst_mil", "r0", "train")
        current_revision = "r0"
        current_config = copy.deepcopy(DEFAULT_CONFIG)
        current_metrics = read_json(run_path("pilot300", "cst_mil", "r0") / "dev_metrics.json")
        history = [
            {
                "revision": "r0",
                "base_revision": None,
                "accepted": True,
                "metrics": current_metrics,
                "meets_stop_target": stopping_gate(current_metrics, baseline_metrics),
            }
        ]
        ledger["history"] = history
        write_json(ledger_path, ledger)
        revisions = [("r1", "word_features"), ("r2", "risk_weighted_topics"), ("r3", "multiscale")]
        for revision, flag in revisions:
            if stopping_gate(current_metrics, baseline_metrics):
                break
            config = {**copy.deepcopy(current_config), flag: True}
            spec = {
                "revision": revision,
                "base_revision": current_revision,
                "change": flag,
                "config": config,
                "selection_data": "dev only",
            }
            spec_path = BENCHMARK / "revisions" / f"{revision}.json"
            if spec_path.exists() and read_json(spec_path) != spec:
                raise ValueError("Preregistered revision differs; refusing overwrite")
            if not spec_path.exists():
                write_json(spec_path, spec)
            _invoke("cst_mil", revision, "train")
            candidate = read_json(run_path("pilot300", "cst_mil", revision) / "dev_metrics.json")
            accepted = replacement_gate(candidate, current_metrics)
            history.append(
                {
                    "revision": revision,
                    "base_revision": current_revision,
                    "change": flag,
                    "accepted": accepted,
                    "metrics": candidate,
                    "meets_stop_target": stopping_gate(candidate, baseline_metrics),
                }
            )
            if accepted:
                current_revision, current_config, current_metrics = revision, config, candidate
            ledger["history"] = history
            write_json(ledger_path, ledger)
        selected = {
            "revision": current_revision,
            "config": current_config,
            "baseline_reference": baseline,
            "history": history,
            "selected_from": "pilot300 dev only",
            "frozen_at": utc_now(),
            "stop_target_met": stopping_gate(current_metrics, baseline_metrics),
            "test_used_for_selection": False,
        }
        write_json(selection_path, selected)
        ledger["status"] = "selection_frozen"
        write_json(ledger_path, ledger)
        print(
            f"FROZEN CST {current_revision}; target_met={selected['stop_target_met']}", flush=True
        )
    selected = read_json(selection_path)
    for method in METHODS[:-1]:
        _invoke(method, None, "all")
    for item in selected["history"]:
        _invoke("cst_mil", item["revision"], "all")
    ledger.update(status="pilot_complete", completed_at=utc_now(), selection=selected["revision"])
    write_json(ledger_path, ledger)
    from .reporting import create_comparison

    result = create_comparison("pilot300")
    return {"campaign": ledger, "comparison": result, "full677_trained": False}
