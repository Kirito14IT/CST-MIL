"""Independent campaign identities and read-only access to frozen inputs."""
from __future__ import annotations

import os
from pathlib import Path
from collections import Counter

from cst_mil.research_followup.full_joint_comparison.common import (
    TreeMonitor, campaign_lock, digest, identity, read, read_jsonl, save,
    source_hashes, verify_hashes, write_jsonl,
)

PROJECT = Path(__file__).resolve().parents[4]
MODULE = Path(__file__).resolve().parent
BASE = PROJECT / "benchmark/research_followup/20261014/full_joint_comparison"
REVISION = BASE / "evaluation_revisions/raw_replay_v2"
OUTPUT = PROJECT / "benchmark/research_followup/20261015/semantic_patent_followup"
REPORT = PROJECT / "reports/专利交底书/20261015_语义方案"
OLD_CASES = PROJECT / "benchmark/research_followup/20261005/unified_semantic_v1_1/qa/cases.json"
SEEDS = (42, 43, 44)
CATEGORIES = ("attack", "hate", "spam_scam")
ABLATIONS = ("no_char", "no_message_semantics", "no_session_semantics", "no_session_reply", "no_session_risk", "no_source_balance")
MAIN_TASKS = ("sccd.attack", "conda.attack", "hatebr.attack", "toxicn.attack", "toxicn.hate", "scamspot.spam_scam", "sccd.conversation")


def configure(threads: int = 4) -> None:
    if not 1 <= threads <= 4:
        raise ValueError("This protocol permits one through four CPU threads")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = str(threads)
    from cst_mil.bench.resources import configure_cpu
    configure_cpu(threads)


def load_data(splits=("train", "dev", "test")) -> tuple[list[dict], list[dict]]:
    records = [r for r in read_jsonl(BASE / "prepared/records.jsonl") if r["split"] in splits]
    sessions = [r for r in read_jsonl(BASE / "prepared/sessions.jsonl") if r["split"] in splits]
    return records, sessions


def original_code_hashes() -> dict[str, str]:
    paths = [p for p in (PROJECT / "src").rglob("*.py")
             if not p.is_relative_to(MODULE) and "__pycache__" not in p.parts]
    return source_hashes(paths)


def input_hashes() -> dict[str, str]:
    paths = [BASE / "prepared" / name for name in ("records.jsonl", "sessions.jsonl", "manifest.json")]
    paths += [BASE / "experiment.json", OLD_CASES, REVISION / "completion_audit.json"]
    for method in ("cst_v1_1", "minilm_lr"):
        for seed in SEEDS:
            paths += [BASE / "runs" / method / f"seed_{seed}" / name
                      for name in ("model.joblib", "train_state.json", "fit_details.json")]
            paths += [REVISION / "runs" / method / f"seed_{seed}" / name
                      for name in ("run_manifest.json", "predictions_message_dev.jsonl", "predictions_session_dev.jsonl")]
    return source_hashes(paths)


def freeze_protocol(output: Path = OUTPUT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "protocol.json"
    if target.exists():
        protocol = read(target)
        verify_hashes(protocol["input_hashes"])
        verify_hashes(protocol["original_code_hashes"])
        return protocol
    experiment = read(BASE / "experiment.json")
    verify_hashes(experiment["code_hashes"])
    verify_hashes(experiment["encoder"]["asset_hashes"])
    if read(REVISION / "completion_audit.json")["status"] != "passed":
        raise RuntimeError("Original experiment audit is not accepted")
    value = {
        "protocol": "semantic-patent-fixed-followup-20261015",
        "base_experiment_identity": identity(experiment), "base_output": str(BASE),
        "seeds": list(SEEDS), "threads": 4, "categories": list(CATEGORIES),
        "candidate_architectures": ["context_a", "context_b"], "ablations": list(ABLATIONS),
        "maximum_new_configuration_seed_runs": 24,
        "selection": {"primary": "three-seed mean DEV SCCD attack AP",
                      "minimum_primary_gain": .01, "maximum_other_AP_drop": .01,
                      "maximum_SCCD_session_AP_AUC_drop": .01,
                      "maximum_SCCD_attack_macro_F1_drop": .01,
                      "tie_tolerance": 1e-12, "tie_order": ["context_a", "context_b"],
                      "fallback": "frozen full architecture", "test_selection_forbidden": True},
        "thresholds": "DEV macro-F1 maximum; exact lower-threshold tie rule",
        "lr": {"C": 1., "solver": "liblinear", "max_iter": 1000},
        "topic_distance": .35, "short_reply_nonspace_character_limit": 12,
        "review": {"old_regression_cases": 42, "new_dev_cases": 30,
                   "new_cases_per_session_label_length_cell": 5,
                   "reviewers": 2, "human_gold": False, "test_first_unseen_claim": False},
        "encoder": experiment["encoder"], "input_hashes": input_hashes(),
        "original_code_hashes": original_code_hashes(),
        "representation_policy": "TRAIN original audited cache; DEV/TEST newly sealed fixed-TRAIN-warmup canonical raw bank shared by all variants",
        "cost_policy": "risk raw predictions and topic interpretation separately timed; no representation cache for raw inference cost",
        "test_exposure_disclosed": True, "paid_API_calls": 0,
    }
    value["identity"] = identity(value)
    save(target, value)
    return value


def freeze_execution(output: Path = OUTPUT) -> dict:
    protocol = freeze_protocol(output)
    target = output / "execution_freeze.json"
    risk_names = ("__init__.py", "__main__.py", "common.py", "data.py", "model.py", "execution.py", "representations.py")
    hashes = source_hashes(MODULE / name for name in risk_names)
    value = {"protocol_identity": protocol["identity"], "code_hashes": hashes}
    value["identity"] = identity(value)
    if target.exists():
        old = read(target)
        if old != value:
            raise RuntimeError("Frozen follow-up source changed; preserve evidence and use a revision")
        return old
    save(target, value)
    return value


def guard(output: Path = OUTPUT, *, execution: bool = True) -> dict:
    protocol = read(output / "protocol.json")
    verify_hashes(protocol["input_hashes"])
    verify_hashes(protocol["original_code_hashes"])
    if execution:
        frozen = read(output / "execution_freeze.json")
        verify_hashes(frozen["code_hashes"])
    return protocol


def preservation(output: Path = OUTPUT, *, verify: bool = False) -> dict:
    target = output / "preservation_before.json"
    if target.exists():
        hashes = read(target)
        if verify:
            verify_hashes(hashes)
            result = {"status": "passed", "files": len(hashes), "unchanged": True}
            save(output / "preservation_after.json", result)
            return result
        return {"files": len(hashes), "reused": True}
    if verify:
        raise RuntimeError("No initial protection snapshot")
    roots = [PROJECT / "data", PROJECT / "benchmark/data", PROJECT / "benchmark/runs/full677",
             PROJECT / "benchmark/comparison/full677", PROJECT / "reports"]
    roots += [p for p in (PROJECT / "benchmark/research_followup").iterdir()
              if p.is_dir() and p.name != "20261015"]
    def eligible(p: Path) -> bool:
        return (p.is_file() and not p.is_relative_to(REPORT)
                and not any(part.startswith(("_pytest", "pytest", "test_tmp"))
                            or part in ("__pycache__", ".venv", "node_modules", ".cache", "tools") for part in p.parts)
                and p.suffix not in (".pyc", ".lock", ".tmp", ".pending", ".log", ".sqlite-shm", ".sqlite-wal"))
    paths = [p for root in roots if root.exists() for p in root.rglob("*") if eligible(p)]
    paths += [Path(p) for p in original_code_hashes()]
    paths += [PROJECT.parent / "计算机项目综合开发创新实践/app/config/deployment.json"]
    hashes = source_hashes(paths)
    save(target, hashes)
    return {"files": len(hashes), "snapshot_created": True}


def cached_train_embeddings(records: list[dict]):
    import numpy as np
    folder = BASE / "cache/train"
    state, binding = read(folder / "complete.json"), read(folder / "binding.json")
    verify_hashes(state["artifacts"])
    if state["record_ids"] != binding["record_ids"]:
        raise RuntimeError("Frozen embedding IDs mismatch")
    if binding["prepared_manifest_sha256"] != digest(BASE / "prepared/manifest.json"):
        raise RuntimeError("Embedding corpus fingerprint mismatch")
    values = np.load(folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    positions = {rid: i for i, rid in enumerate(binding["record_ids"])}
    matrix = np.zeros((len(records), 384), dtype=np.float32)
    for i, record in enumerate(records):
        if record["split"] != "train":
            continue
        j = positions[record["record_id"]]
        if identity(record["text"]) != binding["text_sha256"][j]:
            raise RuntimeError("Training embedding text mismatch")
        matrix[i] = values[j]
    return matrix


def data_profile(records: list[dict], sessions: list[dict]) -> dict:
    return {"record_count": len(records), "source_split": dict(Counter(f"{r['source']}.{r['split']}" for r in records)),
            "session_split": dict(Counter(r["split"] for r in sessions)),
            "fit_eligible_train": sum(r["split"] == "train" and r.get("fit_eligible", True) for r in records),
            "unknown_all_categories": sum(not any(r["observed_mask"]) for r in records)}
