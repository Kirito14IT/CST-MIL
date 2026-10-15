"""Shared canonical raw representation banks, distinct from measured inference."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from cst_mil.research_followup.full_joint_comparison.model import canonical_batches
from cst_mil.research_followup.full_joint_comparison.raw_replay_v2 import select_warmup
from cst_mil.unified_semantic_v1_1.encoder import LocalSemanticEncoder
from .common import (OUTPUT, TreeMonitor, cached_train_embeddings, digest, guard, identity,
                     load_data, read, save, verify_hashes)


def new_encoder(output: Path = OUTPUT):
    metadata = read(output / "protocol.json")["encoder"]
    return LocalSemanticEncoder(metadata["encoder_path"], metadata["fingerprint"], threads=4)


def fixed_warmup(encoder, records: list[dict]) -> dict:
    selected = select_warmup(records)
    start = time.perf_counter()
    encoder.encode([r["text"] for r in selected], batch_size=32)
    return {"record_ids": [r["record_id"] for r in selected], "split": "train",
            "seconds": time.perf_counter() - start,
            "input_identity": identity([{ "record_id": r["record_id"], "text_sha256": identity(r["text"])} for r in selected])}


def encode_raw_records(encoder, records: list[dict], *, progress=None) -> np.ndarray:
    result = np.empty((len(records), 384), dtype=np.float32)
    units = canonical_batches(records)
    for index, indices in enumerate(units):
        result[indices] = encoder.encode([records[i]["text"] for i in indices], batch_size=32)
        if progress is not None:
            progress(index, len(units), indices)
    if not np.isfinite(result).all():
        raise ValueError("Raw embeddings are nonfinite")
    return result


def encode_bank(split: str, output: Path = OUTPUT, *, resume: bool = True) -> dict:
    if split not in ("dev", "test"):
        raise ValueError("Only DEV and TEST raw banks are generated")
    protocol = guard(output, execution=False)
    if split == "test":
        if not (output / "selection_freeze.json").exists():
            raise RuntimeError("TEST bank cannot open before validation selection freeze")
        selection = read(output / "selection_freeze.json")
        verify_hashes(selection["input_hashes"])
        if identity({k: v for k, v in selection.items() if k != "identity"}) != selection["identity"]:
            raise RuntimeError("Selection fingerprint mismatch")
    folder = output / "raw_embeddings" / split
    complete = folder / "complete.json"
    if complete.exists():
        state = read(complete)
        if not resume or state["protocol_identity"] != protocol["identity"]:
            raise RuntimeError("Existing raw bank has a different protocol")
        verify_hashes(state["artifacts"])
        return state
    all_records, _ = load_data()
    records = [r for r in all_records if r["split"] == split]
    binding = {"protocol_identity": protocol["identity"], "split": split,
               "record_ids": [r["record_id"] for r in records],
               "text_identities": [identity(r["text"]) for r in records],
               "canonical_batches": [[records[i]["record_id"] for i in ids] for ids in canonical_batches(records)]}
    if (folder / "binding.json").exists() and read(folder / "binding.json") != binding:
        raise RuntimeError("Raw bank binding changed")
    save(folder / "binding.json", binding)
    encoder = new_encoder(output)
    with TreeMonitor() as monitor:
        warm = fixed_warmup(encoder, all_records)
        started = time.perf_counter()
        def progress(index, total, indices):
            if index % 25 == 0 or index + 1 == total:
                print(f"RAW {split} {index+1}/{total} canonical units", flush=True)
        values = encode_raw_records(encoder, records, progress=progress)
        seconds = time.perf_counter() - started
    path = folder / "embeddings.npy"
    temporary = folder / "embeddings.pending.npy"
    np.save(temporary, values, allow_pickle=False)
    temporary.replace(path)
    state = {"status": "complete", "protocol_identity": protocol["identity"], "split": split,
             "records": len(records), "seconds": seconds, "warmup": warm,
             "resources": monitor.result(), "representation_cache_not_cost_measurement": True,
             "artifacts": {str(path): digest(path), str(folder / "binding.json"): digest(folder / "binding.json")}}
    save(complete, state)
    return state


def embeddings_for(records: list[dict], output: Path = OUTPUT) -> np.ndarray:
    result = cached_train_embeddings(records)
    for split in ("dev", "test"):
        requested = [i for i, r in enumerate(records) if r["split"] == split]
        if not requested:
            continue
        folder = output / "raw_embeddings" / split
        state, binding = read(folder / "complete.json"), read(folder / "binding.json")
        verify_hashes(state["artifacts"])
        positions = {rid: i for i, rid in enumerate(binding["record_ids"])}
        values = np.load(folder / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        for i in requested:
            j = positions[records[i]["record_id"]]
            if identity(records[i]["text"]) != binding["text_identities"][j]:
                raise RuntimeError("Raw bank text identity mismatch")
            result[i] = values[j]
    return result
