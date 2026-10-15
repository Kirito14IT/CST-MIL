"""Isolated CPU timing, separating model load, first inference and residency."""

from __future__ import annotations

import json
import platform
import threading
import time
from importlib.metadata import distribution
from pathlib import Path

import joblib
import numpy as np
import psutil

from .encoder import file_sha256
from .model import predict_session


def dependency_assets() -> list[dict]:
    """Installed inference stack, separate from checkpoint and encoder snapshots."""
    result = []
    for name in (
        "torch",
        "sentence-transformers",
        "transformers",
        "tokenizers",
        "safetensors",
        "numpy",
        "scipy",
        "scikit-learn",
        "jieba",
        "joblib",
        "psutil",
    ):
        package = distribution(name)
        files = [Path(package.locate_file(p)) for p in package.files or []]
        result.append(
            {
                "distribution": name,
                "version": package.version,
                "installed_file_bytes": sum(p.stat().st_size for p in files if p.is_file()),
            }
        )
    return result


def measure(output: Path) -> dict:
    cases = json.loads((output / "qa/cases.json").read_text(encoding="utf-8"))["cases"]
    # Report the exact set; sampled real conversations are not a full test benchmark.
    selected = cases
    peak = [0]
    stop = threading.Event()
    process = psutil.Process()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.wait(0.1)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    model = joblib.load(output / "model.joblib")
    loaded = time.perf_counter() - started
    try:
        first_start = time.perf_counter()
        predict_session(model, selected[0]["session"], selected[0]["messages"])
        first_seconds = time.perf_counter() - first_start
        rows = []
        for repetition in range(3):
            for case in selected:
                before = time.perf_counter()
                result = predict_session(model, case["session"], case["messages"])
                elapsed = time.perf_counter() - before
                rows.append(
                    {
                        "case_id": case["id"],
                        "kind": case["kind"],
                        "repetition": repetition + 1,
                        "messages": len(case["messages"]),
                        "seconds": elapsed,
                        "topics": len(result["topics"]),
                    }
                )
        values = [r["seconds"] for r in rows]
        training = json.loads((output / "training_summary.json").read_text(encoding="utf-8"))
        result = {
            "model_version": model.model_version,
            "model_sha256": file_sha256(output / "model.joblib"),
            "n_cases": len(selected),
            "repetitions": 3,
            "model_load_seconds": loaded,
            "first_inference_seconds": first_seconds,
            "first_case_id": selected[0]["id"],
            "warm_total_seconds": sum(values),
            "warm_median_seconds": float(np.median(values)),
            "warm_p95_seconds": float(np.quantile(values, 0.95)),
            "peak_process_rss_mib": peak[0] / 1024**2,
            "training_seconds": training["training_seconds"],
            "training_peak_rss_mib": training["peak_process_rss_bytes"] / 1024**2,
            "checkpoint_mib": (output / "model.joblib").stat().st_size / 1024**2,
            "encoder_assets_mib": training["encoder_asset_bytes"] / 1024**2,
            "GPU_required": False,
            "paid_API_calls": 0,
            "deployment_dependencies": dependency_assets(),
            "rows": rows,
            "hardware": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "processor_details": json.loads(
                    (output / "hardware.json").read_text(encoding="utf-8-sig")
                )
                if (output / "hardware.json").is_file()
                else None,
                "logical_cpus": psutil.cpu_count(),
                "affinity": process.cpu_affinity(),
                "ram_mib": psutil.virtual_memory().total / 1024**2,
            },
            "notes": [
                "编码器进程内驻留；3次重复均重新编码正文，未复用磁盘句向量缓存。",
                "计时集是30个真实开发会话＋12个编写挑战，不是完整136测试集。",
                "模型加载与首次编码分别计时；Python解释器/顶层导入时间不在model_load_seconds中。",
                "训练计时为实际开发执行墙钟时间，独立推理计时期间不运行其他本轮训练。",
                "依赖大小为列出的已安装distribution文件，非最小打包大小或全部传递依赖。",
            ],
        }
        for kind in ("real_dev", "authored_challenge"):
            times = [r["seconds"] for r in rows if r["kind"] == kind]
            result[kind] = {
                "n_timings": len(times),
                "median_seconds": float(np.median(times)),
                "p95_seconds": float(np.quantile(times, 0.95)),
                "total_seconds": sum(times),
            }
        path = output / "cost.json"
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        return {key: value for key, value in result.items() if key != "rows"}
    finally:
        stop.set()
        thread.join()
