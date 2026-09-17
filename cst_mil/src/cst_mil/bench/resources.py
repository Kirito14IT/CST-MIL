"""Enforce the benchmark CPU budget before importing numerical libraries.

Environment thread hints are insufficient for nested native thread pools. The
process affinity mask is the operating-system cap, inherited by native child
processes. This module deliberately imports no NumPy, SciPy, Torch or tokenizers.
"""

from __future__ import annotations

import os

import psutil

_ORIGINAL_ELIGIBLE: dict[int, list[int]] = {}


def configure_cpu(threads: int = 4) -> dict:
    """Set thread hints, enforce affinity, and return JSON-safe policy evidence.

    Repeated calls use the first observed eligible CPU set for this process.
    After a module reload, the already-enforced affinity remains the eligible
    set, so applying the same limit again preserves exactly the same CPU IDs.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    environment = {
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "NUMEXPR_MAX_THREADS": str(threads),
        "NUMBA_NUM_THREADS": str(threads),
        "OMP_WAIT_POLICY": "PASSIVE",
        "MKL_DYNAMIC": "FALSE",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
    }
    os.environ.update(environment)
    try:
        process = psutil.Process()
        available = sorted(set(process.cpu_affinity()))
        if not available or any(not isinstance(cpu, int) or cpu < 0 for cpu in available):
            raise RuntimeError("Operating system returned an invalid eligible CPU set")
        eligible = _ORIGINAL_ELIGIBLE.get(process.pid, available)
        assigned = eligible[:threads]
        process.cpu_affinity(assigned)
        actual = sorted(set(process.cpu_affinity()))
        if actual != assigned:
            raise RuntimeError(
                f"CPU affinity enforcement failed: requested {assigned}, observed {actual}"
            )
    except (AttributeError, NotImplementedError, OSError, psutil.Error) as exc:
        raise RuntimeError(
            "Cannot enforce the benchmark CPU affinity limit; refusing to run "
            "with an unverifiable CPU budget"
        ) from exc
    # Retain the initial eligible set only after the OS confirms enforcement.
    _ORIGINAL_ELIGIBLE.setdefault(process.pid, list(eligible))
    return {
        "policy_version": "cpu-affinity-v1",
        "eligible_cpus": list(eligible),
        "assigned_cpus": list(actual),
        "thread_limit": threads,
        "effective_cpu_count": len(actual),
        "affinity_enforced": True,
        "environment": environment,
    }
