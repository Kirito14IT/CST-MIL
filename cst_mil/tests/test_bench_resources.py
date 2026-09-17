"""CPU policy tests use fake process affinity and never alter the pytest mask."""

from __future__ import annotations

import importlib
import json
import os

import pytest

from cst_mil.bench import resources


class _Process:
    pid = 1901

    def __init__(self, eligible):
        self.affinity = list(eligible)
        self.assigned = []

    def cpu_affinity(self, cpus=None):
        if cpus is not None:
            self.assigned.append(list(cpus))
            self.affinity = list(cpus)
        return list(self.affinity)


@pytest.fixture(autouse=True)
def isolate_policy(monkeypatch):
    monkeypatch.setattr(resources, "_ORIGINAL_ELIGIBLE", {})
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "NUMBA_NUM_THREADS",
        "OMP_WAIT_POLICY", "MKL_DYNAMIC", "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES",
    ):
        # monkeypatch restores the environment after configure_cpu writes to it.
        monkeypatch.setenv(key, "previous-test-value")


def test_affinity_caps_first_four_sorted_eligible_cpus_and_sets_thread_hints(monkeypatch):
    process = _Process([12, 2, 9, 4, 7, 15])
    monkeypatch.setattr(resources.psutil, "Process", lambda: process)
    policy = resources.configure_cpu(4)
    assert policy["eligible_cpus"] == [2, 4, 7, 9, 12, 15]
    assert policy["assigned_cpus"] == [2, 4, 7, 9]
    assert policy["effective_cpu_count"] == 4
    assert policy["thread_limit"] == 4
    assert policy["affinity_enforced"]
    assert json.loads(json.dumps(policy)) == policy
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "NUMBA_NUM_THREADS",
    ):
        assert os.environ[key] == "4"
    assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"
    assert os.environ["MKL_DYNAMIC"] == "FALSE"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_repeated_call_and_module_reload_keep_assigned_cpu_ids(monkeypatch):
    process = _Process([1, 3, 5, 7, 9, 11])
    monkeypatch.setattr(resources.psutil, "Process", lambda: process)
    first = resources.configure_cpu(4)
    assert resources.configure_cpu(4) == first
    reloaded = importlib.reload(resources)
    assert reloaded.configure_cpu(4)["assigned_cpus"] == first["assigned_cpus"]
    assert process.affinity == [1, 3, 5, 7]


def test_fewer_than_four_eligible_cpus_never_expands_affinity(monkeypatch):
    process = _Process([6, 2])
    monkeypatch.setattr(resources.psutil, "Process", lambda: process)
    policy = resources.configure_cpu(4)
    assert policy["assigned_cpus"] == [2, 6]
    assert policy["effective_cpu_count"] == 2
    assert policy["thread_limit"] == 4


@pytest.mark.parametrize("failure", [PermissionError("denied"), NotImplementedError("unavailable")])
def test_affinity_query_failures_refuse_to_run(monkeypatch, failure):
    class Broken(_Process):
        def cpu_affinity(self, cpus=None):
            raise failure

    monkeypatch.setattr(resources.psutil, "Process", lambda: Broken([0, 1, 2, 3]))
    with pytest.raises(RuntimeError, match="Cannot enforce the benchmark CPU affinity"):
        resources.configure_cpu(4)


def test_affinity_set_failure_refuses_to_run(monkeypatch):
    class Broken(_Process):
        def cpu_affinity(self, cpus=None):
            if cpus is not None:
                raise PermissionError("setting affinity denied")
            return list(self.affinity)

    monkeypatch.setattr(resources.psutil, "Process", lambda: Broken(range(8)))
    with pytest.raises(RuntimeError, match="Cannot enforce the benchmark CPU affinity"):
        resources.configure_cpu(4)
    assert resources._ORIGINAL_ELIGIBLE == {}


def test_affinity_readback_must_match_requested_mask(monkeypatch):
    class IgnoredSet(_Process):
        def cpu_affinity(self, cpus=None):
            return list(self.affinity)

    monkeypatch.setattr(resources.psutil, "Process", lambda: IgnoredSet(range(8)))
    with pytest.raises(RuntimeError, match="CPU affinity enforcement failed"):
        resources.configure_cpu(4)


def test_empty_eligible_mask_is_not_interpreted_as_all_cpus(monkeypatch):
    monkeypatch.setattr(resources.psutil, "Process", lambda: _Process([]))
    with pytest.raises(RuntimeError, match="invalid eligible CPU set"):
        resources.configure_cpu(4)


@pytest.mark.parametrize("threads", [0, -1, 1.5, "4", True, None])
def test_invalid_thread_limit_is_rejected(threads):
    with pytest.raises(ValueError, match="positive integer"):
        resources.configure_cpu(threads)
