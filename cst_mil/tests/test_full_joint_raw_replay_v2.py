from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from cst_mil.research_followup.full_joint_comparison.common import save
from cst_mil.research_followup.full_joint_comparison.raw_replay_v2 import (
    _lineage_handshake_proof,
)


def test_lineage_accepts_self_or_redirector_but_checks_birth_owner_and_stage(monkeypatch):
    class FakeProcess:
        def __init__(self, pid, created, parents):
            self.pid, self.created, self.parent_ids = pid, created, parents

        def parents(self):
            return [objects[pid] for pid in self.parent_ids]

        def create_time(self):
            return self.created

        def is_running(self):
            return True

    objects = {
        10: FakeProcess(10, 1.0, []),
        20: FakeProcess(20, 2.0, [10]),
        30: FakeProcess(30, 3.0, [20, 10]),
    }
    monkeypatch.setattr(psutil, "Process", lambda pid: objects[pid])
    owner = {"pid": 10, "created": 1.0}
    receipt = {"pid": 20, "created": 2.0, "stage": "dev", "method": "lda", "seed": 42}
    proof = _lineage_handshake_proof(30, receipt, owner, stage="dev", method="lda", seed=42)
    assert proof["pid_match_kind"] == "ancestor_redirector"
    assert proof["launcher_create_time_verified"] is True
    direct = {**receipt, "pid": 30, "created": 3.0}
    assert (
        _lineage_handshake_proof(30, direct, owner, stage="dev", method="lda", seed=42)[
            "pid_match_kind"
        ]
        == "self"
    )
    with pytest.raises(RuntimeError, match="PID/create-time"):
        _lineage_handshake_proof(
            30, {**receipt, "created": 99.0}, owner, stage="dev", method="lda", seed=42
        )
    with pytest.raises(RuntimeError, match="live parent ancestor"):
        _lineage_handshake_proof(
            30, receipt, {"pid": 999, "created": 1.0}, stage="dev", method="lda", seed=42
        )
    with pytest.raises(RuntimeError, match="stage/method/seed"):
        _lineage_handshake_proof(30, receipt, owner, stage="test", method="lda", seed=42)
    with pytest.raises(RuntimeError, match="owner.*PID/create-time"):
        _lineage_handshake_proof(
            30, receipt, {**owner, "created": 99.0}, stage="dev", method="lda", seed=42
        )
    assert (
        _lineage_handshake_proof(
            30, {**receipt, "pid": 999}, owner, stage="dev", method="lda", seed=42
        )
        is None
    )


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows virtualenv redirector test")
def test_windows_venv_launcher_matches_real_python_child_lineage(tmp_path: Path):
    # This imports only the launcher helper and constants: no model, dataset,
    # fitting, embedding or experimental prediction is performed.
    python = Path(sys.executable)
    owner = psutil.Process()
    save(tmp_path / "campaign.lock", {"pid": owner.pid, "created": owner.create_time()})
    script = (
        "import json,os;from pathlib import Path;"
        "from cst_mil.research_followup.full_joint_comparison.raw_replay_v2 "
        "import _await_child_receipt;"
        f"p=_await_child_receipt(Path({str(tmp_path)!r}),os.getpid(),"
        "stage='dev',method='lda',seed=42);print(json.dumps(p))"
    )
    process = subprocess.Popen(
        [str(python), "-X", "utf8", "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        launcher = psutil.Process(process.pid)
        save(
            tmp_path / "active_child.json",
            {
                "pid": process.pid,
                "created": launcher.create_time(),
                "stage": "dev",
                "method": "lda",
                "seed": 42,
            },
        )
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        proof = json.loads(stdout.strip())
        assert proof["launcher_pid"] == process.pid
        assert proof["parent_owner_pid"] == owner.pid
        assert proof["launcher_create_time_verified"] is True
        assert proof["parent_lock_create_time_verified"] is True
        assert proof["owner_is_live_ancestor"] is True
        if proof["python_pid"] != process.pid:
            assert proof["pid_match_kind"] == "ancestor_redirector"
        result = {
            "status": "passed",
            "python_executable": str(python),
            "owner_pid": owner.pid,
            "popen_launcher_pid": process.pid,
            "handshake": proof,
            "model_or_dataset_loaded": False,
            "training_or_inference_performed": False,
        }
        (tmp_path / "windows_venv_launcher_v2_probe.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
