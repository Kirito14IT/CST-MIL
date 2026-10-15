import json

import pytest

from cst_mil.research_followup.round2 import core


def test_protocol_is_immutable_and_small(tmp_path):
    value = core.initialize(tmp_path, preserve=False)
    assert value["gemini"]["n_dev"] == 6
    assert value["gemini"]["n_test"] == 30
    assert value["gemini"]["budget_usd"] == 2
    assert value["seeds"] == [42, 43, 44, 45, 46]
    file = tmp_path / "protocol.json"
    changed = json.loads(file.read_text(encoding="utf-8"))
    changed["gemini"]["n_test"] = 136
    file.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        core.initialize(tmp_path, preserve=False)


def test_output_cannot_point_at_old_campaign():
    with pytest.raises(ValueError, match="protected"):
        core.guard_output(core.OLD / "extra")


def test_source_changes_stop_experiment(tmp_path):
    source = tmp_path / "model.py"
    source.write_text("first", encoding="utf-8")
    binding = core.source_hashes([source])
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        core.verify_sources(binding)
