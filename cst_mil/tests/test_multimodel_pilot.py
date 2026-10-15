import io
import json
import urllib.error

import pytest

from cst_mil.research_followup.multimodel import core

SESSION = {"post_id": "case", "split": "dev", "label": 1}
MESSAGES = [{"comment_id": "m1", "raw_text": "原文句子", "parent_id": None, "cb_label": 1}]
PRICES = {"input_usd_per_million": 0.66, "output_usd_per_million": 1.98}


def response(model="gpt-4o-mini", content=None, usage=True):
    obj = {
        "model": model,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        content
                        or {
                            "risk_score": 0.8,
                            "top3_evidence": [
                                {"message_id": "m1", "quote": "原文", "reason": "原文解释"}
                            ],
                        },
                        ensure_ascii=False,
                    )
                },
            }
        ],
    }
    if usage:
        obj["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 0},
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    return obj


def call(tmp_path, transport, model="gpt-4o-mini", **kwargs):
    return core.call_one(
        SESSION,
        MESSAGES,
        model,
        tmp_path,
        core.Ledger(tmp_path),
        PRICES,
        transport,
        wait=lambda _: None,
        **kwargs,
    )


def test_metadata_and_explicit_disabled_thinking():
    value = core.request(SESSION, MESSAGES, "deepseek-v4-flash")
    assert value["thinking"] == {"type": "disabled"}
    assert value["model"] == "deepseek-v4-flash"
    assert value["temperature"] == 0 and value["max_tokens"] == 2048
    user = json.loads(value["messages"][1]["content"])
    assert set(user["messages"][0]) == {"message_id", "parent_id", "text"}
    assert "label" not in user


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retry_only_transient_http(tmp_path, status):
    calls = []

    def transport(_):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("mock", status, "error", {}, io.BytesIO(b"temporary"))
        return response()

    result = call(tmp_path, transport)
    assert result["status"] == "success" and len(calls) == 2


@pytest.mark.parametrize(
    "value",
    ["nonjson", {"risk_score": 2, "top3_evidence": []}, {"risk_score": True, "top3_evidence": []}],
)
def test_illegal_output_not_retried(tmp_path, value):
    calls = []

    def transport(_):
        calls.append(1)
        r = response(content=value if isinstance(value, dict) else None)
        if isinstance(value, str):
            r["choices"][0]["message"]["content"] = value
        return r

    assert call(tmp_path, transport)["status"] == "format_error"
    assert len(calls) == 1


def test_missing_usage_unknown_cost(tmp_path):
    result = call(tmp_path, lambda _: response(usage=False))
    assert result["status"] == "success"
    assert result["attempts"][0]["cost_estimate_usd"] is None
    assert core.Ledger(tmp_path).entries[0]["status"] == "cost_unknown"


def test_resume_never_resends_completed(tmp_path):
    first = call(tmp_path, lambda _: response())

    def forbidden(_):
        raise AssertionError("Must not send paid request on resume")

    assert call(tmp_path, forbidden) == first


def test_ambiguous_crash_does_not_resend(tmp_path):
    payload = core.request(SESSION, MESSAGES, "gpt-4o-mini")
    core.Ledger(tmp_path).reserve("case", core.frozen.signature(payload), 0.1)
    assert (
        call(tmp_path, lambda _: pytest.fail("Ambiguous delivery resent"))["status"]
        == "interrupted_unknown"
    )


def test_request_fingerprint_mismatch(tmp_path):
    call(tmp_path, lambda _: response())
    with pytest.raises(ValueError, match="fingerprint"):
        core.call_one(
            SESSION,
            [{**MESSAGES[0], "raw_text": "changed"}],
            "gpt-4o-mini",
            tmp_path,
            core.Ledger(tmp_path),
            PRICES,
            lambda _: response(),
        )


def test_source_fingerprint_mismatch(tmp_path):
    p = tmp_path / "source.txt"
    p.write_text("before")
    binding = {str(p): core.digest(p)}
    p.write_text("after")
    with pytest.raises(RuntimeError, match="fingerprint"):
        core.verify(binding)


def test_model_identity_mismatch_not_retried(tmp_path):
    assert call(tmp_path, lambda _: response("wrong"))["status"] == "model_mismatch"
    assert len(core.Ledger(tmp_path).entries) == 1


def test_official_gpt_snapshot_explicitly_accepted(tmp_path):
    result = call(tmp_path, lambda _: response("gpt-4o-mini-2024-07-18"))
    assert result["status"] == "success"
    assert result["returned_model"] == "gpt-4o-mini-2024-07-18"


def test_unsupported_thinking_no_silent_fallback(tmp_path):
    def transport(_):
        raise urllib.error.HTTPError("mock", 400, "error", {}, io.BytesIO(b"unsupported thinking"))

    assert call(tmp_path, transport, "deepseek-v4-flash")["status"] == "unsupported_thinking"
    assert len(core.Ledger(tmp_path).entries) == 1


def test_fixed_request_limit_and_optional_budget(tmp_path):
    ledger = core.Ledger(tmp_path, None, max_requests=1)
    ledger.reserve("case", "hash", 0.1)
    with pytest.raises(ValueError, match="request limit"):
        ledger.reserve("other", "hash", 0.1)
    budget = core.Ledger(tmp_path / "budget", 0.01)
    with pytest.raises(ValueError, match="Budget"):
        budget.reserve("case", "hash", 0.1)


def test_two_timeout_attempts_maximum(tmp_path):
    def timeout(_):
        raise TimeoutError()

    result = call(tmp_path, timeout)
    assert result["status"] == "infrastructure_failure" and len(result["attempts"]) == 2


def test_three_infrastructure_failures_callback_stops_retry(tmp_path):
    state = {"n": 2}

    def after(_):
        state["n"] += 1
        return state["n"] >= 3

    result = call(tmp_path, lambda _: (_ for _ in ()).throw(TimeoutError()), after_attempt=after)
    assert len(result["attempts"]) == 1 and state["n"] == 3


def test_fabricated_quote_and_nonexistent_id_are_separate(tmp_path):
    output = {
        "risk_score": 0.9,
        "top3_evidence": [
            {"message_id": "m1", "quote": "不存在的原文", "reason": "x"},
            {"message_id": "missing", "quote": "原文", "reason": "y"},
        ],
    }
    result = call(tmp_path, lambda _: response(content=output))
    assert result["validation"]["nonexistent_id_count"] == 1
    assert result["validation"]["fabricated_quote_count"] == 1


def mock_prepared(tmp_path, monkeypatch):
    selected = [
        {"post_id": f"s{i}", "split": "dev" if i < 6 else "test", "label": i % 2} for i in range(36)
    ]
    selection = {"sessions": selected}
    core.save(tmp_path / "selection.json", selection)
    core.save(tmp_path / "gpt-4o-mini" / "quote.json", {"highest": PRICES})
    monkeypatch.setattr(
        core,
        "prepare",
        lambda _: (
            selection,
            {r["post_id"]: r for r in selected},
            {r["post_id"]: MESSAGES for r in selected},
        ),
    )


def test_end_to_end_fixed36_and_terminal_resume_no_requests(tmp_path, monkeypatch):
    mock_prepared(tmp_path, monkeypatch)
    calls = []

    def transport(_):
        calls.append(1)
        return response()

    first = core.run_model("gpt-4o-mini", tmp_path, transport)
    assert first["completed"] == 36 and first["attempts"] == 36
    assert len(calls) == 36
    second = core.run_model(
        "gpt-4o-mini", tmp_path, lambda _: pytest.fail("Terminal resume repeated requests")
    )
    assert second["completed"] == 36 and second["attempts"] == 36


def test_end_to_end_three_infrastructure_failures_stop_model(tmp_path, monkeypatch):
    mock_prepared(tmp_path, monkeypatch)
    calls = []

    def transport(_):
        calls.append(1)
        raise TimeoutError()

    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    result = core.run_model("gpt-4o-mini", tmp_path, transport)
    assert len(calls) == 3
    assert result["stop_reason"] == "three_consecutive_infrastructure_failures"
    assert result["attempts"] == 3


def test_end_to_end_validation_format_failure_no_test_calls(tmp_path, monkeypatch):
    mock_prepared(tmp_path, monkeypatch)
    calls = []

    def transport(_):
        calls.append(1)
        r = response()
        r["choices"][0]["message"]["content"] = "not json"
        return r

    result = core.run_model("gpt-4o-mini", tmp_path, transport)
    assert len(calls) == 6
    assert result["stop_reason"] == "validation_format_gate_failed"
