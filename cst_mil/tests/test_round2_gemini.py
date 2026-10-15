import json
import urllib.error

import pytest

from cst_mil.research_followup.round2 import gemini


def sample_data():
    sessions, grouped = [], {}
    for split in ["train", "dev", "test"]:
        for label in [0, 1]:
            for length in [100, 500, 2000]:
                for i in range(8):
                    sid = f"{split}-{label}-{length}-{i}"
                    sessions.append(dict(post_id=sid, label=label, split=split))
                    grouped[sid] = [
                        dict(
                            comment_id=sid + ":0",
                            raw_text="x" * length,
                            comment_label=label,
                            parent_id=None,
                        )
                    ]
    return sessions, grouped


def test_sample_is_frozen_stratified_and_ignores_test_text_for_cutoffs():
    sessions, grouped = sample_data()
    selected, bounds = gemini.sample(sessions, grouped)
    assert len([s for s in selected if s["split"] == "dev"]) == 6
    assert len([s for s in selected if s["split"] == "test"]) == 30
    assert len({s["post_id"] for s in selected}) == 36
    assert gemini.sample(sessions[::-1], grouped) == (selected, bounds)
    grouped[sessions[-1]["post_id"]][0]["raw_text"] = "x" * 100000
    assert gemini.sample(sessions, grouped)[1] == bounds


def response(model="gemini-2.5-flash", usage=None):
    return {
        "model": model,
        "usage": usage,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "risk_score": 0.7,
                            "top3_evidence": [
                                {"message_id": "a", "quote": "原文", "reason": "上下文"}
                            ],
                        }
                    )
                },
            }
        ],
    }


def test_max_two_attempts_and_durable_resume(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        raise TimeoutError("do-not-log-this-secret")

    out = gemini.call_one(
        {"post_id": "s", "split": "test"},
        [{"comment_id": "a", "raw_text": "原文"}],
        tmp_path,
        gemini.BudgetLedger(tmp_path, 2),
        0.1,
        1,
        transport,
        wait=lambda _: None,
    )
    assert out["status"] == "infrastructure_failure"
    assert len(calls) == 2
    gemini.call_one(
        {"post_id": "s", "split": "test"},
        [{"comment_id": "a", "raw_text": "原文"}],
        tmp_path,
        gemini.BudgetLedger(tmp_path, 2),
        0.1,
        1,
        transport,
        wait=lambda _: None,
    )
    assert len(calls) == 2
    assert "do-not-log" not in "".join(
        p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json")
    )


def test_receipt_survives_unknown_usage_and_budget_overrun(tmp_path):
    s, m = {"post_id": "s", "split": "test"}, [{"comment_id": "a", "raw_text": "原文"}]
    result = gemini.call_one(
        s,
        m,
        tmp_path,
        gemini.BudgetLedger(tmp_path, 0.1),
        1,
        1,
        lambda _: response(usage={"prompt_tokens": 1000000, "completion_tokens": 1}),
    )
    assert result["budget_stop"]
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 1
    assert len(list((tmp_path / "sessions").glob("*.json"))) == 1


def test_wrong_model_is_explicit_and_budget_prevents_call(tmp_path):
    s, m = {"post_id": "s"}, [{"comment_id": "a", "raw_text": "原文"}]
    result = gemini.call_one(
        s, m, tmp_path, gemini.BudgetLedger(tmp_path, 2), 0.1, 1, lambda _: response("other-model")
    )
    assert result["status"] == "model_mismatch"
    with pytest.raises(ValueError, match="budget"):
        gemini.call_one(
            {"post_id": "s2"},
            m,
            tmp_path,
            gemini.BudgetLedger(tmp_path, 0.000001),
            1,
            1,
            lambda _: pytest.fail("should not call"),
        )


def test_non_timeout_transport_failure_is_not_retried(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        raise urllib.error.URLError("certificate verification failed")

    result = gemini.call_one(
        {"post_id": "s"},
        [{"comment_id": "a", "raw_text": "原文"}],
        tmp_path,
        gemini.BudgetLedger(tmp_path, 2),
        0.1,
        1,
        transport,
        wait=lambda _: None,
    )
    assert len(calls) == 1
    assert result["status"] == "transport_failure"


def test_interrupted_unknown_reservation_never_dispatches(tmp_path):
    ledger = gemini.BudgetLedger(tmp_path, 2)
    ledger.reserve("s", 0.05)
    result = gemini.call_one(
        {"post_id": "s"},
        [{"comment_id": "a", "raw_text": "原文"}],
        tmp_path,
        ledger,
        0.1,
        1,
        lambda _: pytest.fail("Unknown request must not repeat"),
    )
    assert result["status"] == "interrupted_unknown"
    assert ledger.spent_bound == 0.05


def test_global_failure_callback_stops_retry(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        raise TimeoutError()

    result = gemini.call_one(
        {"post_id": "s"},
        [{"comment_id": "a", "raw_text": "原文"}],
        tmp_path,
        gemini.BudgetLedger(tmp_path, 2),
        0.1,
        1,
        transport,
        wait=lambda _: None,
        after_attempt=lambda _: True,
    )
    assert len(calls) == 1 and result["status"] == "infrastructure_failure"


def test_paid_redirect_cannot_forward_credential():
    req = gemini.urllib.request.Request(
        "https://api.quickrouter.ai/v1/chat/completions",
        data=b"{}",
        headers={"Authorization": "Bearer sentinel"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError):
        gemini.NoCredentialRedirect().redirect_request(
            req, None, 302, "redirect", {}, "http://other.invalid/"
        )
