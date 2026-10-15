import json
import urllib.error

import pytest

from cst_mil.research_followup.api import (
    BudgetLedger,
    build_request,
    call_one,
    parse_completion,
    validate_evidence,
)


def fixture_session():
    return {"post_id": "s", "label": 1}, [
        {"comment_id": "a", "raw_text": "请停止辱骂", "comment_label": 1, "parent_id": None}
    ]


def test_payload_excludes_gold_and_overrides_default_model():
    s, m = fixture_session()
    payload = build_request(s, m)
    assert payload["model"] == "gemini-2.5-flash"
    assert "comment_label" not in json.dumps(payload)
    assert '"label"' not in json.dumps(payload)


def test_evidence_reports_fabrication_separately():
    _, messages = fixture_session()
    out = validate_evidence(
        {
            "risk_score": 0.8,
            "top3_evidence": [
                {"message_id": "a", "quote": "不存在的文字", "reason": "x"},
                {"message_id": "unknown", "quote": "q", "reason": "r"},
            ],
        },
        messages,
    )
    assert out["fabricated_quote_count"] == 1
    assert out["nonexistent_id_count"] == 1


def test_refusal_bad_json_wrong_model_and_nonfinite_scores():
    with pytest.raises(ValueError, match="model"):
        parse_completion({"model": "gpt-5", "choices": []})
    for content in ["not json", '{"risk_score": NaN,"top3_evidence":[]}']:
        with pytest.raises(ValueError):
            parse_completion(
                {
                    "model": "gemini-2.5-flash",
                    "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                }
            )


def test_budget_reservation_survives_crash_and_prevents_overspend(tmp_path):
    ledger = BudgetLedger(tmp_path, 0.1)
    ledger.reserve("s", 0.08)
    resumed = BudgetLedger(tmp_path, 0.1)
    assert resumed.spent_bound == pytest.approx(0.08)
    with pytest.raises(ValueError, match="budget"):
        resumed.reserve("s2", 0.03)


def test_retry_records_error_and_resume_does_not_call_again(tmp_path):
    s, m = fixture_session()
    calls = []

    def transport(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise TimeoutError("secret-in-exception-must-not-be-written")
        return {
            "model": "gemini-2.5-flash",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "risk_score": 0.7,
                                "top3_evidence": [
                                    {"message_id": "a", "quote": "停止", "reason": "risk candidate"}
                                ],
                            }
                        )
                    },
                }
            ],
        }

    ledger = BudgetLedger(tmp_path, 1)
    out = call_one(s, m, tmp_path, ledger, 0.1, 1, transport, wait=lambda _: None)
    assert out["status"] == "success"
    assert len(calls) == 2
    call_one(s, m, tmp_path, ledger, 0.1, 1, transport, wait=lambda _: None)
    assert len(calls) == 2
    assert "secret-in-exception" not in "".join(
        p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json")
    )


@pytest.mark.parametrize("status, expected_calls", [(429, 3), (503, 3), (401, 1)])
def test_http_retry_policy(tmp_path, status, expected_calls):
    s, messages = fixture_session()
    calls = []

    def transport(payload):
        calls.append(payload)
        raise urllib.error.HTTPError("https://example.invalid", status, "private", {}, None)

    out = call_one(
        s, messages, tmp_path, BudgetLedger(tmp_path, 1), 0.1, 1, transport, wait=lambda _: None
    )
    assert len(calls) == expected_calls
    assert out["attempts"][0]["http_status"] == status


def test_threshold_is_frozen_before_test_and_cannot_change(tmp_path):
    from cst_mil.research_followup.api import freeze_validation

    selected = [{"post_id": str(i), "split": "dev", "label": i % 2} for i in range(20)]
    results = [
        {
            "session_id": str(i),
            "split": "dev",
            "status": "success",
            "output": {"risk_score": 0.9 if i % 2 else 0.1},
        }
        for i in range(20)
    ]
    frozen = freeze_validation(results, selected, tmp_path)
    assert (tmp_path / "frozen_threshold.json").exists()
    assert freeze_validation(results, selected, tmp_path) == frozen
    results[0]["output"]["risk_score"] = 0.2
    with pytest.raises(ValueError, match="validation changed"):
        freeze_validation(results, selected, tmp_path)


def test_incomplete_validation_blocks_threshold(tmp_path):
    from cst_mil.research_followup.api import freeze_validation

    with pytest.raises(ValueError, match="20"):
        freeze_validation([], [], tmp_path)
    assert not (tmp_path / "frozen_threshold.json").exists()


def test_invalid_budget_does_not_read_credentials(monkeypatch):
    from cst_mil.research_followup import api

    monkeypatch.setattr(api, "read_credentials", lambda: pytest.fail("read secret too early"))
    with pytest.raises(ValueError, match="approved"):
        api._run_locked(budget=None, input_price=0.1, output_price=1)


@pytest.mark.parametrize("usage", [None, [], "bad"])
def test_malformed_usage_retains_receipt_and_reservation(tmp_path, usage):
    s, messages = fixture_session()
    response = {
        "model": "gemini-2.5-flash",
        "usage": usage,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": '{"risk_score":0.7,"top3_evidence":[]}'},
            }
        ],
    }
    ledger = BudgetLedger(tmp_path, 1)
    out = call_one(s, messages, tmp_path, ledger, 0.1, 1, lambda _: response)
    assert out["status"] == "success"
    assert out["attempts"][0]["usage_status"] == "unknown_keep_reservation"
    assert ledger.entries[0]["status"] == "usage_unknown_keep_reservation"
    assert len(list((tmp_path / "responses").glob("*.json"))) == 1
    assert len(list((tmp_path / "sessions").glob("*.json"))) == 1


def test_billing_overrun_preserves_receipt_and_stops_run(tmp_path, monkeypatch):
    from cst_mil.research_followup import api

    monkeypatch.setattr(api, "OUT", tmp_path)
    monkeypatch.setattr(api, "read_credentials", lambda: "test-placeholder")
    selected = [{"post_id": str(i), "split": "dev", "label": i % 2} for i in range(20)]
    _, messages = fixture_session()
    monkeypatch.setattr(
        api,
        "select_sessions",
        lambda: (
            {"content_fingerprint": "test"},
            selected,
            {s["post_id"]: messages for s in selected},
        ),
    )
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps(
                {
                    "model": "gemini-2.5-flash",
                    "usage": {"prompt_tokens": 1000000, "completion_tokens": 1},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"risk_score":0.7,"top3_evidence":[]}'},
                        }
                    ],
                }
            )

    def transport(*args, **kwargs):
        calls.append(1)
        return Response()

    monkeypatch.setattr(api.urllib.request, "urlopen", transport)
    api._run_locked(budget=0.1, input_price=1, output_price=1)
    assert len(calls) == 1
    folder = tmp_path / "api_run"
    assert len(list((folder / "responses").glob("*.json"))) == 1
    assert len(list((folder / "attempts").glob("*.json"))) == 1
    results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    assert results[0]["budget_stop"] is True
    assert (folder / "STOPPED.json").exists()
