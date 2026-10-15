"""Bounded, resumable 6+30 Gemini pilot. Never invokes the full-suite API runner."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .. import api as legacy
from ..io import inputs
from .core import OUT, code_sources, guard_output, read, save, stable, verify_sources

BudgetLedger = legacy.BudgetLedger


class NoCredentialRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a paid request or its bearer credential through a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def sample(sessions, grouped):
    lengths = {
        s["post_id"]: sum(len(m["raw_text"]) for m in grouped[s["post_id"]]) for s in sessions
    }
    bounds = np.quantile(
        [lengths[s["post_id"]] for s in sessions if s["split"] == "train"], [1 / 3, 2 / 3]
    ).tolist()
    selected = []
    for split, n in [("dev", 1), ("test", 5)]:
        for label in [0, 1]:
            for band in range(3):
                pool = [
                    s
                    for s in sessions
                    if s["split"] == split
                    and s["label"] == label
                    and int(np.searchsorted(bounds, lengths[s["post_id"]])) == band
                ]
                pool.sort(key=lambda s: stable("20260929|" + s["post_id"]))
                if len(pool) < n:
                    raise ValueError(f"Insufficient stratum {split}/{label}/{band}")
                selected.extend(pool[:n])
    assert len({s["post_id"] for s in selected}) == 36
    return selected, bounds


def _cost(response, inp, output):
    usage = response.get("usage") if isinstance(response, dict) else None
    if isinstance(usage, dict) and all(
        type(usage.get(k)) is int and usage[k] >= 0 for k in ["prompt_tokens", "completion_tokens"]
    ):
        return (usage["prompt_tokens"] * inp + usage["completion_tokens"] * output) / 1e6
    return None


def _interpret(result, response, messages):
    if not isinstance(response, dict):
        result["status"] = "format_error"
        return
    result["returned_model"] = response.get("model")
    if response.get("model") != legacy.MODEL:
        result["status"] = "model_mismatch"
        return
    choices = response.get("choices") or []
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
        or not isinstance(choices[0].get("message"), dict)
    ):
        result["status"] = "format_error"
        return
    if choices and (
        choices[0].get("message", {}).get("refusal")
        or choices[0].get("finish_reason") == "content_filter"
    ):
        result["status"] = "refusal"
        return
    try:
        output = legacy.parse_completion(response)
        result.update(
            status="success", output=output, validation=legacy.validate_evidence(output, messages)
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        result["status"] = "format_error"


def call_one(
    session,
    messages,
    folder,
    ledger,
    input_price,
    output_price,
    transport,
    wait=time.sleep,
    *,
    max_attempts=2,
    before_request=None,
    after_attempt=None,
):
    folder = guard_output(folder)
    payload = legacy.build_request(session, messages)
    sid = session["post_id"]
    token = stable(sid)
    signature = legacy.signature(payload)
    dest = folder / "sessions" / f"{token}.json"
    if dest.exists():
        previous = read(dest)
        if previous["request_sha256"] != signature:
            raise ValueError("Resume request mismatch")
        return previous
    result = dict(
        session_id=sid,
        split=session.get("split"),
        request_sha256=signature,
        requested_model=legacy.MODEL,
        status="infrastructure_failure",
        attempts=[],
    )
    prior = [(i, e) for i, e in enumerate(ledger.entries) if e["session_id"] == sid]
    # Recovery never repeats an already reserved attempt. A receipt can be replayed locally.
    for number, (index, entry) in enumerate(prior, 1):
        receipt_path = folder / "receipts" / f"{token}_{number}.json"
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt["request_sha256"] != signature:
                raise ValueError("Receipt request mismatch")
            result["attempts"].append(receipt["attempt"])
            try:
                ledger.settle(index, receipt["attempt"]["cost_estimate_usd"])
            except ValueError:
                result["budget_stop"] = True
            if receipt["response"] is not None:
                _interpret(result, receipt["response"], messages)
                save(dest, result)
                return result
            if not receipt["attempt"]["retryable"]:
                result["status"] = "transport_failure"
                save(dest, result)
                return result
        else:
            result["attempts"].append(
                dict(
                    attempt=number,
                    status="interrupted_unknown",
                    reserved_usd=entry["accounted_usd"],
                )
            )
            result["status"] = "interrupted_unknown"
            save(dest, result)
            return result
    cost_bound = (legacy.input_bound(payload) * input_price + 2048 * output_price) / 1e6
    for number in range(len(prior), min(max_attempts, 2)):
        if before_request:
            before_request()
        index = ledger.reserve(sid, cost_bound)
        start = time.perf_counter()
        response, error, http_status, retryable = None, None, None, False
        try:
            response = transport(payload)
        except urllib.error.HTTPError as exc:
            error, http_status = type(exc).__name__, exc.code
            retryable = exc.code == 429 or 500 <= exc.code < 600
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            error = type(exc).__name__
            retryable = isinstance(getattr(exc, "reason", exc), TimeoutError)
        except (ValueError, UnicodeDecodeError) as exc:
            error = type(exc).__name__
        seconds = time.perf_counter() - start
        cost = _cost(response, input_price, output_price)
        attempt = dict(
            attempt=number + 1,
            seconds=seconds,
            cost_estimate_usd=cost,
            reserved_usd=cost_bound,
            transport_error=error,
            http_status=http_status,
            retryable=retryable,
            ledger_index=index,
            usage=response.get("usage") if isinstance(response, dict) else None,
        )
        receipt = dict(
            session_id=sid,
            request_sha256=signature,
            attempt=attempt,
            response={k: response.get(k) for k in ["model", "usage", "choices", "id"]}
            if isinstance(response, dict)
            else response,
        )
        # Persist received content even when actual usage exceeds its reservation.
        save(folder / "receipts" / f"{token}_{number + 1}.json", receipt)
        result["attempts"].append(attempt)
        try:
            ledger.settle(index, cost)
        except ValueError:
            if ledger.spent_bound <= ledger.budget:
                raise
            result["budget_stop"] = True
        stop = after_attempt(attempt) if after_attempt else False
        if response is not None:
            _interpret(result, response, messages)
            break
        result["status"] = "infrastructure_failure" if retryable else "transport_failure"
        if stop or result.get("budget_stop") or not retryable:
            break
        if number + 1 < min(max_attempts, 2):
            wait(2**number)
    save(dest, result)
    return result


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as stream:
        return json.load(stream)


def prepare(out=OUT):
    folder = guard_output(out) / "gemini"
    manifest, sessions, _, grouped = inputs()
    selected, bounds = sample(sessions, grouped)
    selection = dict(
        data_fingerprint=manifest["content_fingerprint"],
        training_length_bounds=bounds,
        sessions=[
            dict(
                post_id=s["post_id"],
                split=s["split"],
                label=s["label"],
                length=sum(len(m["raw_text"]) for m in grouped[s["post_id"]]),
                prompt_sha256=legacy.signature(legacy.build_request(s, grouped[s["post_id"]])),
            )
            for s in selected
        ],
    )
    path = folder / "selection.json"
    if path.exists() and read(path) != selection:
        raise ValueError("Frozen sample changed")
    save(path, selection)
    # Refresh prices before the first paid campaign; once begun, retain quoted upper rates.
    if not (folder / "run_config.json").exists():
        pricing = _get("https://api.quickrouter.ai/api/pricing")
        status = _get("https://api.quickrouter.ai/api/status")["data"]
        model = next(x for x in pricing["data"] if x["model_name"] == legacy.MODEL)
        quota = float(status["quota_per_unit"])
        if status.get("currency_type") != "USD" or quota <= 0:
            raise ValueError("Unverified currency conversion")
        groups = []
        for group in model["enable_groups"]:
            inp = 1e6 * model["model_ratio"] * pricing["group_ratio"][group] / quota
            groups.append(
                dict(
                    group=group,
                    input_usd_per_million=inp,
                    output_usd_per_million=inp * model["completion_ratio"],
                )
            )
        highest = max(groups, key=lambda x: x["input_usd_per_million"])
        input_total = sum(
            legacy.input_bound(legacy.build_request(s, grouped[s["post_id"]])) for s in selected
        )
        upper = (
            input_total * highest["input_usd_per_million"]
            + 36 * 2048 * highest["output_usd_per_million"]
        ) / 1e6
        quote = dict(
            model=legacy.MODEL,
            groups=groups,
            highest=highest,
            one_pass_upper_usd=upper,
            two_attempt_upper_usd=2 * upper,
            input_utf8_bound=input_total,
            output_token_cap=2048,
            checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            account_group="unknown",
            account_debit="unknown; public-rate usage estimate only",
        )
        save(
            folder / "pricing_snapshot.json",
            dict(
                model=model,
                group_ratio=pricing["group_ratio"],
                quota_per_unit=quota,
                currency_type="USD",
            ),
        )
        save(folder / "quote.json", quote)
    return selection, read(folder / "quote.json")


def run(out=OUT, budget=2.0):
    if budget <= 0 or budget > 2:
        raise ValueError("Budget must be positive and at most approved USD 2")
    folder = guard_output(out) / "gemini"
    selection, quote = prepare(out)
    _, sessions, _, grouped = inputs()
    lookup = {s["post_id"]: s for s in sessions}
    config_path = folder / "run_config.json"
    binding = code_sources(
        Path(__file__), Path(legacy.__file__), folder / "selection.json", folder / "quote.json"
    )
    config = dict(
        budget_usd=budget, model=legacy.MODEL, threshold=0.5, max_attempts=2, sources=binding
    )
    if config_path.exists() and read(config_path) != config:
        raise ValueError("Frozen API configuration changed")
    save(config_path, config)
    ledger = BudgetLedger(folder, budget)
    key = legacy.read_credentials()
    base = legacy.BASE
    opener = urllib.request.build_opener(NoCredentialRedirect())

    def transport(payload):
        request = urllib.request.Request(
            base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            method="POST",
        )
        with opener.open(request, timeout=120) as stream:
            return json.load(stream)

    state = {"consecutive_infrastructure_failures": 0}
    history = sorted(
        (folder / "receipts").glob("*.json"), key=lambda p: read(p)["attempt"]["ledger_index"]
    )
    for path in history:
        a = read(path)["attempt"]
        state["consecutive_infrastructure_failures"] = (
            state["consecutive_infrastructure_failures"] + 1 if a["retryable"] else 0
        )

    def after(attempt):
        state["consecutive_infrastructure_failures"] = (
            state["consecutive_infrastructure_failures"] + 1 if attempt["retryable"] else 0
        )
        return state["consecutive_infrastructure_failures"] >= 3

    results = []
    stop_reason = None
    for row in selection["sessions"]:
        if row["split"] == "test" and (
            not (folder / "validation_gate.json").exists()
            or not read(folder / "validation_gate.json")["passed"]
        ):
            stop_reason = "validation_format_gate_failed"
            break
        if state["consecutive_infrastructure_failures"] >= 3:
            stop_reason = "three_consecutive_infrastructure_failures"
            break
        s = lookup[row["post_id"]]
        try:
            result = call_one(
                s,
                grouped[s["post_id"]],
                folder,
                ledger,
                quote["highest"]["input_usd_per_million"],
                quote["highest"]["output_usd_per_million"],
                transport,
                before_request=lambda: verify_sources(binding),
                after_attempt=after,
            )
        except ValueError as exc:
            if "budget" not in str(exc).lower():
                raise
            stop_reason = "budget_insufficient_before_request"
            break
        results.append(result)
        print(
            json.dumps(
                dict(
                    completed=len(results),
                    split=s["split"],
                    status=result["status"],
                    cost_bound_usd=ledger.spent_bound,
                )
            ),
            flush=True,
        )
        if result.get("budget_stop") or result["status"] in [
            "model_mismatch",
            "transport_failure",
            "interrupted_unknown",
        ]:
            stop_reason = "budget_or_fatal_response"
            break
        dev = [r for r in results if r["split"] == "dev"]
        if len(dev) == 6 and not (folder / "validation_gate.json").exists():
            gate = dict(
                passed=all(r["status"] == "success" for r in dev),
                n_dev=6,
                prompt_sha256=stable(legacy.SYSTEM),
                threshold=0.5,
            )
            save(folder / "validation_gate.json", gate)
            if not gate["passed"]:
                stop_reason = "validation_format_gate_failed"
                break
    verify_sources(binding)
    # Terminal resume keeps every durable session, even if the failure gate stopped the loop early.
    results = [
        read(folder / "sessions" / f"{stable(row['post_id'])}.json")
        for row in selection["sessions"]
        if (folder / "sessions" / f"{stable(row['post_id'])}.json").exists()
    ]
    summary = dict(
        completed=len(results),
        status="complete" if len(results) == 36 else "stopped",
        stop_reason=stop_reason,
        accounted_upper_rate_usd=ledger.spent_bound,
        account_debit_usd=None,
        attempts=len(ledger.entries),
        results=results,
        **state,
    )
    save(folder / "summary.json", summary)
    return summary
