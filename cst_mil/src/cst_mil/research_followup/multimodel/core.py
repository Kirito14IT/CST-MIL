"""Crash-safe fixed-cohort API comparison with explicit model identity checks."""

from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .. import api as frozen
from .. import io as frozen_io
from ..io import PROJECT, digest, inputs, read, save

OUT = PROJECT / "benchmark/research_followup/20261004/multimodel_pilot"
OLD = PROJECT / "benchmark/research_followup/20260929/gemini"
MODELS = ("gpt-4o-mini", "deepseek-v4-flash")
ACCEPTED_RETURNED_MODELS = {
    "gpt-4o-mini": ["gpt-4o-mini", "gpt-4o-mini-2024-07-18"],
    "deepseek-v4-flash": ["deepseek-v4-flash"],
}
IDENTITY_SOURCE = {
    "gpt-4o-mini": "https://developers.openai.com/api/docs/models/gpt-4o-mini",
    "deepseek-v4-flash": "https://api.quickrouter.ai/api/pricing",
}


def request(session: dict, messages: list[dict], model: str) -> dict:
    if model not in MODELS:
        raise ValueError("Model must be explicitly authorized")
    payload = frozen.build_request(session, messages)
    payload["model"] = model
    if model == "deepseek-v4-flash":
        payload["thinking"] = {"type": "disabled"}
    return payload


def source_binding(extra: list[Path]) -> dict[str, str]:
    from cst_mil.bench import dataset

    files = (
        list(Path(__file__).parent.glob("*.py"))
        + [Path(frozen.__file__), Path(frozen_io.__file__), Path(dataset.__file__)]
        + list((PROJECT / "benchmark/data/full677").glob("*.*"))
        + extra
    )
    return {str(p.resolve()): digest(p) for p in files}


def verify(binding: dict[str, str]) -> None:
    if any(not Path(p).is_file() or digest(Path(p)) != sha for p, sha in binding.items()):
        raise RuntimeError("Frozen campaign source/input fingerprint changed; stop and review")


def _public(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as stream:
        return json.load(stream)


def quote(model: str, payloads: list[dict]) -> dict:
    pricing = _public("https://api.quickrouter.ai/api/pricing")
    status = _public("https://api.quickrouter.ai/api/status")["data"]
    record = next(x for x in pricing["data"] if x["model_name"] == model)
    if not record.get("available") or "openai" not in record["supported_endpoint_types"]:
        raise ValueError("Requested model unavailable or lacks OpenAI endpoint")
    quota = float(status["quota_per_unit"])
    if status.get("currency_type") != "USD" or quota <= 0 or record["quota_type"] != 0:
        raise ValueError("Unverified public token-pricing conversion")
    groups = []
    for group in record["enable_groups"]:
        inp = 1e6 * record["model_ratio"] * pricing["group_ratio"][group] / quota
        groups.append(
            dict(
                group=group,
                input_usd_per_million=inp,
                output_usd_per_million=inp * record["completion_ratio"],
            )
        )
    highest = max(groups, key=lambda x: x["input_usd_per_million"])
    bound = sum(frozen.input_bound(p) for p in payloads)
    upper = (
        bound * highest["input_usd_per_million"]
        + len(payloads) * 2048 * highest["output_usd_per_million"]
    ) / 1e6
    return dict(
        model=model,
        checked_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        groups=groups,
        highest=highest,
        one_pass_planning_usd=upper,
        two_attempt_planning_usd=2 * upper,
        input_utf8_byte_bound=bound,
        output_token_cap=2048,
        pricing_record=record,
        quota_per_unit=quota,
        currency_type="USD",
        budget_cap_usd=None,
        account_group="unknown",
        account_debit_usd=None,
        cache_policy="No cache discount in upper-rate estimate",
        reasoning_rate="Separate reasoning token price not disclosed",
        estimation=(
            "Public-rate usage estimate; account debit unknown. "
            "UTF-8 bytes are a planning bound, not provider tokenization."
        ),
    )


def prepare(out: Path = OUT) -> tuple[dict, dict, dict]:
    out = out.resolve()
    if out == OLD.resolve() or OLD.resolve() in out.parents:
        raise ValueError("Cannot write inside frozen Gemini directory")
    manifest, sessions, _, grouped = inputs()
    selection = read(OLD / "selection.json")
    if manifest["content_fingerprint"] != selection["data_fingerprint"]:
        raise ValueError("Original Gemini cohort data changed")
    lookup = {s["post_id"]: s for s in sessions}
    if len(selection["sessions"]) != 36 or len({r["post_id"] for r in selection["sessions"]}) != 36:
        raise ValueError("Expected fixed distinct 6-dev/30-test cohort")
    if sum(r["split"] == "dev" for r in selection["sessions"]) != 6:
        raise ValueError("Validation cohort must contain 6 sessions")
    for row in selection["sessions"]:
        s = lookup[row["post_id"]]
        if s["split"] != row["split"] or s["label"] != row["label"]:
            raise ValueError("Frozen split/label changed")
        if frozen.signature(frozen.build_request(s, grouped[s["post_id"]])) != row["prompt_sha256"]:
            raise ValueError("Frozen original prompt/context changed")
    selection_path = out / "selection.json"
    if selection_path.exists() and read(selection_path) != selection:
        raise ValueError("Pilot cohort changed on resume")
    save(selection_path, selection)
    for model in MODELS:
        folder = out / model
        if not (folder / "quote.json").exists():
            payloads = [
                request(lookup[r["post_id"]], grouped[r["post_id"]], model)
                for r in selection["sessions"]
            ]
            save(folder / "quote.json", quote(model, payloads))
    return selection, lookup, grouped


class Ledger:
    """Bound requests even when the authorized monetary cap is None."""

    def __init__(self, folder: Path, cap: float | None = None, max_requests: int = 72):
        if cap is not None and (not math.isfinite(cap) or cap <= 0):
            raise ValueError("Budget cap must be positive or None")
        self.path, self.cap, self.max_requests = folder / "attempt_ledger.json", cap, max_requests
        self.entries = read(self.path) if self.path.exists() else []

    def reserve(self, sid: str, signature: str, bound: float) -> int:
        if len(self.entries) >= self.max_requests:
            raise ValueError("Fixed request limit reached")
        if (
            self.cap is not None
            and sum(e["accounted_usd"] for e in self.entries) + bound > self.cap
        ):
            raise ValueError("Budget insufficient")
        self.entries.append(
            dict(
                session_id=sid,
                request_sha256=signature,
                accounted_usd=bound,
                reservation_usd=bound,
                status="reserved_before_request",
            )
        )
        save(self.path, self.entries)
        return len(self.entries) - 1

    def settle(self, index: int, cost: float | None) -> None:
        entry = self.entries[index]
        entry.update(
            accounted_usd=cost if cost is not None else entry["reservation_usd"],
            cost_estimate_usd=cost,
            status="usage_estimated" if cost is not None else "cost_unknown",
        )
        save(self.path, self.entries)


def cost(response: object, prices: dict) -> float | None:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict) or any(
        type(usage.get(k)) is not int or usage[k] < 0
        for k in ("prompt_tokens", "completion_tokens")
    ):
        return None
    return (
        usage["prompt_tokens"] * prices["input_usd_per_million"]
        + usage["completion_tokens"] * prices["output_usd_per_million"]
    ) / 1e6


def interpret(result: dict, response: object, messages: list[dict], model: str) -> None:
    if not isinstance(response, dict):
        result["status"] = "format_error"
        return
    result["returned_model"] = response.get("model")
    if response.get("model") not in ACCEPTED_RETURNED_MODELS[model]:
        result["status"] = "model_mismatch"
        return
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        result["status"] = "format_error"
        return
    message = choices[0].get("message")
    if not isinstance(message, dict):
        result["status"] = "format_error"
        return
    if message.get("refusal") or choices[0].get("finish_reason") == "content_filter":
        result["status"] = "refusal"
        return
    try:
        # Reuse frozen schema parser without changing its Gemini model identity constant.
        normalized = dict(response, model=frozen.MODEL)
        value = frozen.parse_completion(normalized)
        validation = frozen.validate_evidence(value, messages)
        validation["unsupported_reason_status"] = "requires blind Codex source-support review"
        result.update(status="success", output=value, validation=validation)
    except (ValueError, TypeError, KeyError, AttributeError):
        result["status"] = "format_error"


def call_one(
    session: dict,
    messages: list[dict],
    model: str,
    folder: Path,
    ledger: Ledger,
    prices: dict,
    transport: Callable,
    *,
    wait: Callable = time.sleep,
    before_request: Callable | None = None,
    after_attempt: Callable | None = None,
) -> dict:
    payload = request(session, messages, model)
    sid, signature = session["post_id"], frozen.signature(payload)
    token = hashlib.sha256(sid.encode()).hexdigest()
    dest = folder / "sessions" / f"{token}.json"
    if dest.exists():
        previous = read(dest)
        if previous["request_sha256"] != signature:
            raise ValueError("Resume request fingerprint mismatch")
        return previous
    result = dict(
        session_id=sid,
        split=session["split"],
        requested_model=model,
        request_sha256=signature,
        status="infrastructure_failure",
        attempts=[],
    )
    prior = [(i, e) for i, e in enumerate(ledger.entries) if e["session_id"] == sid]
    for number, (index, entry) in enumerate(prior, 1):
        if entry["request_sha256"] != signature:
            raise ValueError("Resume ledger fingerprint mismatch")
        receipt_path = folder / "receipts" / f"{token}_{number}.json"
        if not receipt_path.exists():
            result["status"] = "interrupted_unknown"
            save(dest, result)
            return result  # Do not repeat an ambiguously delivered paid request.
        receipt = read(receipt_path)
        if receipt["request_sha256"] != signature:
            raise ValueError("Resume receipt fingerprint mismatch")
        result["attempts"].append(receipt["attempt"])
        ledger.settle(index, receipt["attempt"]["cost_estimate_usd"])
        if receipt["response"] is not None:
            interpret(result, receipt["response"], messages, model)
            save(dest, result)
            return result
        result["status"] = receipt["attempt"]["status"]
        if not receipt["attempt"]["retryable"]:
            save(dest, result)
            return result
    bound = (
        frozen.input_bound(payload) * prices["input_usd_per_million"]
        + 2048 * prices["output_usd_per_million"]
    ) / 1e6
    for number in range(len(prior), 2):
        if before_request:
            before_request()
        index = ledger.reserve(sid, signature, bound)
        start = time.perf_counter()
        response, error, http_status, retryable = None, None, None, False
        status = "transport_failure"
        try:
            response = transport(payload)
        except urllib.error.HTTPError as exc:
            http_status, error = exc.code, type(exc).__name__
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            # Error text is classified in memory only; never persist credentials/headers.
            body = exc.read(65536).decode("utf-8", errors="replace").lower()
            if model == "deepseek-v4-flash" and exc.code in (400, 422) and "thinking" in body:
                status = "unsupported_thinking"
            elif retryable:
                status = "infrastructure_failure"
        except (TimeoutError, OSError, urllib.error.URLError) as exc:
            error = type(exc).__name__
            retryable = isinstance(getattr(exc, "reason", exc), TimeoutError)
            status = "infrastructure_failure" if retryable else "transport_failure"
        except (ValueError, UnicodeDecodeError) as exc:
            error = type(exc).__name__
        estimate = cost(response, prices)
        attempt = dict(
            attempt=number + 1,
            ledger_index=index,
            seconds=time.perf_counter() - start,
            cost_estimate_usd=estimate,
            reserved_usd=bound,
            http_status=http_status,
            transport_error=error,
            retryable=retryable,
            status=status,
            usage=response.get("usage") if isinstance(response, dict) else None,
        )
        retained = (
            {k: response.get(k) for k in ("model", "usage", "choices", "id")}
            if isinstance(response, dict)
            else response
        )
        save(
            folder / "receipts" / f"{token}_{number + 1}.json",
            dict(session_id=sid, request_sha256=signature, attempt=attempt, response=retained),
        )
        result["attempts"].append(attempt)
        ledger.settle(index, estimate)
        stop = after_attempt(attempt) if after_attempt else False
        if response is not None:
            interpret(result, response, messages, model)
            break
        result["status"] = status
        if stop or not retryable:
            break
        if number == 0:
            wait(1)
    save(dest, result)
    return result


class NoCredentialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


def run_model(model: str, out: Path = OUT, transport: Callable | None = None) -> dict:
    if model not in MODELS:
        raise ValueError("Unauthorized model")
    selection, lookup, grouped = prepare(out)
    folder = out / model
    prices = read(folder / "quote.json")["highest"]
    binding = source_binding(
        [out / "selection.json", folder / "quote.json", OLD / "selection.json"]
    )
    config = dict(
        model=model,
        accepted_returned_models=ACCEPTED_RETURNED_MODELS[model],
        identity_source=IDENTITY_SOURCE[model],
        budget_cap_usd=None,
        max_requests=72,
        threshold=0.5,
        temperature=0,
        max_tokens=2048,
        max_attempts=2,
        thinking={"type": "disabled"} if model == "deepseek-v4-flash" else None,
        prompt_sha256=hashlib.sha256(frozen.SYSTEM.encode()).hexdigest(),
        sources=binding,
    )
    path = folder / "run_config.json"
    if path.exists() and read(path) != config:
        raise ValueError("Frozen campaign fingerprint changed")
    save(path, config)
    ledger = Ledger(folder)
    if transport is None:
        key = frozen.read_credentials()
        opener = urllib.request.build_opener(NoCredentialRedirect())

        def transport(payload):
            req = urllib.request.Request(
                frozen.BASE + "/chat/completions",
                method="POST",
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            )
            with opener.open(req, timeout=120) as stream:
                return json.load(stream)

    state = {"consecutive_infrastructure_failures": 0}
    receipts = sorted(
        (folder / "receipts").glob("*.json"), key=lambda p: read(p)["attempt"]["ledger_index"]
    )

    def after(attempt):
        state["consecutive_infrastructure_failures"] = (
            state["consecutive_infrastructure_failures"] + 1 if attempt["retryable"] else 0
        )
        return state["consecutive_infrastructure_failures"] >= 3

    for receipt in receipts:
        after(read(receipt)["attempt"])
    stop_reason = None
    results = []
    for row in selection["sessions"]:
        if state["consecutive_infrastructure_failures"] >= 3:
            stop_reason = "three_consecutive_infrastructure_failures"
            break
        if row["split"] == "test" and not read(folder / "validation_gate.json")["passed"]:
            stop_reason = "validation_format_gate_failed"
            break
        s = lookup[row["post_id"]]
        r = call_one(
            s,
            grouped[s["post_id"]],
            model,
            folder,
            ledger,
            prices,
            transport,
            before_request=lambda: verify(binding),
            after_attempt=after,
        )
        results.append(r)
        print(
            json.dumps(
                dict(model=model, completed=len(results), split=r["split"], status=r["status"])
            ),
            flush=True,
        )
        if r["status"] in (
            "model_mismatch",
            "unsupported_thinking",
            "transport_failure",
            "interrupted_unknown",
        ):
            stop_reason = r["status"]
            break
        dev = [r for r in results if r["split"] == "dev"]
        if len(dev) == 6:
            gate = dict(
                passed=all(r["status"] == "success" for r in dev),
                n_dev=6,
                prompt_sha256=config["prompt_sha256"],
                threshold=0.5,
            )
            gate_path = folder / "validation_gate.json"
            if gate_path.exists() and read(gate_path) != gate:
                raise ValueError("Frozen validation gate changed")
            save(gate_path, gate)
            if not gate["passed"]:
                stop_reason = "validation_format_gate_failed"
                break
    verify(binding)
    results = [
        read(folder / "sessions" / f"{hashlib.sha256(r['post_id'].encode()).hexdigest()}.json")
        for r in selection["sessions"]
        if (
            folder / "sessions" / f"{hashlib.sha256(r['post_id'].encode()).hexdigest()}.json"
        ).exists()
    ]
    summary = dict(
        model=model,
        completed=len(results),
        attempts=len(ledger.entries),
        status="complete" if len(results) == 36 else "stopped",
        stop_reason=stop_reason,
        account_debit_usd=None,
        results=results,
        **state,
    )
    save(folder / "summary.json", summary)
    return summary
