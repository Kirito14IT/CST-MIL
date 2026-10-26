"""Read-only CST-MIL inference. Stdout is reserved for one JSON object per line."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback
import unicodedata
from time import perf_counter

PROTOCOL = sys.stdout
APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
RESEARCH_ROOT = APP_ROOT.parent / "cst_mil"
ALLOWED = {1: {("cst_mil.bench.model", "BenchmarkModel")},
           2: {("cst_mil.unified_semantic_v1_1.model", "UnifiedSemanticModel"),
               ("cst_mil.demo_semantic_adapter", "FollowupPackage"),
               ("cst_mil.deployment.adapter", "FollowupPackage")}}


def emit(value: dict) -> None:
    PROTOCOL.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    PROTOCOL.flush()


def repo_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Deployment paths must be relative to the repository")
    resolved = (REPO_ROOT / relative).resolve(strict=True)
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("Deployment paths must stay inside the repository")
    return resolved


def warm_encoder(model) -> None:
    """Load the process-shared encoder using its public API, without risk heads."""
    model.encoder().encode([""])


def validate_topic_grounding(result: dict, messages: list[dict], is_generic) -> None:
    """Check schema2 extractive names against untouched input codepoint offsets.

    The callable is the generic-word rule from the manifest-bound topic module.
    No result or source text is normalized in place.
    """
    originals = {message["comment_id"]: message["raw_text"] for message in messages}
    topics = result.get("topics")
    if not isinstance(topics, list):
        raise ValueError("Invalid topic list format")
    for topic in topics:
        if not isinstance(topic, dict):
            raise ValueError("Invalid topic structure")
        keywords = topic.get("keywords")
        if (not isinstance(keywords, list) or not keywords
                or any(not isinstance(word, str) or not word for word in keywords)
                or len(set(keywords)) != len(keywords)):
            raise ValueError("Topic keywords are missing or duplicated")
        if topic.get("title") != " / ".join(keywords[:2]):
            raise ValueError("Topic title must use the first two keywords")
        members = topic.get("message_ids")
        if (not isinstance(members, list) or not members
                or any(not isinstance(mid, str) or mid not in originals for mid in members)
                or len(set(members)) != len(members)):
            raise ValueError("Invalid topic message IDs")
        member_ids = set(members)
        locations = topic.get("support_messages")
        if not isinstance(locations, list):
            raise ValueError("Invalid topic source support format")
        for location in locations:
            if (not isinstance(location, dict) or location.get("comment_id") not in member_ids
                    or location.get("raw_text") != originals[location["comment_id"]]):
                raise ValueError("Topic source support is outside the topic or differs from the input")
        supports = topic.get("keyword_supports")
        if not isinstance(supports, dict) or set(supports) != set(keywords):
            raise ValueError("Incomplete keyword source support")
        for word in keywords:
            if (word != unicodedata.normalize("NFC", word).casefold()
                    or is_generic(word) or len(word.replace(" ", "")) < 2):
                raise ValueError("Topic keywords are not normalized or contain only generic terms")
            entries = supports[word]
            if not isinstance(entries, list) or not entries:
                raise ValueError("Keyword has no source support")
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("message_id") not in member_ids:
                    raise ValueError("Keyword support message does not belong to the topic")
                raw = originals[entry["message_id"]]
                start, end, excerpt = entry.get("start"), entry.get("end"), entry.get("excerpt")
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(raw)
                        or not isinstance(excerpt, str) or raw[start:end] != excerpt
                        or unicodedata.normalize("NFC", excerpt).casefold() != word):
                    raise ValueError("Keyword offsets or excerpt differ from the source text")


def validate_topic_associations(result: dict, messages: list[dict]) -> None:
    """Validate disjoint basic groups and one-hop associations without score inheritance."""
    originals = {message["comment_id"]: message for message in messages}
    scored = {message["comment_id"]: message for message in result["message_scores"]}
    memberships: dict[str, int] = {}
    for topic in result["topics"]:
        basic, linked = topic.get("basic_message_ids"), topic.get("context_association_message_ids")
        if (not isinstance(basic, list) or len(basic) < 2 or not isinstance(linked, list)
                or len(set(basic + linked)) != len(basic + linked)
                or set(basic + linked) != set(topic["message_ids"])):
            raise ValueError("Core and parent-linked topic members must be complete and disjoint")
        for mid in basic + linked:
            if mid not in originals or mid in memberships:
                raise ValueError("Topic member is missing or assigned more than once")
            memberships[mid] = topic["topic_id"]
            message = scored[mid]
            if message["topic_ids"] != [topic["topic_id"]]:
                raise ValueError("Message topic assignment differs from the member list")
            if mid in basic and message.get("assignment_provenance") != "semantic_cluster":
                raise ValueError("Core member is missing semantic grouping provenance")
            if mid in linked:
                parent = originals[mid].get("parent_id")
                if (parent not in basic or message.get("context_parent_message_id") != parent
                        or message.get("assignment_provenance") != "direct_parent_context"
                        or len("".join(originals[mid]["raw_text"].split())) > 12
                        or message["topic_strengths"][topic["topic_id"]] != 0):
                    raise ValueError("Linked members require a real direct parent; cascading links or fabricated semantic strength are invalid")
        for entries in topic["keyword_supports"].values():
            if any(entry["message_id"] not in basic for entry in entries):
                raise ValueError("Parent-linked members must not contribute to topic naming statistics")
    for mid, message in scored.items():
        if (message["assigned"] != (mid in memberships)
                or (mid not in memberships and message["topic_ids"])):
            raise ValueError("Topic coverage differs from the member list")


def load(manifest_path: Path):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8-sig"))
    schema = manifest.get("schemaVersion")
    if (manifest.get("manifestVersion") != 1 or schema not in ALLOWED
            or (manifest.get("predictorModule"), manifest.get("modelClass")) not in ALLOWED[schema]
            or manifest.get("predictFunction") != "predict_session"):
        raise ValueError("Unsupported deployment manifest interface")
    model_path = repo_path(manifest["modelPath"])
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != manifest["modelSha256"]:
        raise ValueError("Model fingerprint mismatch; loading stopped")
    fingerprints = manifest.get("codeFingerprints")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("Missing deployment source fingerprints")
    for relative, expected in fingerprints.items():
        if hashlib.sha256(repo_path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Deployment source fingerprint mismatch: {relative}")
    for relative, expected in manifest.get("assetFingerprints", {}).items():
        if hashlib.sha256(repo_path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Deployment asset fingerprint mismatch: {relative}")
    sys.path.insert(0, str(RESEARCH_ROOT / "src"))
    from cst_mil.bench.resources import configure_cpu
    policy = configure_cpu(4)
    # Numerical libraries are imported only after process affinity and thread hints.
    import joblib
    module = importlib.import_module(manifest["predictorModule"])
    if manifest["predictorModule"] == "cst_mil.deployment.adapter":
        # joblib is executable serialization. A caller-supplied matching hash is
        # insufficient: authenticate the public release before deserialization,
        # including direct worker launches which do not pass through Node.
        from cst_mil.deployment.assets import validate_manifest
        validate_manifest(REPO_ROOT, manifest)
    model = joblib.load(model_path)
    if not isinstance(model, getattr(module, manifest["modelClass"])):
        raise RuntimeError("Model type mismatch")
    if schema == 1:
        if model.method != "cst_mil" or model.threshold != manifest["thresholds"]["attack"] or model.threshold != manifest["conversationThreshold"]:
            raise ValueError("Legacy model thresholds differ from the deployment manifest")
    elif manifest["modelClass"] == "FollowupPackage":
        binding = manifest.get("bundle", {})
        for key in ("manifestPath", "explanationPath"):
            if binding.get(key) not in manifest.get("assetFingerprints", {}):
                raise ValueError("Semantic deployment is missing the frozen model bundle fingerprint")
        if manifest.get("encoder", {}).get("manifestPath") not in manifest.get("assetFingerprints", {}):
            raise ValueError("Semantic deployment is missing the encoder manifest fingerprint")
        module.validate_package(model, manifest, REPO_ROOT)
        warm_encoder(model)
    else:
        if model.model_version != manifest["modelVersion"] or model.thresholds != manifest["thresholds"] or model.conversation_threshold != manifest["conversationThreshold"]:
            raise ValueError("Semantic model version or thresholds differ from the deployment manifest")
        encoder = manifest["encoder"]
        if Path(model.encoder_path).resolve() != repo_path(encoder["path"]) or model.encoder_fingerprint != encoder["fingerprint"]:
            raise ValueError("Model encoder identity differs from the deployment manifest")
        # The research model verifies exact inventory, byte sizes and SHA256 once.
        model.validate_assets()
        warm_encoder(model)
    identity = {"modelHash": actual_hash, "modelVersion": manifest["modelVersion"],
                "deploymentHash": hashlib.sha256(raw).hexdigest(),
                "schemaVersion": schema, "taskName": manifest["taskName"]}
    return model, getattr(module, manifest["predictFunction"]), policy, manifest, identity


def convert(session: dict) -> tuple[dict, list[dict]]:
    # Whitelist fields. Dataset gold labels never enter the prediction function.
    sid = session["id"]
    rows = []
    for index, message in enumerate(session["messages"]):
        rows.append({
            "post_id": sid,
            "comment_id": message["id"],
            "raw_text": message["text"],
            "order": message.get("order", index),
            "parent_id": message.get("parentId"),
            "speaker_id": message.get("speaker"),
            "comment_time": message.get("time"),
        })
    if session.get("includePost"):
        existing = {row["comment_id"] for row in rows}
        root_id, suffix = "__post__", 0
        while root_id in existing:
            suffix += 1
            root_id = f"__post___{suffix}"
        rows.insert(0, {
            "post_id": sid, "comment_id": root_id, "raw_text": session["postText"],
            "order": min((row["order"] for row in rows), default=0) - 1,
            "parent_id": None, "speaker_id": "Original post", "comment_time": None,
        })
    if not rows:
        raise ValueError("No messages to analyze")
    return {"post_id": sid, "split": "inference"}, rows


def main() -> int:
    startup_started = perf_counter()
    try:
        manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else APP_ROOT / "config/deployment.json"
        with contextlib.redirect_stdout(sys.stderr):
            model, predict, policy, manifest, identity = load(manifest_path)
            if identity["schemaVersion"] == 2:
                topic_module = (importlib.import_module(manifest["predictorModule"]).topics
                                if manifest["modelClass"] == "FollowupPackage"
                                else importlib.import_module(manifest["predictorModule"].rsplit(".", 1)[0] + ".topics"))
                generic_word = (topic_module.legacy.generic_word if manifest["modelClass"] == "FollowupPackage"
                                else topic_module.generic_word)
        emit({"type": "ready", **identity, "startupSeconds": perf_counter() - startup_started,
              "encoderWarmed": identity["schemaVersion"] == 2, "thresholds": manifest["thresholds"],
              "conversationThreshold": manifest["conversationThreshold"],
              **({"threshold": model.threshold} if identity["schemaVersion"] == 1 else {}), "policy": policy})
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit({"type": "fatal", "error": f"Model startup failed: {type(error).__name__}: {error}"})
        return 1
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            if request.get("type") == "shutdown":
                return 0
            request_id = request["id"]
            with contextlib.redirect_stdout(sys.stderr):
                session, messages = convert(request["session"])
                result = predict(model, session, messages)
                if identity["schemaVersion"] == 2:
                    if result.get("schemaVersion") != 2 or result.get("modelVersion") != identity["modelVersion"]:
                        raise ValueError("Semantic prediction identity mismatch")
                    validate_topic_grounding(result, messages, generic_word)
                    if manifest["modelClass"] == "FollowupPackage":
                        validate_topic_associations(result, messages)
                else:
                    result = {**result, "schemaVersion": 1}
            emit({"type": "result", "id": request_id, **identity, "result": result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({"type": "error", "id": request_id,
                  "error": f"Inference failed: {type(error).__name__}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
