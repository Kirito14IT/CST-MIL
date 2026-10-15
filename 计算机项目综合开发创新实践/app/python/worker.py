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
REPO_ROOT = APP_ROOT.parent.parent
RESEARCH_ROOT = APP_ROOT.parent.parent / "cst_mil"
ALLOWED = {1: {("cst_mil.bench.model", "BenchmarkModel")},
           2: {("cst_mil.unified_semantic_v1_1.model", "UnifiedSemanticModel"),
               ("cst_mil.demo_semantic_adapter", "FollowupPackage"),
               ("cst_mil.deployment.adapter", "FollowupPackage")}}


def emit(value: dict) -> None:
    PROTOCOL.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    PROTOCOL.flush()


def repo_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("部署路径必须相对仓库")
    resolved = (REPO_ROOT / relative).resolve(strict=True)
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("部署路径不能越出仓库")
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
        raise ValueError("话题列表格式无效")
    for topic in topics:
        if not isinstance(topic, dict):
            raise ValueError("话题结构无效")
        keywords = topic.get("keywords")
        if (not isinstance(keywords, list) or not keywords
                or any(not isinstance(word, str) or not word for word in keywords)
                or len(set(keywords)) != len(keywords)):
            raise ValueError("话题关键词缺失或重复")
        if topic.get("title") != " / ".join(keywords[:2]):
            raise ValueError("话题名称必须来自前两个关键词")
        members = topic.get("message_ids")
        if (not isinstance(members, list) or not members
                or any(not isinstance(mid, str) or mid not in originals for mid in members)
                or len(set(members)) != len(members)):
            raise ValueError("话题消息编号无效")
        member_ids = set(members)
        locations = topic.get("support_messages")
        if not isinstance(locations, list):
            raise ValueError("话题原文支持格式无效")
        for location in locations:
            if (not isinstance(location, dict) or location.get("comment_id") not in member_ids
                    or location.get("raw_text") != originals[location["comment_id"]]):
                raise ValueError("话题原文支持不属于本话题或不匹配输入")
        supports = topic.get("keyword_supports")
        if not isinstance(supports, dict) or set(supports) != set(keywords):
            raise ValueError("关键词原文支持集合不完整")
        for word in keywords:
            if (word != unicodedata.normalize("NFC", word).casefold()
                    or is_generic(word) or len(word.replace(" ", "")) < 2):
                raise ValueError("话题关键词未规范化或只有泛词")
            entries = supports[word]
            if not isinstance(entries, list) or not entries:
                raise ValueError("关键词没有原文支持")
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("message_id") not in member_ids:
                    raise ValueError("关键词支持消息不属于本话题")
                raw = originals[entry["message_id"]]
                start, end, excerpt = entry.get("start"), entry.get("end"), entry.get("excerpt")
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(raw)
                        or not isinstance(excerpt, str) or raw[start:end] != excerpt
                        or unicodedata.normalize("NFC", excerpt).casefold() != word):
                    raise ValueError("关键词字符位置或摘录与原文不一致")


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
            raise ValueError("主题基本成员与父回复关联成员必须完整且互斥")
        for mid in basic + linked:
            if mid not in originals or mid in memberships:
                raise ValueError("主题成员不存在或重复归属")
            memberships[mid] = topic["topic_id"]
            message = scored[mid]
            if message["topic_ids"] != [topic["topic_id"]]:
                raise ValueError("消息主题归属与成员清单不一致")
            if mid in basic and message.get("assignment_provenance") != "semantic_cluster":
                raise ValueError("基本成员没有语义分组来源")
            if mid in linked:
                parent = originals[mid].get("parent_id")
                if (parent not in basic or message.get("context_parent_message_id") != parent
                        or message.get("assignment_provenance") != "direct_parent_context"
                        or len("".join(originals[mid]["raw_text"].split())) > 12
                        or message["topic_strengths"][topic["topic_id"]] != 0):
                    raise ValueError("关联成员必须来自真实直接父消息，不能级联或伪造语义强度")
        for entries in topic["keyword_supports"].values():
            if any(entry["message_id"] not in basic for entry in entries):
                raise ValueError("父回复关联成员不能参与主题命名统计")
    for mid, message in scored.items():
        if (message["assigned"] != (mid in memberships)
                or (mid not in memberships and message["topic_ids"])):
            raise ValueError("主题覆盖与成员清单不一致")


def load(manifest_path: Path):
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8-sig"))
    schema = manifest.get("schemaVersion")
    if (manifest.get("manifestVersion") != 1 or schema not in ALLOWED
            or (manifest.get("predictorModule"), manifest.get("modelClass")) not in ALLOWED[schema]
            or manifest.get("predictFunction") != "predict_session"):
        raise ValueError("部署清单接口不受支持")
    model_path = repo_path(manifest["modelPath"])
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_hash != manifest["modelSha256"]:
        raise ValueError("模型指纹不匹配，停止加载")
    fingerprints = manifest.get("codeFingerprints")
    if not isinstance(fingerprints, dict) or not fingerprints:
        raise ValueError("缺少部署源码指纹")
    for relative, expected in fingerprints.items():
        if hashlib.sha256(repo_path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"部署源码指纹不匹配：{relative}")
    for relative, expected in manifest.get("assetFingerprints", {}).items():
        if hashlib.sha256(repo_path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"部署资产指纹不匹配：{relative}")
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
        raise RuntimeError("正式模型类型不匹配")
    if schema == 1:
        if model.method != "cst_mil" or model.threshold != manifest["thresholds"]["attack"] or model.threshold != manifest["conversationThreshold"]:
            raise ValueError("旧模型阈值与部署清单不一致")
    elif manifest["modelClass"] == "FollowupPackage":
        binding = manifest.get("bundle", {})
        for key in ("manifestPath", "explanationPath"):
            if binding.get(key) not in manifest.get("assetFingerprints", {}):
                raise ValueError("全量语义部署缺少冻结研究包指纹")
        if manifest.get("encoder", {}).get("manifestPath") not in manifest.get("assetFingerprints", {}):
            raise ValueError("全量语义部署缺少编码器清单指纹")
        module.validate_package(model, manifest, REPO_ROOT)
        warm_encoder(model)
    else:
        if model.model_version != manifest["modelVersion"] or model.thresholds != manifest["thresholds"] or model.conversation_threshold != manifest["conversationThreshold"]:
            raise ValueError("新模型版本或阈值与部署清单不一致")
        encoder = manifest["encoder"]
        if Path(model.encoder_path).resolve() != repo_path(encoder["path"]) or model.encoder_fingerprint != encoder["fingerprint"]:
            raise ValueError("模型编码器身份与部署清单不一致")
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
            "parent_id": None, "speaker_id": "原帖", "comment_time": None,
        })
    if not rows:
        raise ValueError("没有可分析的消息")
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
        emit({"type": "fatal", "error": f"模型启动失败：{type(error).__name__}: {error}"})
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
                        raise ValueError("新模型预测身份不一致")
                    validate_topic_grounding(result, messages, generic_word)
                    if manifest["modelClass"] == "FollowupPackage":
                        validate_topic_associations(result, messages)
                else:
                    result = {**result, "schemaVersion": 1}
            emit({"type": "result", "id": request_id, **identity, "result": result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({"type": "error", "id": request_id,
                  "error": f"推理失败：{type(error).__name__}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
