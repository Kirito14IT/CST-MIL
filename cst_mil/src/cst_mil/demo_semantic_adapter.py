"""Read-only demonstration adapter for the frozen full-data FollowupPackage.

The browser schema is separate from the experiment's prediction tables. Original
message text, IDs and reply edges are preserved. One semantic encoding is shared
by risk prediction and interpretation; this module never trains a model.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from cst_mil.bench.interpretation import message_location
from cst_mil.research_followup.semantic_patent_followup.model import FollowupPackage
from cst_mil.research_followup.semantic_patent_followup import topics
from cst_mil.unified_semantic_v1_1 import CATEGORY_IDS, CATEGORY_NAMES
from cst_mil.unified_semantic_v1_1.encoder import ENCODING_VERSION
from cst_mil.unified_semantic_v1_1.model import canonical_messages


def validate_package(model: FollowupPackage, manifest: dict[str, Any], repo_root: Path) -> None:
    """Verify the adapter's explicit package and explanation contract before use."""
    binding = manifest.get("bundle")
    if not isinstance(binding, dict):
        raise ValueError("全量语义模型缺少研究包绑定")
    source = json.loads((repo_root / binding["manifestPath"]).read_text(encoding="utf-8-sig"))
    explanation = json.loads((repo_root / binding["explanationPath"]).read_text(encoding="utf-8-sig"))
    if (model.seed != 42 or model.config.architecture_base != "full" or model.config.ablation is not None
            or model.method != "semantic_patent_followup"
            or model.model_version != "semantic-patent-followup-v1"
            or model.model_version != manifest["modelVersion"]
            or source["seed"] != model.seed or source["architecture"] != model.config.variant
            or source["model_sha256"] != manifest["modelSha256"]):
        raise ValueError("全量语义模型身份或预定配置不一致")
    if (model.thresholds != manifest["thresholds"]
            or {key: source["thresholds"][key] for key in CATEGORY_IDS} != model.thresholds
            or source["thresholds"]["conversation"] != model.conversation_threshold
            or manifest["conversationThreshold"] != model.conversation_threshold
            or source["thresholds"]["conversation_common16"] != model.common16_threshold):
        raise ValueError("全量语义模型阈值与冻结研究包不一致")
    if (source["categories"] != list(CATEGORY_IDS)
            or source["training_sources"] != manifest["trainingSources"]
            or explanation != source["explanation_config"]
            or explanation["distance_threshold"] != 0.35
            or explanation["link_short_reply_max_nonspace"] != 12
            or explanation["protected_clause_max_nonspace"] != 96
            or explanation["inherit_parent_risk"] is not False
            or explanation["association_contributes_to_names"] is not False
            or explanation["pure_responses"] != sorted(topics.PURE_RESPONSES)):
        raise ValueError("全量语义解释配置与研究包不一致")
    encoder = manifest["encoder"]
    metadata = model.encoder_metadata
    source_encoder = source["encoder"]
    # The pickle additionally retains TRAIN/DEV cost measurements. They are not
    # encoder identity fields and the published bundle intentionally omits them.
    cost_metadata_keys = {"encoding_seconds_by_split", "encoding_peak_tree_rss_bytes"}
    if (Path(metadata["encoder_path"]).resolve() != (repo_root / encoder["path"]).resolve()
            or metadata["fingerprint"] != encoder["fingerprint"]
            or any(metadata.get(key) != expected for key, expected in source_encoder.items())
            or set(metadata) - set(source_encoder) - cost_metadata_keys
            or metadata["encoding_version"] != ENCODING_VERSION):
        raise ValueError("全量语义模型编码器身份不一致")
    model.encoder().validate_assets()


def inference_records(session: dict[str, Any], messages: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Use SCCD's conversation scorer without introducing source metadata as features."""
    current = canonical_messages(session, messages)
    sid = str(session.get("post_id", session.get("session_id", "session")))
    records = [{
        "record_id": f"demo:{sid}:{message['comment_id']}",
        "message_id": message["comment_id"], "text": message["raw_text"],
        "source": "sccd", "split": "inference", "session_id": sid,
        "order": message["order"], "parent_id": message["parent_id"],
        "labels": [None, None, None], "observed_mask": [False, False, False],
        "fit_eligible": False,
    } for message in current]
    return current, records


def predict_session(model: FollowupPackage, session: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    current, records = inference_records(session, messages)
    sid = str(session.get("post_id", session.get("session_id", "session")))
    started = perf_counter()
    embeddings = model.encode_records(records)
    encoded = perf_counter()
    payload = model.predict_payload(records, [{"session_id": sid, "split": "inference"}], embeddings=embeddings)
    predicted = perf_counter()
    scores = np.asarray([row["scores"] for row in payload["message"]], dtype=float)
    if scores.shape != (len(current), 3) or len(payload["session"]) != 1:
        raise ValueError("全量语义研究预测结果维度无效")
    interpretation = topics.interpret_topics(current, embeddings, scores, category_thresholds=model.thresholds,
                                            variant="full", distance_threshold=0.35, evidence_top_k=3)
    interpreted = perf_counter()
    for topic in interpretation["topics"]:
        topic["basic_message_ids"] = list(topic["semantic_message_ids"])
        topic["context_association_message_ids"] = list(topic["context_linked_message_ids"])
    categories = []
    for j, (category, name) in enumerate(zip(CATEGORY_IDS, CATEGORY_NAMES, strict=True)):
        threshold = float(model.thresholds[category])
        selected = [i for i in range(len(current)) if scores[i, j] >= threshold]
        selected.sort(key=lambda i: (-scores[i, j], i))
        categories.append({
            "id": category, "name": name, "score": float(scores[:, j].max()),
            "threshold": threshold, "detected": bool(selected),
            "message_ids": [current[i]["comment_id"] for i in selected],
            "evidence": [{**message_location(current[i], i), "risk_score": float(scores[i, j])}
                         for i in selected[:3]],
        })
    for i, message in enumerate(interpretation["message_scores"]):
        message["riskCategoryScores"] = {category: float(scores[i, j]) for j, category in enumerate(CATEGORY_IDS)}
    conversation = payload["session"][0]
    detected = any(category["detected"] for category in categories)
    return {
        "schemaVersion": 2, "post_id": sid, "split": session.get("split", "inference"), "label": None,
        "prediction": int(detected), "prediction_name": "发现风险线索" if detected else "未发现明确风险线索",
        "overallEvidenceScore": float(scores.max()), "scoreType": "max_message_category_score",
        "riskCategories": categories,
        "conversationRisk": {"score": float(conversation["score"]),
                             "threshold": float(model.conversation_threshold),
                             "prediction": int(conversation["score"] >= model.conversation_threshold),
                             "domain": "SCCD", "note": "SCCD网络欺凌会话评分；校园与其他领域仅作迁移参考，非校准概率。"},
        "modelVersion": model.model_version,
        "inferenceTrace": {"semanticEncodingPasses": 1, "encodingSeconds": encoded - started,
                           "riskPredictionSeconds": predicted - encoded,
                           "interpretationSeconds": interpreted - predicted},
        **interpretation,
    }
