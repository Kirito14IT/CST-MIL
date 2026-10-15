"""Machine grounding checks and blinded model-assisted review preparation.

No function here generates human labels or a theme-correctness accuracy. Review
records are qualitative development evidence, with both reviewers retained.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from cst_mil.bench.interpretation import message_location
from cst_mil.unified_semantic_v1_1.topics import generic_word

REVIEW_DIMENSIONS = (
    "name_faithfulness",
    "group_consistency",
    "coverage_appropriateness",
    "stance_separation",
)


def _write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normal(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def machine_quality_audit(
    messages: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    baseline: dict | None = None,
) -> dict[str, Any]:
    """Check names/offsets/IDs/partition/provenance; not semantic correctness."""
    locations = [message_location(message, i) for i, message in enumerate(messages)]
    lookup = {item["comment_id"]: item for item in locations}
    violations = []
    if len(lookup) != len(locations):
        violations.append({"kind": "duplicate_input_id"})
    group_sets, member_counts = [], Counter()
    score_lookup = {}
    for item in result.get("message_scores", []):
        message_id = str(item.get("comment_id"))
        if message_id not in lookup or message_id in score_lookup:
            violations.append({"kind": "invalid_or_duplicate_score_id", "message_id": message_id})
        score_lookup[message_id] = item
    if set(score_lookup) != set(lookup):
        violations.append({"kind": "incomplete_message_scores"})
    supported_keywords = total_keywords = linked = 0
    for topic in result.get("topics", []):
        ids = [str(value) for value in topic.get("message_ids", [])]
        members = frozenset(ids)
        if not ids or len(ids) != len(members) or not members.issubset(lookup):
            violations.append({"kind": "invalid_topic_ids", "topic_id": topic.get("topic_id")})
        if members in group_sets:
            violations.append({"kind": "duplicate_member_group", "topic_id": topic.get("topic_id")})
        group_sets.append(members)
        member_counts.update(ids)
        semantic = set(map(str, topic.get("semantic_message_ids", ids)))
        name_members = set(map(str, topic.get("name_statistics_message_ids", ids)))
        contextual = set(map(str, topic.get("context_linked_message_ids", [])))
        if semantic & contextual or semantic | contextual != set(ids) or name_members != semantic:
            violations.append(
                {"kind": "invalid_member_provenance", "topic_id": topic.get("topic_id")}
            )
        supports = topic.get("keyword_supports", {})
        keywords = topic.get("keywords", [])
        if topic.get("title") != " / ".join(keywords[:2]) or not keywords:
            violations.append(
                {"kind": "unsupported_title_structure", "topic_id": topic.get("topic_id")}
            )
        for word in keywords:
            total_keywords += 1
            if generic_word(word):
                violations.append({"kind": "generic_keyword", "keyword": word})
            support = supports.get(word, [])
            valid = bool(support)
            for item in support:
                mid = str(item.get("message_id"))
                raw = lookup.get(mid, {}).get("raw_text", "")
                start, end = item.get("start"), item.get("end")
                exact = (
                    mid in semantic
                    and isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start < end <= len(raw)
                    and raw[start:end] == item.get("excerpt")
                    and _normal(raw[start:end]) == _normal(word)
                )
                if not exact:
                    valid = False
                    violations.append(
                        {"kind": "invalid_original_span", "message_id": mid, "keyword": word}
                    )
            if valid:
                supported_keywords += 1
            else:
                violations.append({"kind": "unsupported_keyword", "keyword": word})
        for mid in contextual:
            linked += 1
            item = score_lookup.get(mid, {})
            parent = str(item.get("context_parent_message_id"))
            if (
                parent not in semantic
                or str(lookup.get(mid, {}).get("parent_id")) != parent
                or lookup.get(mid, {}).get("post_id") != lookup.get(parent, {}).get("post_id")
                or item.get("assignment_provenance") != "direct_parent_context"
            ):
                violations.append({"kind": "invalid_context_link", "message_id": mid})
        for item in topic.get("evidence", []):
            if str(item.get("comment_id")) not in members:
                violations.append({"kind": "invalid_topic_evidence_id"})
    if any(count > 1 for count in member_counts.values()):
        violations.append({"kind": "overlapping_hard_topic_membership"})
    for item in result.get("evidence", []):
        mid = str(item.get("comment_id"))
        if mid not in lookup or item.get("raw_text") != lookup[mid]["raw_text"]:
            violations.append({"kind": "invalid_global_evidence"})
    fraction = len(member_counts) / len(messages) if messages else 0.0
    if abs(fraction - result.get("coverage", -1)) > 1e-12:
        violations.append({"kind": "inconsistent_coverage"})
    drop = None if baseline is None else float(baseline["coverage"]) - fraction
    if baseline is not None and baseline.get("topics") and not result.get("topics"):
        violations.append({"kind": "all_topics_abstained"})
    if drop is not None and drop > 0.05 + 1e-12:
        violations.append({"kind": "coverage_drop_over_five_percentage_points", "drop": drop})
    return {
        "passed": not violations,
        "violations": violations,
        "messages": len(messages),
        "topics": len(group_sets),
        "assigned_messages": len(member_counts),
        "coverage": fraction,
        "linked_messages": linked,
        "keyword_count": total_keywords,
        "supported_keyword_count": supported_keywords,
        "original_support_fraction": (
            supported_keywords / total_keywords if total_keywords else None
        ),
        "coverage_drop_from_frozen": drop,
        "semantic_correctness": "not_measured_by_machine_grounding_checks",
    }


REVIEW_INSTRUCTIONS = """你是独立的模型辅助审阅者，不是人工金标准标注员。
仅依据本文件提供的原始消息、真实回复编号和匿名输出判断，不读取另一评审文件或身份映射。
每个 case_id × anonymous_configuration 分别记录四项0/1/2评分：
name_faithfulness：0=明显错义或反转否定/疑问，1=有原文但不够恰当，2=忠实且能概括；
group_consistency：0=明显混合无关讨论或相反立场，1=部分混杂，2=组内连贯；
coverage_appropriateness：0=遗漏明显或强行分组严重，1=仍有可改进，2=覆盖与未分组处理恰当；
stance_separation：0=将攻击与引用/劝阻明显混为一谈，1=存在疑点，2=区分恰当。
主题缺失时名称项填null，不能自动给2分；不存在可评价立场时立场项填null。
记录具体 message_ids、topic_ids 和简短原文依据；不根据风险分数替代语义判断。
不宣称主题准确率、人工一致率或专利创造性。输出逐例JSON记录，不生成总体结论。
"""


def export_review_packets(
    cases: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Export two separated anonymous review sets and an audit-only identity key.

    Each case has case_id, messages, and variants={internal_variant: result}.
    Labels and other case metadata are deliberately excluded from review files.
    """
    destination = Path(output_dir)
    case_ids = [str(case["case_id"]) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate review case IDs")
    variants = sorted({variant for case in cases for variant in case["variants"]})
    mapping = {}
    paths = []
    for reviewer in (1, 2):
        rng = random.Random(seed + reviewer * 1009)
        aliases = list(range(101, 101 + len(variants)))
        rng.shuffle(aliases)
        key = {variant: f"方案{alias}" for variant, alias in zip(variants, aliases, strict=True)}
        mapping[f"reviewer_{reviewer}"] = key
        order = list(cases)
        rng.shuffle(order)
        directory = destination / f"reviewer_{reviewer}"
        index = []
        for index_number, case in enumerate(order, 1):
            messages = [message_location(message, i) for i, message in enumerate(case["messages"])]
            outputs = []
            order_variants = list(case["variants"])
            rng.shuffle(order_variants)
            for variant in order_variants:
                result = case["variants"][variant]
                # Remove provenance which would reveal the ablation name.
                topics = [
                    {
                        field: topic.get(field)
                        for field in (
                            "topic_id",
                            "title",
                            "keywords",
                            "message_ids",
                            "keyword_supports",
                            "semantic_message_ids",
                            "context_linked_message_ids",
                        )
                    }
                    for topic in result["topics"]
                ]
                outputs.append(
                    {
                        "anonymous_configuration": key[variant],
                        "topics": topics,
                        "unassigned_message_ids": result["coverage_details"][
                            "unassigned_message_ids"
                        ],
                    }
                )
            payload = {
                "case_id": str(case["case_id"]),
                "messages": messages,
                "anonymous_outputs": outputs,
            }
            filename = f"case_{index_number:03d}.json"
            _write(directory / filename, payload)
            index.append(
                {
                    "case_id": str(case["case_id"]),
                    "file": filename,
                    "sha256": hashlib.sha256((directory / filename).read_bytes()).hexdigest(),
                }
            )
        _write(
            directory / "index.json",
            {
                "reviewer": reviewer,
                "cases": index,
                "instructions": REVIEW_INSTRUCTIONS,
                "evidence_type": "model_assisted_not_human_gold",
            },
        )
        paths.append(str(directory / "index.json"))
    _write(destination / "audit_only_identity_mapping.json", mapping)
    return {
        "review_packet_indices": paths,
        "case_count": len(cases),
        "configuration_count": len(variants),
        "identity_mapping": str(destination / "audit_only_identity_mapping.json"),
        "evidence_type": "two_isolated_model_assisted_reviews_not_human_gold",
    }


def merge_assisted_reviews(
    review_1: list[dict],
    review_2: list[dict],
    identity_mapping: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Keep raw evidence and flag rating differences without inventing a gold label."""
    normalized = []
    for reviewer, rows in enumerate((review_1, review_2), 1):
        inverse = {
            alias: variant for variant, alias in identity_mapping[f"reviewer_{reviewer}"].items()
        }
        seen = set()
        items = {}
        for row in rows:
            variant = inverse.get(row.get("anonymous_configuration"))
            if variant is None:
                raise ValueError("Review contains an unknown anonymous configuration")
            key = (str(row["case_id"]), variant)
            if key in seen:
                raise ValueError("Duplicate case/configuration in assisted review")
            seen.add(key)
            for dimension in REVIEW_DIMENSIONS:
                value = row.get(dimension)
                if value is not None and (type(value) is not int or value not in (0, 1, 2)):
                    raise ValueError(f"Invalid assisted rating for {dimension}")
            items[key] = {**row, "internal_variant": variant, "reviewer": reviewer}
        normalized.append(items)
    if set(normalized[0]) != set(normalized[1]):
        raise ValueError("Assisted reviewers did not cover identical case/configuration pairs")
    disagreements = []
    for key in sorted(normalized[0]):
        dimensions = [
            field
            for field in REVIEW_DIMENSIONS
            if normalized[0][key].get(field) != normalized[1][key].get(field)
        ]
        if dimensions:
            disagreements.append(
                {
                    "case_id": key[0],
                    "internal_variant": key[1],
                    "dimensions": dimensions,
                    "review_1": normalized[0][key],
                    "review_2": normalized[1][key],
                }
            )
    return {
        "evidence_type": "model_assisted_not_human_gold",
        "case_configuration_pairs": len(normalized[0]),
        "review_records": [list(items.values()) for items in normalized],
        "disagreements": disagreements,
        "semantic_accuracy": "not_estimated_without_human_gold",
    }
