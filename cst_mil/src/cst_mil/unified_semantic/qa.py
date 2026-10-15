"""Development-only topic checks and grounded review materials."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def make_cases(output: Path) -> Path:
    path = output / "qa/cases.json"
    if path.exists():
        return path
    base = ROOT / "cst_mil/benchmark/data/full677"
    sessions = _rows(base / "sessions.jsonl")
    messages = _rows(base / "messages.jsonl")
    grouped: dict[str, list[dict]] = {}
    for m in messages:
        grouped.setdefault(m["post_id"], []).append(m)
    cases = []
    dev = [s for s in sessions if s["split"] == "dev"]
    if not dev:
        dev = [s for s in sessions if s["split"] == "validation"]
    # Six label x length strata. Boundaries depend on train conversations only.
    train_lengths = [len(grouped.get(s["post_id"], [])) for s in sessions if s["split"] == "train"]
    boundaries = np.quantile(train_lengths, [1 / 3, 2 / 3]).tolist()
    for label in (0, 1):
        for band in range(3):
            pool = [
                s
                for s in dev
                if int(s["label"]) == label
                and int(
                    np.searchsorted(boundaries, len(grouped.get(s["post_id"], [])), side="right")
                )
                == band
            ]
            pool.sort(key=lambda s: hashlib.sha256(("42:" + s["post_id"]).encode()).hexdigest())
            if len(pool) < 5:
                raise ValueError("Insufficient development strata for fixed 30-case review")
            for s in pool[:5]:
                cases.append(
                    {
                        "id": "sccd:" + s["post_id"],
                        "kind": "real_dev",
                        "reference_label": label,
                        "session": {
                            "post_id": s["post_id"],
                            "split": "dev",
                            "post_raw_text": s.get("post_raw_text", ""),
                        },
                        "messages": sorted(
                            grouped.get(s["post_id"], []),
                            key=lambda m: (m["order"], m["comment_id"]),
                        ),
                    }
                )
    example = ROOT / "计算机项目综合开发创新实践/app/examples/校园推荐演示-20261004.json"
    examples = _json(example)
    if isinstance(examples, dict):
        examples = examples.get("examples", examples.get("sessions", []))
    attack = next((e for e in examples if "比赛帖下的围攻" in str(e)), None)
    if attack:
        current = attack.get("session", attack)
        case = {
            "id": "campus:competition",
            "kind": "authored_challenge",
            "note": "既有演示输入，非独立准确率金标",
            "session": {"post_id": "campus:competition", "split": "development_challenge"},
            "messages": [
                {
                    "post_id": "campus:competition",
                    "comment_id": m.get("id", f"m{i + 1}"),
                    "raw_text": m.get("text", ""),
                    "order": m.get("order", i),
                    "parent_id": m.get("parentId"),
                }
                for i, m in enumerate(current["messages"])
            ],
        }
    else:
        raise ValueError("Missing original campus competition demonstration")
    cases.append(case)
    challenges = [
        (
            "sport_review",
            [
                "最后一节失误可以复盘。",
                "下周练一下传球配合。",
                "认真训练，下场继续加油。",
                "输赢都尊重队员。",
            ],
        ),
        (
            "counter_attack",
            [
                "不要辱骂同学，讨论技术就好。",
                "停止人身攻击，让管理员处理。",
                "这个废物赶紧滚出去。",
                "垃圾别再参加比赛了。",
            ],
        ),
        (
            "regional_hate",
            [
                "那个地方的人都是骗子。",
                "这种地域歧视太过分了。",
                "不要因为出生地攻击同学。",
                "招新请按能力评价。",
            ],
        ),
        (
            "quoted_hate",
            [
                "有人说“那个地方的人都是骗子”，这种说法不对。",
                "不要传播地域歧视。",
                "宿舍分配不应该看籍贯。",
            ],
        ),
        (
            "financial_promotion",
            [
                "Contact my investment mentor for guaranteed daily profits.",
                "DM now and double your money today.",
                "Do not send money to strangers online.",
                "Which official website verifies this offer?",
            ],
        ),
        (
            "normal_activity",
            [
                "读书会周五下午举行。",
                "我想分享最近看的小说。",
                "请提前报名，教室位置稍后公布。",
                "下次活动考虑户外阅读吗？",
            ],
        ),
        (
            "implicit_exclusion",
            [
                "他来了我们就不参加。",
                "大家以后都别理他。",
                "这样排斥同学不合适。",
                "有意见可以当面沟通。",
            ],
        ),
        (
            "multirisk_english",
            [
                "You are trash, leave the team.",
                "Guaranteed profits! Contact my broker now.",
                "Stop insulting our classmates.",
                "Please report suspicious advertisements.",
            ],
        ),
        (
            "portuguese",
            [
                "Você é um lixo, saia do time.",
                "Pare de ofender nossos colegas.",
                "Vamos conversar sobre o jogo.",
                "Respeite as outras pessoas.",
            ],
        ),
        (
            "unsupported_rumor",
            [
                "听说食堂明天要停业是真的吗？",
                "官方通知还没发布，先别转发。",
                "建议问一下学校服务中心。",
            ],
        ),
        ("single_short", ["滚。"]),
    ]
    for name, texts in challenges:
        sid = "campus:" + name
        cases.append(
            {
                "id": sid,
                "kind": "authored_challenge",
                "note": "编写的开发功能挑战，非真实风险金标",
                "session": {"post_id": sid, "split": "development_challenge"},
                "messages": [
                    {
                        "post_id": sid,
                        "comment_id": f"m{i + 1}",
                        "raw_text": text,
                        "order": i,
                        "parent_id": None,
                    }
                    for i, text in enumerate(texts)
                ],
            }
        )
    if len(cases) != 42 or len({c["id"] for c in cases}) != 42:
        raise ValueError("QA case list must contain exactly 30 real dev and 12 campus challenges")
    _write(path, {"seed": 42, "length_boundaries_train_only": boundaries, "cases": cases})
    return path


def run_topic_qa(output: Path) -> dict:
    from .model import canonical_messages
    from .topics import generic_word, interpret_topics

    case_path = make_cases(output)
    cases = _json(case_path)["cases"]
    model = joblib.load(output / "model.joblib")
    encoder = model.encoder()
    from cst_mil.bench.model import predict_session as old_predict

    old = joblib.load(ROOT / "cst_mil/benchmark/runs/full677/cst_mil_r0/model.joblib")
    baseline = []
    results = {str(d): [] for d in (0.25, 0.35, 0.45)}
    for case in cases:
        messages = canonical_messages(case["session"], case["messages"])
        cache_key = hashlib.sha256(case["id"].encode("utf-8")).hexdigest()[:20]
        emb = encoder.encode(
            [m["raw_text"] for m in messages], cache_path=output / f"qa/embeddings/{cache_key}.npz"
        )
        scores = model.score_embeddings([m["raw_text"] for m in messages], emb)
        baseline.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "result": old_predict(old, case["session"], case["messages"]),
            }
        )
        for distance in (0.25, 0.35, 0.45):
            result = interpret_topics(
                messages,
                emb,
                scores,
                distance_threshold=distance,
                category_thresholds=model.thresholds,
            )
            results[str(distance)].append(
                {"id": case["id"], "kind": case["kind"], "result": result}
            )
    _write(output / "qa/baseline.json", baseline)
    summary = {}
    for distance, items in results.items():
        unsupported = generic = duplicate_groups = 0
        count = 0
        real_count = real_assigned = real_total = 0
        for item in items:
            seen = set()
            for topic in item["result"]["topics"]:
                count += 1
                group = frozenset(topic["message_ids"])
                if group in seen:
                    duplicate_groups += 1
                seen.add(group)
                for word in topic["keywords"]:
                    if generic_word(word) or len(word.replace(" ", "")) < 2:
                        generic += 1
                    support = topic.get("keyword_supports", {}).get(word, [])
                    if not support:
                        unsupported += 1
                    for s in support:
                        raw = next(
                            (
                                m["raw_text"]
                                for m in topic["support_messages"]
                                if m["comment_id"] == s["message_id"]
                            ),
                            None,
                        )
                        if (
                            raw is None
                            or unicodedata.normalize("NFC", raw[s["start"] : s["end"]]).casefold()
                            != word
                        ):
                            unsupported += 1
            if item["kind"] == "real_dev":
                real_count += len(item["result"]["topics"])
                real_assigned += item["result"]["coverage_details"]["assigned_messages"]
                real_total += item["result"]["coverage_details"]["total_messages"]
        _write(output / f"qa/distance_{distance}.json", items)
        summary[distance] = {
            "topics": count,
            "real_dev_topics": real_count,
            "real_dev_coverage": real_assigned / max(real_total, 1),
            "unsupported_keywords": unsupported,
            "generic_only_keywords": generic,
            "duplicate_membership_groups": duplicate_groups,
            "hard_checks_passed": unsupported == generic == duplicate_groups == 0
            and real_count >= 20,
        }
    _write(output / "qa/automatic_checks.json", summary)
    for case in cases:
        packet = {
            "id": case["id"],
            "kind": case["kind"],
            "messages": [
                {"id": m["comment_id"], "text": m["raw_text"], "parent_id": m.get("parent_id")}
                for m in case["messages"]
            ],
            "candidates": {},
        }
        for distance, items in results.items():
            result = next(item["result"] for item in items if item["id"] == case["id"])
            packet["candidates"][distance] = {
                "topics": [
                    {
                        "title": t["title"],
                        "keywords": t["keywords"],
                        "message_ids": t["message_ids"],
                        "keyword_supports": t["keyword_supports"],
                        "contextHint": t["contextHint"],
                    }
                    for t in result["topics"]
                ],
                "unassigned_ids": result["coverage_details"]["unassigned_message_ids"],
            }
        name = hashlib.sha256(case["id"].encode("utf-8")).hexdigest()[:20]
        _write(output / f"qa/review_packets/{name}.json", packet)
    _write(
        output / "qa/review_packets/index.json",
        [
            {
                "id": c["id"],
                "kind": c["kind"],
                "file": hashlib.sha256(c["id"].encode("utf-8")).hexdigest()[:20] + ".json",
            }
            for c in cases
        ],
    )
    return summary
