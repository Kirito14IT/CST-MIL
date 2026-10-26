"""Contract tests using supplied embeddings, without loading or training an encoder."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cst_mil/src"))
from cst_mil.bench.resources import configure_cpu
configure_cpu(4)
import numpy as np
from cst_mil import demo_semantic_adapter as adapter
from cst_mil.research_followup.semantic_patent_followup.model import VariantConfig

spec = importlib.util.spec_from_file_location("campus_worker", Path(__file__).with_name("worker.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class SuppliedPackage:
    thresholds = {"attack": 0.5, "hate": 0.5, "spam_scam": 0.5}
    conversation_threshold = 0.4
    model_version = "semantic-patent-followup-v1"

    def __init__(self, scores=None):
        self.calls = 0
        self.scores = scores

    def encode_records(self, records):
        self.calls += 1
        self.records = records
        self.embeddings = np.zeros((len(records), 384), dtype=np.float32)
        self.embeddings[:, 0] = 1
        if len(records) > 2:
            self.embeddings[2] = 0
            self.embeddings[2, 1] = 1
        return self.embeddings

    def predict_payload(self, records, sessions, embeddings=None):
        assert embeddings is self.embeddings
        self.sessions = sessions
        values = self.scores or [[0.2, 0.1, 0.1], [0.9, 0.2, 0.1], [0.05, 0.2, 0.05]][:len(records)]
        return {"message": [{"scores": score} for score in values],
                "session": [{"score": 0.3}]}


class LatestAdapterTests(unittest.TestCase):
    def setUp(self):
        self.session = {"post_id": "demo", "label": 1}
        self.messages = [
            {"comment_id": "p", "raw_text": "东北锅包肉真好吃", "parent_id": None, "order": 0, "label": 1},
            {"comment_id": "b", "raw_text": "东北锅包肉很好吃", "parent_id": None, "order": 1, "risk": 1},
            {"comment_id": "c", "raw_text": "是的", "parent_id": "p", "order": 2, "label": 0},
        ]

    def result(self):
        model = SuppliedPackage()
        result = adapter.predict_session(model, self.session, self.messages)
        return model, result

    def test_one_encoding_is_shared_and_labels_never_enter_features(self):
        model = SuppliedPackage()
        with patch.object(adapter.topics, "interpret_topics", wraps=adapter.topics.interpret_topics) as interpret:
            result = adapter.predict_session(model, self.session, self.messages)
            self.assertEqual(model.calls, 1)
            self.assertIs(interpret.call_args.args[1], model.embeddings)
        self.assertTrue(all(record["labels"] == [None] * 3 and record["observed_mask"] == [False] * 3 for record in model.records))
        self.assertTrue(all("risk" not in record and "label" not in record for record in model.records))
        self.assertEqual(model.records[2]["parent_id"], "p")
        self.assertEqual(model.sessions, [{"session_id": "demo", "split": "inference"}])
        self.assertIsNone(result["label"])
        self.assertEqual(result["inferenceTrace"]["semanticEncodingPasses"], 1)

    def test_parent_association_does_not_inherit_risk_or_name_statistics(self):
        _, result = self.result()
        self.assertEqual(len(result["topics"]), 1)
        topic = result["topics"][0]
        self.assertEqual(topic["basic_message_ids"], ["p", "b"])
        self.assertEqual(topic["context_association_message_ids"], ["c"])
        self.assertTrue(all(s["message_id"] != "c" for entries in topic["keyword_supports"].values() for s in entries))
        child = result["message_scores"][2]
        self.assertEqual(child["riskCategoryScores"], {"attack": .05, "hate": .2, "spam_scam": .05})
        self.assertEqual(child["risk_score"], .2)
        self.assertEqual(child["context_parent_message_id"], "p")
        self.assertEqual(child["topic_strengths"], [0.0])
        worker.validate_topic_grounding(result, self.messages, adapter.topics.legacy.generic_word)
        worker.validate_topic_associations(result, self.messages)

    def test_unmodified_unicode_long_original_and_category_overlaps(self):
        raw = "😀 Café Straße 不能辱骂别人！" * 120
        model = SuppliedPackage([[.8, .9, .1]])
        result = adapter.predict_session(model, self.session, [{"comment_id": "u", "raw_text": raw}])
        self.assertEqual(model.records[0]["text"], raw)
        self.assertEqual(result["message_scores"][0]["raw_text"], raw)
        self.assertEqual(result["message_scores"][0]["riskCategoryScores"], {"attack": .8, "hate": .9, "spam_scam": .1})
        self.assertEqual([r["id"] for r in result["riskCategories"] if r["detected"]], ["attack", "hate"])
        self.assertEqual(result["topics"], [])
        self.assertEqual(result["coverage"], 0)

    def test_invalid_and_cyclic_parent_edges_never_form_context_associations(self):
        for parent in ("missing", "c"):
            messages = copy.deepcopy(self.messages)
            messages[2]["parent_id"] = parent
            result = adapter.predict_session(SuppliedPackage(), self.session, messages)
            self.assertEqual(result["topics"][0]["context_association_message_ids"], [])
        messages = copy.deepcopy(self.messages)
        messages[0]["parent_id"] = "c"
        result = adapter.predict_session(SuppliedPackage(), self.session, messages)
        self.assertEqual(result["topics"][0]["context_association_message_ids"], [])

    def test_association_validator_rejects_forged_members_and_cascades(self):
        _, good = self.result()
        for change in ("duplicate", "forged_parent", "name_source", "semantic_strength"):
            bad = copy.deepcopy(good)
            if change == "duplicate": bad["topics"][0]["basic_message_ids"].append("c")
            elif change == "forged_parent": bad["message_scores"][2]["context_parent_message_id"] = "b"
            elif change == "name_source": next(iter(bad["topics"][0]["keyword_supports"].values()))[0]["message_id"] = "c"
            else: bad["message_scores"][2]["topic_strengths"][0] = .8
            with self.assertRaises(ValueError): worker.validate_topic_associations(bad, self.messages)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            adapter.predict_session(SuppliedPackage(), self.session, self.messages + [self.messages[0]])

    def test_package_identity_allows_only_known_cost_metadata_extras(self):
        with tempfile.TemporaryDirectory(prefix="cst-adapter-") as directory:
            _check_legacy_package_identity(self, Path(directory))


def _check_legacy_package_identity(testcase: unittest.TestCase, tmp_path: Path):
    """Exercise the legacy contract without requiring an unpublished research bundle."""
    explanation = {
        "distance_threshold": .35,
        "link_short_reply_max_nonspace": 12,
        "protected_clause_max_nonspace": 96,
        "inherit_parent_risk": False,
        "association_contributes_to_names": False,
        "pure_responses": sorted(adapter.topics.PURE_RESPONSES),
    }
    thresholds = {"attack": .5, "hate": .5, "spam_scam": .5}
    encoder_metadata = {
        "encoder_path": str(tmp_path / "encoder"),
        "fingerprint": "a" * 64,
        "encoding_version": adapter.ENCODING_VERSION,
        "encoder_code_sha256": "b" * 64,
        "asset_bytes": 10,
        "asset_hashes": {str(tmp_path / "encoder/model.safetensors"): "c" * 64},
    }
    source = {
        "seed": 42,
        "architecture": "full",
        "model_sha256": "d" * 64,
        "thresholds": {**thresholds, "conversation": .4, "conversation_common16": .45},
        "categories": list(adapter.CATEGORY_IDS),
        "training_sources": ["synthetic-fixture"],
        "explanation_config": explanation,
        "encoder": encoder_metadata,
    }
    manifest = {
        "bundle": {"manifestPath": "manifest.json", "explanationPath": "explanation.json"},
        "modelVersion": "semantic-patent-followup-v1",
        "modelSha256": source["model_sha256"],
        "thresholds": thresholds,
        "conversationThreshold": .4,
        "trainingSources": source["training_sources"],
        "encoder": {"path": "encoder", "fingerprint": encoder_metadata["fingerprint"]},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(source), encoding="utf-8")
    (tmp_path / "explanation.json").write_text(json.dumps(explanation), encoding="utf-8")
    model = SuppliedPackage()
    model.seed = 42
    model.config = VariantConfig()
    model.method = "semantic_patent_followup"
    model.thresholds = thresholds
    model.conversation_threshold = .4
    model.common16_threshold = .45
    model.encoder_metadata = copy.deepcopy(encoder_metadata)
    model.encoder_metadata.update(
        encoding_seconds_by_split={"train": 1.2, "dev": .3},
        encoding_peak_tree_rss_bytes=1234,
    )

    class Encoder:
        calls = 0

        def validate_assets(self):
            self.calls += 1

    encoder = Encoder()
    model.encoder = lambda: encoder
    adapter.validate_package(model, manifest, tmp_path)
    testcase.assertEqual(encoder.calls, 1)
    for key, replacement in (
        ("fingerprint", "0" * 64), ("encoding_version", "unverified"),
        ("encoder_code_sha256", "0" * 64), ("asset_bytes", 1),
        ("asset_hashes", {}), ("unknown_config", 1),
    ):
        changed = copy.copy(model)
        changed.encoder_metadata = copy.deepcopy(model.encoder_metadata)
        changed.encoder_metadata[key] = replacement
        with testcase.assertRaises(ValueError):
            adapter.validate_package(changed, manifest, tmp_path)


if __name__ == "__main__":
    unittest.main()
