"""Tokenizer-only regression guards; never load the MiniLM transformer weights."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cst_mil.unified_semantic_v1_1.encoder import (
    DEFAULT_ENCODER_PATH,
    ENCODING_VERSION,
    LocalSemanticEncoder,
    SpecialTokenTemplate,
    token_chunks,
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    if not (DEFAULT_ENCODER_PATH / "tokenizer.json").is_file():
        pytest.skip("The verified local benchmark tokenizer is not installed")
    result = transformers.AutoTokenizer.from_pretrained(
        str(DEFAULT_ENCODER_PATH), local_files_only=True, trust_remote_code=False
    )
    assert isinstance(result, transformers.PreTrainedTokenizerFast)
    return result


@pytest.mark.parametrize(
    "text", ["", "  ", "校园比赛，请尊重同学。", "Hello, teammates!", "<s>中文</s>"]
)
def test_native_fast_tokenizer_single_sequence_equivalence(tokenizer, text):
    content = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    actual = SpecialTokenTemplate.from_tokenizer(tokenizer).prepare(content)
    native = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    assert actual["input_ids"] == native["input_ids"]
    assert actual["attention_mask"] == native["attention_mask"]
    assert actual["input_ids"][0] == tokenizer.cls_token_id
    assert actual["input_ids"][-1] == tokenizer.sep_token_id
    assert actual["input_ids"][1:-1] == content
    assert len(actual["input_ids"]) == len(content) + 2


def test_long_chunks_match_native_postprocessing_and_preserve_every_token(tokenizer):
    text = ("校园讨论 language 葡萄牙语 colegas 日本語。" * 100) + "最后一个尾词！"
    original = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    assert len(original) > 224
    chunks = token_chunks(original)
    restored = chunks[0] + [token for chunk in chunks[1:] for token in chunk[16:]]
    assert restored == original
    assert all(len(chunk) <= 112 for chunk in chunks)
    assert chunks[-1][-1] == original[-1]

    # Native tokenizers.Encoding can be split without decoding or re-tokenizing.
    encoding = tokenizer.backend_tokenizer.encode(text, add_special_tokens=False)
    encoding.truncate(112, stride=16)
    native_chunks = [encoding, *encoding.overflowing]
    assert [part.ids for part in native_chunks] == chunks
    template = SpecialTokenTemplate.from_tokenizer(tokenizer)
    for content, part in zip(chunks, native_chunks, strict=True):
        actual = template.prepare(content)
        native = tokenizer.backend_tokenizer.post_process(part, add_special_tokens=True)
        assert actual["input_ids"] == native.ids
        assert actual["attention_mask"] == native.attention_mask
        assert actual["input_ids"][1:-1] == content
        assert len(actual["input_ids"]) <= 114


@pytest.mark.parametrize("problem", ["missing", "processor_type", "no_sep", "wrong_id", "type_id"])
def test_unverified_template_is_rejected_without_fallback(tokenizer, problem):
    document = json.loads(tokenizer.backend_tokenizer.to_str())
    processor = document["post_processor"]
    if problem == "missing":
        document["post_processor"] = None
    elif problem == "processor_type":
        processor["type"] = "UnknownProcessor"
    elif problem == "no_sep":
        processor["single"] = processor["single"][:-1]
    elif problem == "wrong_id":
        processor["special_tokens"][tokenizer.cls_token]["ids"] = [tokenizer.unk_token_id]
    elif problem == "type_id":
        processor["single"][1]["Sequence"]["type_id"] = 1
    proxy = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(to_str=lambda: json.dumps(document)),
        cls_token=tokenizer.cls_token,
        cls_token_id=tokenizer.cls_token_id,
        sep_token=tokenizer.sep_token,
        sep_token_id=tokenizer.sep_token_id,
        convert_tokens_to_ids=tokenizer.convert_tokens_to_ids,
        num_special_tokens_to_add=tokenizer.num_special_tokens_to_add,
    )
    with pytest.raises(ValueError, match="[Tt]emplate|[Ss]pecial|[Ss]equence|[Pp]rocessor"):
        SpecialTokenTemplate.from_tokenizer(proxy)


def test_encoder_batches_empty_and_long_inputs_without_generic_prepare(tokenizer, monkeypatch):
    torch = pytest.importorskip("torch")
    captured = []

    class TokenizerOnlyEncoder:
        max_seq_length = 128

        def __init__(self):
            self.tokenizer = tokenizer

        def __call__(self, features):
            captured.append({key: value.clone() for key, value in features.items()})
            return {"sentence_embedding": torch.ones((len(features["input_ids"]), 384))}

    encoder = LocalSemanticEncoder()
    encoder.fingerprint = "tokenizer-regression-fixture"
    monkeypatch.setattr(encoder, "validate_assets", lambda: {})
    monkeypatch.setattr(encoder, "_encoder", lambda: TokenizerOnlyEncoder())

    def forbidden(*args, **kwargs):
        raise AssertionError("Generic prepare_for_model must not be used")

    monkeypatch.setattr(tokenizer, "prepare_for_model", forbidden)
    texts = ["", "正常评论", "长对话 mixed words。" * 100]
    vectors = encoder.encode(texts, batch_size=2)
    assert vectors.shape == (3, 384)
    assert np.isfinite(vectors).all()
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.ones(3))
    expected = []
    template = SpecialTokenTemplate.from_tokenizer(tokenizer)
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        expected.extend(template.prepare(chunk) for chunk in token_chunks(ids))
    rows = []
    for batch in captured:
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"], strict=True):
            length = int(mask.sum())
            rows.append(ids[:length].tolist())
            assert mask[:length].tolist() == [1] * length
            assert mask[length:].tolist() == [0] * (len(mask) - length)
            assert ids[length:].tolist() == [tokenizer.pad_token_id] * (len(ids) - length)
    assert rows == [row["input_ids"] for row in expected]
    assert rows[0] == [tokenizer.cls_token_id, tokenizer.sep_token_id]
    assert encoder.info["special_tokens_per_chunk"] == 2
    assert encoder.info["postprocessor_sha256"]
    assert encoder.info["truncated_content_tokens"] == 0
    assert ENCODING_VERSION != "content112-overlap16-mean-l2-v1"
