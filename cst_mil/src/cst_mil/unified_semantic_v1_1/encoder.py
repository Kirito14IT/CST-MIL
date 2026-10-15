"""Frozen, verified local MiniLM with lossless overlapping token chunks."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ENCODING_VERSION = "content112-overlap16-verified-special-template-mean-l2-v1.1"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
DEFAULT_ENCODER_PATH = (
    Path(__file__).resolve().parents[4]
    / "benchmark"
    / "models"
    / ("paraphrase-multilingual-MiniLM-L12-v2")
)
_ENCODERS: dict[tuple[str, str], Any] = {}
_VERIFIED: dict[tuple[str, str], tuple[Any, ...]] = {}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


def token_chunks(
    token_ids: list[int], content_limit: int = 112, overlap: int = 16
) -> list[list[int]]:
    """Every original token occurs in at least one chunk, including the suffix."""
    if content_limit < 1 or overlap < 0 or overlap >= content_limit:
        raise ValueError("Invalid token chunk length/overlap")
    result = []
    start = 0
    while start < len(token_ids):
        result.append(token_ids[start : start + content_limit])
        if start + content_limit >= len(token_ids):
            break
        start += content_limit - overlap
    return result or [[]]


@dataclass(frozen=True)
class SpecialTokenTemplate:
    """Verified native single-sequence template, applied directly to content IDs.

    Generic PreTrainedTokenizerFast.prepare_for_model inherits a Python method
    which does not execute the Rust postprocessor. The local snapshot's actual
    backend specifies <s> A </s>; validate that definition rather than silently
    relying on the generic method or decoding/re-tokenizing an existing chunk.
    """

    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    postprocessor_sha256: str

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> SpecialTokenTemplate:
        try:
            document = json.loads(tokenizer.backend_tokenizer.to_str())
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Cannot inspect actual fast-tokenizer postprocessor") from error
        processor = document.get("post_processor")
        if not isinstance(processor, dict) or processor.get("type") != "TemplateProcessing":
            raise ValueError("Unsupported or missing tokenizer TemplateProcessing postprocessor")
        single = processor.get("single")
        if not isinstance(single, list) or len(single) != 3:
            raise ValueError("Single-sequence template must contain CLS, content A, and SEP")
        if single[1] != {"Sequence": {"id": "A", "type_id": 0}}:
            raise ValueError("Sequence template must preserve content A with type_id=0")
        definitions = processor.get("special_tokens")
        if not isinstance(definitions, dict):
            raise ValueError("Missing special token definitions in actual postprocessor")
        expected = (
            (single[0], tokenizer.cls_token, tokenizer.cls_token_id),
            (single[2], tokenizer.sep_token, tokenizer.sep_token_id),
        )
        verified_ids = []
        for piece, token, token_id in expected:
            if not isinstance(token, str) or type(token_id) is not int or token_id < 0:
                raise ValueError("Special token metadata must provide valid CLS/SEP IDs")
            if piece != {"SpecialToken": {"id": token, "type_id": 0}}:
                raise ValueError("Special token template does not match tokenizer CLS/SEP")
            definition = definitions.get(token)
            if (
                not isinstance(definition, dict)
                or definition.get("id") != token
                or definition.get("tokens") != [token]
                or definition.get("ids") != [token_id]
                or tokenizer.convert_tokens_to_ids(token) != token_id
            ):
                raise ValueError("Special token IDs disagree with the actual tokenizer vocabulary")
            verified_ids.append(token_id)
        if tokenizer.num_special_tokens_to_add(pair=False) != len(verified_ids):
            raise ValueError("Native postprocessor special-token count disagrees with template")
        return cls((verified_ids[0],), (verified_ids[1],), digest_json(processor))

    @property
    def special_token_count(self) -> int:
        return len(self.prefix_ids) + len(self.suffix_ids)

    def prepare(self, content_ids: list[int]) -> dict[str, list[int]]:
        if any(type(value) is not int or value < 0 for value in content_ids):
            raise ValueError("Content token IDs must be nonnegative integers")
        input_ids = [*self.prefix_ids, *content_ids, *self.suffix_ids]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


class LocalSemanticEncoder:
    """No downloads, encoder fine tuning, hidden truncation, or paid APIs."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ENCODER_PATH,
        fingerprint: str | None = None,
        *,
        threads: int = 4,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.fingerprint = fingerprint
        if not 1 <= threads <= 4:
            raise ValueError("CPU thread count must be between 1 and 4")
        self.threads = int(threads)
        self.info: dict[str, Any] = {
            "encoding_version": ENCODING_VERSION,
            "device": "cpu",
            "dimensions": 384,
            "content_tokens_per_chunk": 112,
            "overlap_tokens": 16,
            "special_tokens_per_chunk": 2,
            "postprocessor_sha256": None,
            "truncated_content_tokens": 0,
            "encoded_texts": 0,
            "encoded_chunks": 0,
            "original_content_tokens": 0,
            "cached_texts": 0,
            "encoder_load_seconds": 0.0,
            "encoding_seconds": 0.0,
        }

    def validate_assets(self) -> dict[str, Any]:
        root = Path(self.model_path)
        manifest_path = root / "benchmark_model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
            raise ValueError("Encoder identity/revision mismatch")
        files = manifest.get("files", {})
        expected = manifest.get("model_fingerprint")
        # Existing snapshot manifest uses ASCII canonical JSON for its digest.
        actual_digest = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual_digest != expected or "model.safetensors" not in files:
            raise ValueError("Encoder manifest is not self-consistent")
        if self.fingerprint is not None and self.fingerprint != expected:
            raise ValueError("Encoder differs from the model's frozen fingerprint")
        actual_files = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and ".cache" not in p.parts and p.name != "benchmark_model_manifest.json"
        }
        if actual_files != set(files):
            raise ValueError("Encoder snapshot inventory mismatch")
        signature = []
        for relative, expected_file in sorted(files.items()):
            target = (root / relative).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError(f"Missing or unsafe encoder asset: {relative}")
            stat = target.stat()
            if stat.st_size != expected_file["bytes"]:
                raise ValueError(f"Encoder size mismatch: {relative}")
            signature.append((relative, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        key = (self.model_path, str(expected))
        signature_tuple = tuple(signature)
        if _VERIFIED.get(key) != signature_tuple:
            for relative, expected_file in sorted(files.items()):
                if file_sha256(root / relative) != expected_file["sha256"]:
                    raise ValueError(f"Encoder SHA256 mismatch: {relative}")
            _VERIFIED[key] = signature_tuple
        self.fingerprint = str(expected)
        return manifest

    def _encoder(self) -> Any:
        import torch
        from sentence_transformers import SentenceTransformer

        self.validate_assets()
        torch.set_num_threads(self.threads)
        key = (self.model_path, str(self.fingerprint))
        if key not in _ENCODERS:
            started = time.perf_counter()
            encoder = SentenceTransformer(
                self.model_path,
                device="cpu",
                local_files_only=True,
                trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
            )
            encoder.eval()
            if encoder.get_sentence_embedding_dimension() != 384:
                raise ValueError("Encoder output dimension is not the frozen 384")
            _ENCODERS[key] = encoder
            self.info["encoder_load_seconds"] += time.perf_counter() - started
        return _ENCODERS[key]

    def encode(
        self, texts: list[str], *, cache_path: str | Path | None = None, batch_size: int = 32
    ) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.validate_assets()
        identity = digest_json(
            {
                "texts": [hashlib.sha256(str(t).encode("utf-8")).hexdigest() for t in texts],
                "encoder": self.fingerprint,
                "encoding": ENCODING_VERSION,
                "encoder_code": file_sha256(Path(__file__)),
            }
        )
        cache = Path(cache_path) if cache_path is not None else None
        if cache is not None and cache.is_file():
            with np.load(cache, allow_pickle=False) as stored:
                if str(stored["identity"].item()) != identity:
                    raise ValueError("Embedding cache fingerprint mismatch; use a new run")
                values = np.asarray(stored["embeddings"], dtype=np.float32)
            if values.shape != (len(texts), 384) or not np.isfinite(values).all():
                raise ValueError("Invalid cached embedding shape/values")
            self.info["cached_texts"] += len(texts)
            return values
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        import torch

        started = time.perf_counter()
        encoder = self._encoder()
        tokenizer = encoder.tokenizer
        template = SpecialTokenTemplate.from_tokenizer(tokenizer)
        self.info["special_tokens_per_chunk"] = template.special_token_count
        self.info["postprocessor_sha256"] = template.postprocessor_sha256
        if template.special_token_count + 112 > encoder.max_seq_length:
            raise ValueError("Encoder sequence limit cannot accommodate audited 112-token chunks")
        owners, chunks = [], []
        for owner, text in enumerate(texts):
            ids = tokenizer.encode(
                str(text), add_special_tokens=False, truncation=False, verbose=False
            )
            self.info["original_content_tokens"] += len(ids)
            for chunk in token_chunks(ids):
                owners.append(owner)
                chunks.append(chunk)
        totals = np.zeros((len(texts), 384), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(chunks), batch_size):
                prepared = [
                    template.prepare(chunk) for chunk in chunks[start : start + batch_size]
                ]
                features = tokenizer.pad(prepared, padding=True, return_tensors="pt")
                vectors = encoder(features)["sentence_embedding"].cpu().numpy()
                for offset, vector in enumerate(vectors):
                    owner = owners[start + offset]
                    totals[owner] += vector
                    counts[owner] += 1
        values = unit_rows(totals / counts[:, None])
        self.info["encoded_texts"] += len(texts)
        self.info["encoded_chunks"] += len(chunks)
        self.info["encoding_seconds"] += time.perf_counter() - started
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_name(cache.name + f".{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, embeddings=values, identity=np.asarray(identity))
            temporary.replace(cache)
        return values
