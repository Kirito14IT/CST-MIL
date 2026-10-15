"""Frozen, verified local MiniLM with lossless overlapping token chunks."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ENCODING_VERSION = "content112-overlap16-mean-l2-v1"
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
        if tokenizer.num_special_tokens_to_add(pair=False) + 112 > encoder.max_seq_length:
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
                    tokenizer.prepare_for_model(
                        chunk,
                        add_special_tokens=True,
                        truncation=False,
                        return_attention_mask=True,
                        return_token_type_ids=False,
                    )
                    for chunk in chunks[start : start + batch_size]
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
