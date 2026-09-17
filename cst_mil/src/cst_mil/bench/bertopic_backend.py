"""Official BERTopic with pinned local CPU embeddings and lossless token chunks.

The adapter retains UMAP and HDBSCAN in its pickle. Embeddings are external,
content-addressed artifacts; the large sentence encoder is loaded lazily and is
never embedded into a fitted model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from cst_mil.bench.common import file_hash, normalise_rows, topic_tokens, write_json

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
ENCODING_VERSION = "original-token-chunks-weighted-mean-v1"
MODEL_DIRECTORY = Path(__file__).resolve().parents[4] / "benchmark" / "models" / (
    "paraphrase-multilingual-MiniLM-L12-v2"
)
_MANIFEST_NAME = "benchmark_model_manifest.json"
_ENCODERS: dict[str, Any] = {}
_VERIFIED_SNAPSHOTS: dict[tuple[int, str, str], tuple[tuple[Any, ...], ...]] = {}


def _manifest_digest(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stat_signature(paths: list[Path]) -> tuple[tuple[Any, ...], ...]:
    result = []
    for path in paths:
        stat = path.stat()
        result.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
    return tuple(result)


def prepare_encoder(model_dir: str | Path = MODEL_DIRECTORY) -> dict[str, Any]:
    """Fetch only the official pinned safetensors snapshot and hash every file."""
    from huggingface_hub import snapshot_download

    target = Path(model_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=target,
        allow_patterns=[
            "config.json", "config_sentence_transformers.json", "modules.json",
            "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
            "tokenizer_config.json", "sentencepiece.bpe.model", "unigram.json",
            "1_Pooling/config.json", "model.safetensors", "README.md",
        ],
        max_workers=2,
    )
    files = {
        p.relative_to(target).as_posix(): {"bytes": p.stat().st_size, "sha256": file_hash(p)}
        for p in sorted(target.rglob("*"))
        if p.is_file() and ".cache" not in p.parts and p.name != _MANIFEST_NAME
    }
    if "model.safetensors" not in files:
        raise RuntimeError("Pinned encoder snapshot has no model.safetensors")
    manifest = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
        "model_fingerprint": _manifest_digest(files),
        "files": files,
        "download_and_hash_seconds": time.perf_counter() - started,
        "weights_format": "safetensors",
        "device": "cpu",
    }
    write_json(target / _MANIFEST_NAME, manifest)
    return manifest


def token_chunks(token_ids: list[int], content_limit: int) -> list[list[int]]:
    """Partition original IDs without truncation, overlap, or a lost suffix."""
    if content_limit < 1:
        raise ValueError("The encoder must allow at least one content token")
    return [token_ids[i:i + content_limit] for i in range(0, len(token_ids), content_limit)] or [[]]


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 1e-12)


class BERTopicModel:
    """Common topic-backend interface backed by the unmodified official package."""

    def __init__(self, config: dict[str, Any], work_dir: str | Path):
        self.config = dict(config)
        self.options = dict(config.get("bertopic", {}))
        self.work_dir = Path(work_dir)
        self.model_dir = Path(self.options.get("model_dir", MODEL_DIRECTORY)).resolve()
        self.cache_dir = Path(self.options.get(
            "embedding_cache_dir", MODEL_DIRECTORY.parents[1] / "cache" / "bertopic_embeddings"
        )).resolve()
        self.seed = int(config.get("seed", 42))
        self.threads = int(config.get("threads", 4))
        self.model: Any = None
        self.topic_ids: list[int] = []
        self.manifest: dict[str, Any] | None = None
        self.info: dict[str, Any] = {
            "backend": "official-bertopic", "bertopic_version": "0.17.4",
            "encoder_id": MODEL_ID, "encoder_revision": MODEL_REVISION,
            "device": "cpu", "threads": self.threads,
            "encoding_version": ENCODING_VERSION,
            "transform_batch_contract": "one_complete_session_per_call",
            "encoding_cold_seconds": 0.0, "encoding_cached_seconds": 0.0,
            "encoder_load_seconds": 0.0, "encoded_texts": 0, "cached_texts": 0,
            "encoded_chunks": 0, "encoded_content_tokens": 0,
            "truncated_content_tokens": 0, "fitted": False,
        }

    def _read_manifest(self) -> dict[str, Any]:
        """Check disk identity on every call, but hash assets only after stat changes.

        The verification cache is process-local, never serialized with the model.
        Consequently a restored model's first use in another process always hashes
        the actual assets, even though its original manifest is in the pickle.
        """
        path = self.model_dir / _MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(
                f"Encoder is not prepared: {path}. Run uv run --group benchmark python "
                "-m cst_mil.bench.bertopic_backend prepare first."
            )
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("model_id") != MODEL_ID
                or manifest.get("revision") != MODEL_REVISION):
            raise ValueError("Encoder manifest does not match the pinned model revision")
        files = manifest.get("files", {})
        if "model.safetensors" not in files:
            raise ValueError("Encoder manifest is missing model.safetensors")
        if _manifest_digest(files) != manifest.get("model_fingerprint"):
            raise ValueError("Encoder manifest fingerprint is not self-consistent")
        if (self.manifest is not None
                and _manifest_digest(manifest) != _manifest_digest(self.manifest)):
            raise ValueError("Encoder disk manifest differs from the fitted model manifest")
        actual_files = {
            p.relative_to(self.model_dir).as_posix()
            for p in self.model_dir.rglob("*")
            if p.is_file() and ".cache" not in p.parts and p.name != _MANIFEST_NAME
        }
        if actual_files != set(files):
            raise ValueError("Encoder snapshot file inventory differs from its manifest")
        assets = []
        for relative, expected in sorted(files.items()):
            target = (self.model_dir / relative).resolve()
            if not target.is_relative_to(self.model_dir):
                raise ValueError(f"Encoder asset escapes its snapshot directory: {relative}")
            if not target.is_file() or target.stat().st_size != expected["bytes"]:
                raise ValueError(f"Encoder snapshot file missing or changed size: {target}")
            assets.append(target)
        signature = _stat_signature([path, *assets])
        verification_key = (os.getpid(), str(self.model_dir), _manifest_digest(manifest))
        if _VERIFIED_SNAPSHOTS.get(verification_key) != signature:
            started = time.perf_counter()
            for relative, expected in sorted(files.items()):
                if file_hash(self.model_dir / relative) != expected["sha256"]:
                    raise ValueError(f"Encoder snapshot hash mismatch: {relative}")
            if _stat_signature([path, *assets]) != signature:
                raise ValueError("Encoder snapshot changed during hash verification")
            _VERIFIED_SNAPSHOTS[verification_key] = signature
            self.info["encoder_asset_verification_seconds"] = self.info.get(
                "encoder_asset_verification_seconds", 0.0
            ) + time.perf_counter() - started
            self.info["encoder_full_hash_checks"] = self.info.get("encoder_full_hash_checks", 0) + 1
        self.manifest = manifest
        self.info["encoder_fingerprint"] = manifest["model_fingerprint"]
        self.info["encoder_manifest"] = str(path)
        return manifest

    def validate_assets(self) -> dict[str, Any]:
        """Preflight external assets immediately after checkpoint loading."""
        return self._read_manifest()

    def _encoder(self) -> Any:
        import torch
        from sentence_transformers import SentenceTransformer

        manifest = self._read_manifest()
        key = str(self.model_dir) + ":" + manifest["model_fingerprint"]
        torch.set_num_threads(self.threads)
        if key not in _ENCODERS:
            started = time.perf_counter()
            encoder = SentenceTransformer(
                str(self.model_dir), device="cpu", local_files_only=True,
                trust_remote_code=False, model_kwargs={"use_safetensors": True},
            )
            encoder.eval()
            _ENCODERS[key] = encoder
            self.info["encoder_load_seconds"] += time.perf_counter() - started
        return _ENCODERS[key]

    def _encode_uncached(self, texts: list[str]) -> np.ndarray:
        import torch

        encoder = self._encoder()
        tokenizer = encoder.tokenizer
        sequence_limit = min(128, int(encoder.max_seq_length))
        content_limit = sequence_limit - tokenizer.num_special_tokens_to_add(pair=False)
        if (tokenizer.num_special_tokens_to_add(pair=False) != 2
                or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None):
            raise ValueError("Pinned BERT encoder must have the audited CLS/content/SEP format")
        self.info["max_sequence_tokens"] = sequence_limit
        self.info["max_content_tokens"] = content_limit
        owners: list[int] = []
        weights: list[int] = []
        chunks: list[list[int]] = []
        for owner, text in enumerate(texts):
            ids = tokenizer.encode(
                str(text), add_special_tokens=False, truncation=False, verbose=False
            )
            self.info["encoded_content_tokens"] += len(ids)
            for chunk in token_chunks(ids, content_limit):
                owners.append(owner)
                weights.append(max(1, len(chunk)))
                chunks.append(chunk)
        dimensions = int(encoder.get_embedding_dimension())
        pooled = np.zeros((len(texts), dimensions), dtype=np.float64)
        total_weights = np.zeros(len(texts), dtype=np.float64)
        batch_size = int(self.options.get("encoding_batch_size", 32))
        if batch_size < 1:
            raise ValueError("encoding_batch_size must be positive")
        for start in range(0, len(chunks), batch_size):
            stop = min(start + batch_size, len(chunks))
            # Fast-tokenizer prepare_for_model can omit its Rust postprocessor
            # while creating longer token_type_ids. This pinned model uses the
            # audited BERT single-sequence format; wrap the original IDs directly.
            features = []
            for ids in chunks[start:stop]:
                wrapped = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
                features.append({"input_ids": wrapped, "attention_mask": [1] * len(wrapped),
                                 "token_type_ids": [0] * len(wrapped)})
            padded = tokenizer.pad(features, padding=True, return_tensors="pt")
            if padded["input_ids"].shape[1] > sequence_limit:
                raise RuntimeError("Encoder chunk exceeds the audited token limit")
            with torch.inference_mode():
                vectors = encoder(dict(padded))["sentence_embedding"].cpu().numpy()
            for local_index, vector in enumerate(vectors):
                index = start + local_index
                pooled[owners[index]] += vector * weights[index]
                total_weights[owners[index]] += weights[index]
        pooled /= np.maximum(total_weights[:, None], 1)
        self.info["encoded_chunks"] += len(chunks)
        return _unit_rows(pooled)

    def _embeddings(self, texts: list[str]) -> np.ndarray:
        manifest = self._read_manifest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_prefix = manifest["model_fingerprint"] + ":" + ENCODING_VERSION + ":"
        entries: list[np.ndarray | None] = [None] * len(texts)
        missing: dict[str, tuple[str, list[int]]] = {}
        cached_started = time.perf_counter()
        for i, text in enumerate(texts):
            key = hashlib.sha256((cache_prefix + str(text)).encode("utf-8")).hexdigest()
            path = self.cache_dir / key[:2] / (key + ".npy")
            if path.is_file():
                value = np.load(path, allow_pickle=False)
                if value.shape != (384,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid cached embedding: {path}")
                entries[i] = value
                self.info["cached_texts"] += 1
            else:
                missing.setdefault(key, (str(text), []))[1].append(i)
        self.info["encoding_cached_seconds"] += time.perf_counter() - cached_started
        if missing:
            started = time.perf_counter()
            items = list(missing.items())
            vectors = self._encode_uncached([value[0] for _, value in items])
            self.info["encoding_cold_seconds"] += time.perf_counter() - started
            self.info["encoded_texts"] += len(items)
            for (key, (_, indices)), vector in zip(items, vectors, strict=True):
                path = self.cache_dir / key[:2] / (key + ".npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, vector, allow_pickle=False)
                os.replace(temporary, path)
                for index in indices:
                    entries[index] = vector
        return np.vstack(entries).astype(np.float32) if entries else np.zeros((0, 384), np.float32)

    def fit(self, texts: list[str], sample_weight: Any = None) -> BERTopicModel:
        import numba
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        documents = [str(text) for text in texts]
        numba.set_num_threads(min(self.threads, numba.config.NUMBA_NUM_THREADS))
        if not documents:
            raise ValueError("BERTopic needs at least one training document")
        self.validate_assets()
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=float)
            if weights.shape != (len(documents),) or not np.allclose(weights, 1.0):
                raise ValueError("Official BERTopic does not support weighted topic fitting")
        self.info["bertopic_version"] = version("bertopic")
        self.info["training_documents"] = len(documents)
        self.topic_ids = []
        self.model = None
        if len(documents) < 3 or not any(topic_tokens(text) for text in documents):
            self.info.update({"fitted": True, "n_topics": 0, "training_noise_fraction": 1.0,
                              "degenerate_reason": "insufficient_documents_or_empty_vocabulary"})
            return self
        embeddings = self._embeddings(documents)
        neighbors = min(15, len(documents) - 1)
        dimensions = min(5, max(1, len(documents) - 2))
        min_cluster_size = 15
        min_samples = min(5, max(1, len(documents) - 1))
        self.info["umap"] = {"n_neighbors": neighbors, "n_components": dimensions,
                             "random_state": self.seed, "transform_seed": self.seed,
                             "n_jobs": 1, "metric": "cosine",
                             "init": "random" if len(documents) < 8 else "spectral"}
        self.info["hdbscan"] = {"min_cluster_size": min_cluster_size,
                                "min_samples": min_samples, "prediction_data": True,
                                "core_dist_n_jobs": self.threads}
        self.info["tiny_corpus_parameters_adapted"] = len(documents) < 16
        self.model = BERTopic(
            language="multilingual", embedding_model=None,
            calculate_probabilities=True, verbose=False,
            umap_model=UMAP(**self.info["umap"]),
            hdbscan_model=HDBSCAN(**self.info["hdbscan"]),
            vectorizer_model=CountVectorizer(
                tokenizer=topic_tokens, token_pattern=None, lowercase=False, min_df=1,
            ),
            top_n_words=10,
        )
        started = time.perf_counter()
        labels, _ = self.model.fit_transform(documents, embeddings=embeddings)
        self.topic_ids = sorted(int(k) for k in self.model.get_topics() if int(k) >= 0)
        self.info.update({
            "fitted": True, "n_topics": len(self.topic_ids), "topic_ids": self.topic_ids,
            "topic_fit_seconds": time.perf_counter() - started,
            "training_noise_fraction": float(np.mean(np.asarray(labels) < 0)),
            "all_noise": not self.topic_ids,
        })
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform exactly one full session's windows as a stable ordered batch."""
        if not self.info["fitted"]:
            raise RuntimeError("BERTopicModel must be fitted before transform")
        self.validate_assets()
        documents = [str(text) for text in texts]
        if not documents or not self.topic_ids:
            return np.zeros((len(documents), len(self.topic_ids)), dtype=float)
        embeddings = self._embeddings(documents)
        started = time.perf_counter()
        labels, probabilities = self.model.transform(documents, embeddings=embeddings)
        self.info["transform_seconds"] = self.info.get("transform_seconds", 0.0) + (
            time.perf_counter() - started
        )
        labels = np.asarray(labels, dtype=int)
        if probabilities is None:
            raise RuntimeError(
                "Official BERTopic did not provide requested membership probabilities"
            )
        values = np.asarray(probabilities, dtype=float)
        if values.ndim == 1 and len(self.topic_ids) == 1:
            values = values.reshape(-1, 1)
        if values.shape != (len(documents), len(self.topic_ids)):
            raise RuntimeError(f"Unexpected BERTopic membership shape: {values.shape}")
        values[labels < 0] = 0
        # Noise is deliberately unassigned even if HDBSCAN emits weak membership.
        return normalise_rows(values)

    def topic_words(self, top_n: int = 10) -> list[list[str]]:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        if not self.info["fitted"]:
            raise RuntimeError("BERTopicModel must be fitted before topic_words")
        return [
            [str(word) for word, _ in (self.model.get_topic(topic_id) or [])[:top_n] if word]
            for topic_id in self.topic_ids
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare"])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(prepare_encoder(args.model_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
