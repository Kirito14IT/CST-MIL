"""Frozen topic adapters for the full joint-supervision comparison.

Native backends are imported without editing their audited implementation.  The
BERTopic adapter accepts embeddings from the v1.1 encoder and never instantiates
the historical BERTopic tokenizer/encoder.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from cst_mil.research_followup.shorttext_nmf_stable.common import normalise_rows, topic_tokens
from cst_mil.research_followup.shorttext_nmf_stable import native_topics


def verify_native_tools(method: str) -> dict[str, Any]:
    """Fail before fitting rather than rebuilding any historical tool directory."""
    source = native_topics.REPOSITORY_ROOT / "baseline" / method.upper()
    suffixes = {".java"} if method == "gsdmm" else {".c", ".cpp", ".h"}
    sources = [p for p in source.rglob("*") if p.suffix in suffixes]
    build = native_topics.TOOLS_ROOT / "native" / method
    marker = build / "build.json"
    if not sources or not marker.is_file():
        raise RuntimeError(f"Frozen {method} native tool is missing; automatic rebuild is disabled")
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    expected = build / "classes/main/Benchmark.class" if method == "gsdmm" else build / (
        method + (".exe" if native_topics.os.name == "nt" else "")
    )
    artifacts = list((build / "classes").rglob("*.class")) if method == "gsdmm" else [expected]
    if (not expected.is_file() or manifest.get("source_sha256") != native_topics._digest_files(sources)
            or manifest.get("binary_sha256") != native_topics._digest_files(artifacts)):
        raise RuntimeError(f"Frozen {method} native tool fingerprint differs; rebuild is disabled")
    return manifest


@dataclass
class AuditedBERTopic:
    seed: int
    threads: int = 4
    model: Any = None
    topic_ids: list[int] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def fit(self, texts: list[str], embeddings: np.ndarray) -> "AuditedBERTopic":
        import numba
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP

        if embeddings.shape != (len(texts), 384) or not np.isfinite(embeddings).all():
            raise ValueError("BERTopic requires one audited 384D embedding per document")
        if not texts:
            raise ValueError("BERTopic requires training documents")
        numba.set_num_threads(min(self.threads, numba.config.NUMBA_NUM_THREADS))
        self.info = {"method": "bertopic", "seed": self.seed, "threads": self.threads,
                     "bertopic_version": version("bertopic"), "training_documents": len(texts),
                     "embedding_protocol": "v1.1-content112-overlap16-shared-external",
                     "fitted": True}
        if len(texts) < 3 or not any(topic_tokens(text) for text in texts):
            self.info.update(n_topics=0, training_noise_fraction=1.0,
                             degenerate_reason="insufficient_documents_or_empty_vocabulary")
            return self
        self.info["umap"] = {
            "n_neighbors": min(15, len(texts) - 1),
            "n_components": min(5, max(1, len(texts) - 2)),
            "random_state": self.seed, "transform_seed": self.seed,
            "n_jobs": 1, "metric": "cosine", "init": "random" if len(texts) < 8 else "spectral",
        }
        self.info["hdbscan"] = {"min_cluster_size": 15, "min_samples": min(5, len(texts) - 1),
                                "prediction_data": True, "core_dist_n_jobs": self.threads}
        self.model = BERTopic(
            language="multilingual", embedding_model=None, calculate_probabilities=True,
            verbose=False, umap_model=UMAP(**self.info["umap"]),
            hdbscan_model=HDBSCAN(**self.info["hdbscan"]),
            vectorizer_model=CountVectorizer(tokenizer=topic_tokens, token_pattern=None,
                                             lowercase=False, min_df=1), top_n_words=10,
        )
        started = time.perf_counter()
        labels, _ = self.model.fit_transform(texts, embeddings=embeddings)
        self.topic_ids = sorted(int(k) for k in self.model.get_topics() if int(k) >= 0)
        self.info.update(n_topics=len(self.topic_ids), topic_ids=self.topic_ids,
                         fit_seconds=time.perf_counter() - started,
                         training_noise_fraction=float(np.mean(np.asarray(labels) < 0)),
                         all_noise=not self.topic_ids)
        return self

    def transform(self, texts: list[str], embeddings: np.ndarray) -> np.ndarray:
        if not self.info.get("fitted"):
            raise RuntimeError("BERTopic has not been fitted")
        if embeddings.shape != (len(texts), 384) or not np.isfinite(embeddings).all():
            raise ValueError("BERTopic embedding shape/content mismatch")
        if not texts or not self.topic_ids:
            return np.zeros((len(texts), len(self.topic_ids)))
        labels, probabilities = self.model.transform(texts, embeddings=embeddings)
        if probabilities is None:
            raise RuntimeError("BERTopic did not return topic membership probabilities")
        values = np.asarray(probabilities, dtype=float)
        if values.ndim == 1 and len(self.topic_ids) == 1:
            values = values.reshape(-1, 1)
        if values.shape != (len(texts), len(self.topic_ids)):
            raise RuntimeError(f"Unexpected BERTopic membership shape: {values.shape}")
        values = values.copy()
        values[np.asarray(labels) < 0] = 0
        return normalise_rows(values)

    def topic_words(self, top_n: int = 10) -> list[list[str]]:
        return [[str(word) for word, _ in (self.model.get_topic(topic) or [])[:top_n] if word]
                for topic in self.topic_ids]


def fit_topic_backend(method: str, texts: list[str], seed: int, threads: int,
                      work_dir: str | Path, embeddings: np.ndarray | None = None) -> Any:
    config = {"seed": int(seed), "threads": int(threads), "n_topics": 32}
    if method == "bertopic":
        if embeddings is None:
            raise ValueError("Audited BERTopic embeddings are required")
        return AuditedBERTopic(seed, threads).fit(texts, embeddings)
    if method == "nmf":
        backend = native_topics.NMFTopicModel(config)
    elif method in {"lda", "btm", "gsdmm"}:
        verify_native_tools(method)
        backend = native_topics.NativeTopicModel(method, config, Path(work_dir))
    else:
        raise ValueError(f"Unsupported topic backend {method}")
    backend.fit(texts)
    return backend


def topic_transform(backend: Any, texts: list[str], embeddings: np.ndarray | None = None) -> np.ndarray:
    if isinstance(backend, AuditedBERTopic):
        if embeddings is None:
            raise ValueError("Audited BERTopic embeddings are required")
        return backend.transform(texts, embeddings)
    values = backend.transform(texts)
    # An empty training vocabulary is an actual absence of usable topics.
    if not backend.vocabulary_:
        return np.zeros((len(texts), 0))
    return values
