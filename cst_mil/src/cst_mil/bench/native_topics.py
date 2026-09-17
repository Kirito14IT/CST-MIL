"""Faithful native topic backends and the explicitly identified sklearn NMF adapter.

Only training texts build the vocabulary. Native model files and the exact compiled
binary/classes are embedded in the Python object, so saved models do not depend on
temporary inference outputs or on a later rebuild of the native source.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import normalise_rows, topic_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = PROJECT_ROOT.parent
TOOLS_ROOT = PROJECT_ROOT / "benchmark" / "tools"


def _run(command: list[str], cwd: Path, log_path: Path | None = None) -> str:
    """No shell interpolation; surface native failures with the captured log."""
    options = {"cwd": cwd, "check": False, "stderr": subprocess.STDOUT,
               "text": True, "encoding": "utf-8", "errors": "replace",
               "creationflags": subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0}
    if log_path is None:
        result = subprocess.run([str(arg) for arg in command],
                                stdout=subprocess.PIPE, **options)
        output = result.stdout
    else:
        # Native progress stays visible and partial logs survive an interrupted fit.
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run([str(arg) for arg in command], stdout=handle, **options)
        output = log_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(
            f"Native command exited {result.returncode}: {command[0]}\n"
            + output[-8000:]
        )
    return output


def _digest_files(paths: list[Path]) -> str:
    value = hashlib.sha256()
    for path in sorted(paths):
        value.update(path.name.encode("utf-8"))
        value.update(path.read_bytes())
    return value.hexdigest()


def portable_java() -> Path:
    """Use the prepared JDK, without changing PATH/JAVA_HOME or downloading at run time."""
    candidates = sorted(TOOLS_ROOT.glob("jdk-*/bin/java.exe"))
    if candidates:
        return candidates[-1]
    executable = shutil.which("java")
    if executable and os.name != "nt":
        return Path(executable)
    raise RuntimeError(f"Portable JDK not prepared under {TOOLS_ROOT}")


def prepare_native_tools(method: str) -> dict[str, Any]:
    """Compile only missing/stale tools; return a source and binary fingerprint manifest."""
    if method not in {"lda", "btm", "gsdmm"}:
        raise ValueError(method)
    source = REPOSITORY_ROOT / "baseline" / method.upper()
    suffixes = {".java"} if method == "gsdmm" else {".c", ".cpp", ".h"}
    sources = [p for p in source.rglob("*") if p.suffix in suffixes]
    source_hash = _digest_files(sources)
    build = TOOLS_ROOT / "native" / method
    build.mkdir(parents=True, exist_ok=True)
    executable = build / (method + (".exe" if os.name == "nt" else ""))
    marker = build / "build.json"
    expected = build / "classes" / "main" / "Benchmark.class" if method == "gsdmm" else executable
    if marker.exists() and expected.exists():
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") == source_hash:
            return manifest
    if method == "lda":
        compiler = shutil.which("gcc")
        if not compiler:
            raise RuntimeError("gcc is required to build the original LDA-C backend")
        command = [compiler, "-O3", "-std=gnu99", "-fcommon", "-static"]
        command += [str(p) for p in sorted(source.glob("*.c"))]
        command += ["-lm", "-o", str(executable)]
    elif method == "btm":
        compiler = shutil.which("g++")
        if not compiler:
            raise RuntimeError("g++ is required to build the original BTM backend")
        command = [compiler, "-O3", "-std=c++11", "-static"]
        command += [str(p) for p in sorted((source / "src").glob("*.cpp"))]
        command += ["-o", str(executable)]
    else:
        compiler = str(portable_java().with_name("javac.exe" if os.name == "nt" else "javac"))
        classes = build / "classes"
        classes.mkdir(exist_ok=True)
        command = [compiler, "-encoding", "UTF-8", "--release", "8", "-cp",
                   str(source / "lib" / "org.json-20120521.jar"), "-d", str(classes)]
        command += [str(p) for p in sorted((source / "src" / "main").glob("*.java"))]
    _run(command, source, build / "build.log")
    artifacts = list((build / "classes").rglob("*.class")) if method == "gsdmm" else [executable]
    manifest = {"method": method, "source_sha256": source_hash,
                "binary_sha256": _digest_files(artifacts), "compiler": str(compiler),
                "build_dir": str(build), "command": command,
                "java": str(portable_java()) if method == "gsdmm" else None}
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _vocabulary(texts: list[str], max_features: int = 20000) -> dict[str, int]:
    frequencies: Counter[str] = Counter()
    for text in texts:
        frequencies.update(topic_tokens(text))
    ordered = sorted(frequencies, key=lambda word: (-frequencies[word], word))[:max_features]
    return {word: index for index, word in enumerate(ordered)}


def _topic_words(components: np.ndarray, vocabulary: dict[str, int], top_n: int) -> list[list[str]]:
    words = [word for word, _ in sorted(vocabulary.items(), key=lambda item: item[1])]
    return [[words[i] for i in np.argsort(-row, kind="stable")[:top_n] if row[i] > 0]
            for row in components]


class NativeTopicModel:
    """LDA-C variational EM, BTM sum_b, or original Java collapsed-Gibbs GSDMM."""

    def __init__(self, method: str, config: dict, work_dir: Path):
        if method not in {"lda", "btm", "gsdmm"}:
            raise ValueError(f"Unsupported native topic method: {method}")
        self.method = method
        self.config = dict(config)
        self.work_dir = Path(work_dir).resolve()
        defaults = {
            "lda": {"em_max_iter": 100, "var_max_iter": 20, "alpha": 1.0},
            "btm": {"iterations": 100, "alpha": 50 / int(config.get("n_topics", 32)),
                    "beta": 0.005},
            "gsdmm": {"iterations": 100, "alpha": 0.1, "beta": 0.1},
        }
        self.params = {**defaults[method], **config.get(method, {})}
        self.n_topics_ = int(self.params.get("n_topics", config.get("n_topics", 32)))
        self.seed = int(config.get("seed", 42))
        self.info: dict[str, Any] = {}
        self.vocabulary_: dict[str, int] = {}
        self.model_assets_: dict[str, bytes] = {}

    def _encode(self, texts: list[str]) -> list[list[int]]:
        return [[self.vocabulary_[word] for word in topic_tokens(text)
                 if word in self.vocabulary_] for text in texts]

    def _write_documents(self, path: Path, rows: list[list[int]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                if self.method == "lda":
                    counts = Counter(row)
                    handle.write(str(len(counts)))
                    for word, count in sorted(counts.items()):
                        handle.write(f" {word}:{count}")
                else:
                    handle.write(" ".join(map(str, row)))
                handle.write("\n")

    def _java_command(self, directory: Path) -> list[str]:
        return [str(portable_java()), "-Dfile.encoding=UTF-8",
                f"-XX:ActiveProcessorCount={int(self.config.get('threads', 4))}",
                "-Xmx2g", "-cp", str(directory / "classes"), "main.Benchmark"]

    def fit(self, texts: list[str], sample_weight: np.ndarray | None = None) -> NativeTopicModel:
        if sample_weight is not None:
            raise ValueError("Sample weighting is only supported by the NMF adapter")
        started = time.perf_counter()
        self.vocabulary_ = _vocabulary(texts, int(self.params.get("max_features", 20000)))
        self.components_ = np.zeros((self.n_topics_, len(self.vocabulary_)))
        self.info = {"method": self.method, "seed": self.seed, "num_topics": self.n_topics_,
                     "vocabulary_size": len(self.vocabulary_), "training_documents": len(texts),
                     "backend": {"lda": "original LDA-C variational EM",
                                 "btm": "original BTM C++ Gibbs with sum_b inference",
                                 "gsdmm": "original Java GSDMM collapsed Gibbs"}[self.method],
                     "oov_policy": "zero row for empty or entirely unseen vocabulary",
                     "threads": 1, "params": dict(self.params)}
        if not self.vocabulary_:
            self.info.update(empty_training=True, fit_seconds=time.perf_counter() - started)
            return self
        manifest = prepare_native_tools(self.method)
        build = Path(manifest["build_dir"])
        directory = self.work_dir / f"native_{self.method}"
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite native training assets: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        encoded = [row for row in self._encode(texts) if row]
        self.info["training_token_occurrences"] = sum(map(len, encoded))
        if self.method == "btm":
            self.info["training_biterms"] = sum(
                14 * len(row) - 105 if len(row) >= 15 else len(row) * (len(row) - 1) // 2
                for row in encoded
            )
            self.info["biterm_window"] = 15
        self._write_documents(directory / "train.txt", encoded)
        (directory / "vocabulary.json").write_text(
            json.dumps(self.vocabulary_, ensure_ascii=False, indent=2), encoding="utf-8")
        k, v = self.n_topics_, len(self.vocabulary_)
        native_started = time.perf_counter()
        if self.method == "lda":
            shutil.copy2(build / "lda.exe", directory / "lda.exe")
            settings = (
                f"var max iter {int(self.params.get('var_max_iter', 20))}\n"
                "var convergence 0.000001\n"
                f"em max iter {int(self.params.get('em_max_iter', 100))}\n"
                "em convergence 0.0001\nalpha estimate\n"
            )
            (directory / "settings.txt").write_text(settings, encoding="ascii")
            (directory / "model").mkdir(exist_ok=True)
            command = [str(directory / "lda.exe"), "est", str(self.params.get("alpha", 1.0)),
                       str(k), "settings.txt", "train.txt", "random", "model", str(self.seed)]
            _run(command, directory, directory / "train.log")
            self.components_ = np.exp(np.loadtxt(directory / "model" / "final.beta", ndmin=2))
            files = ["lda.exe", "settings.txt", "model/final.beta", "model/final.other"]
        elif self.method == "btm":
            shutil.copy2(build / "btm.exe", directory / "btm.exe")
            (directory / "model").mkdir(exist_ok=True)
            iterations = int(self.params.get("iterations", 100))
            command = [str(directory / "btm.exe"), "est", str(k), str(v),
                       str(self.params.get("alpha", 50 / k)), str(self.params.get("beta", 0.005)),
                       str(iterations), str(max(1, iterations)),
                       "train.txt", "model/", str(self.seed)]
            _run(command, directory, directory / "train.log")
            self.components_ = np.loadtxt(directory / "model" / f"k{k}.pw_z", ndmin=2)
            files = ["btm.exe", f"model/k{k}.pz", f"model/k{k}.pw_z"]
        else:
            shutil.copytree(build / "classes", directory / "classes", dirs_exist_ok=True)
            command = self._java_command(directory) + ["est", str(k), str(v),
                       str(self.params.get("alpha", 0.1)), str(self.params.get("beta", 0.1)),
                       str(int(self.params.get("iterations", 100))), str(self.seed),
                       "train.txt", "model.bin", "phi.txt"]
            _run(command, directory, directory / "train.log")
            self.components_ = np.loadtxt(directory / "phi.txt", ndmin=2)
            files = ["model.bin"] + [p.relative_to(directory).as_posix()
                                     for p in (directory / "classes").rglob("*.class")]
        if self.components_.shape != (k, v) or not np.isfinite(self.components_).all():
            raise RuntimeError(f"Invalid {self.method} topic-word matrix")
        self.model_assets_ = {name: (directory / name).read_bytes() for name in files}
        self.info.update(build=manifest, fit_seconds=time.perf_counter() - started,
                         native_fit_seconds=time.perf_counter() - native_started,
                         nonempty_training_documents=len(encoded),
                         model_asset_bytes=sum(map(len, self.model_assets_.values())),
                         model_assets_sha256=hashlib.sha256(
                             b"".join(self.model_assets_[key] for key in sorted(self.model_assets_))
                         ).hexdigest())
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        encoded = self._encode(texts)
        result = np.zeros((len(texts), self.n_topics_))
        indices = [index for index, row in enumerate(encoded) if row]
        if not indices or not self.model_assets_:
            return result
        # Every invocation has isolated files. Nothing writes into the frozen model.
        with tempfile.TemporaryDirectory(prefix=f"cst-{self.method}-") as temporary:
            directory = Path(temporary)
            for name, content in self.model_assets_.items():
                destination = directory / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            self._write_documents(directory / "input.txt", [encoded[index] for index in indices])
            if self.method == "lda":
                command = [str(directory / "lda.exe"), "inf", "settings.txt", "model/final",
                           "input.txt", "output"]
                output = directory / "output-gamma.dat"
            elif self.method == "btm":
                command = [str(directory / "btm.exe"), "inf", "sum_b", str(self.n_topics_),
                           "input.txt", "model/", "output.txt"]
                output = directory / "output.txt"
            else:
                command = self._java_command(directory) + [
                    "inf", "model.bin", "input.txt", "output.txt"]
                output = directory / "output.txt"
            _run(command, directory)
            values = np.loadtxt(output, ndmin=2).reshape(len(indices), self.n_topics_)
            result[indices] = normalise_rows(values)
        return result

    def topic_words(self, top_n: int = 10) -> list[list[str]]:
        return _topic_words(self.components_, self.vocabulary_, top_n)


class NMFTopicModel:
    """TF-IDF + sklearn NMF, MU/nndsvda, with scale-correct topic memberships."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.params = {"max_iter": 300, "tol": 1e-4, **config.get("nmf", {})}
        self.n_topics_ = int(self.params.get("n_topics", config.get("n_topics", 32)))
        self.vocabulary_: dict[str, int] = {}
        self.info: dict[str, Any] = {}
        self.nmf_: NMF | None = None

    def fit(self, texts: list[str], sample_weight: np.ndarray | None = None) -> NMFTopicModel:
        started = time.perf_counter()
        self.vocabulary_ = _vocabulary(texts, int(self.params.get("max_features", 20000)))
        self.components_ = np.zeros((self.n_topics_, len(self.vocabulary_)))
        self.info = {"method": "nmf", "backend": "sklearn.decomposition.NMF",
                     "version": sklearn.__version__, "solver": "mu", "init": "nndsvda",
                     "num_topics": self.n_topics_, "vocabulary_size": len(self.vocabulary_),
                     "training_documents": len(texts), "seed": int(self.config.get("seed", 42)),
                     "risk_weighted_topics": sample_weight is not None,
                     "weighting": "multiply training rows by sqrt(sample_weight)",
                     "scale_correction": "row-normalize W times the L1 norm of each H row",
                     "oov_policy": "zero row"}
        if not self.vocabulary_:
            self.info.update(empty_training=True, fit_seconds=time.perf_counter() - started)
            return self
        self.vectorizer_ = TfidfVectorizer(
            tokenizer=topic_tokens, token_pattern=None, lowercase=False,
            vocabulary=self.vocabulary_, sublinear_tf=True,
        )
        matrix = self.vectorizer_.fit_transform(texts)
        train_matrix = matrix
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float64)
            if (weights.shape != (len(texts),) or not np.isfinite(weights).all()
                    or np.any(weights <= 0)):
                raise ValueError("Expected one positive finite weight per training document")
            train_matrix = matrix.multiply(np.sqrt(weights)[:, None]).tocsr()
        self.effective_topics_ = min(self.n_topics_, *train_matrix.shape)
        self.nmf_ = NMF(
            n_components=self.effective_topics_, init="nndsvda", solver="mu",
            random_state=int(self.config.get("seed", 42)),
            max_iter=int(self.params.get("max_iter", 300)),
            tol=float(self.params.get("tol", 1e-4)),
        )
        self.nmf_.fit(train_matrix)
        self.components_[:self.effective_topics_] = self.nmf_.components_
        self.factor_scale_ = self.nmf_.components_.sum(axis=1)
        self.info.update(effective_topics=self.effective_topics_, n_iter=int(self.nmf_.n_iter_),
                         fit_seconds=time.perf_counter() - started,
                         max_iter=int(self.nmf_.max_iter), tol=float(self.nmf_.tol))
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self.n_topics_))
        if not texts or self.nmf_ is None:
            return result
        matrix = self.vectorizer_.transform(texts)
        nonempty = np.flatnonzero(np.asarray(matrix.getnnz(axis=1)) > 0)
        if len(nonempty):
            membership = self.nmf_.transform(matrix[nonempty]) * self.factor_scale_[None, :]
            result[nonempty, :self.effective_topics_] = normalise_rows(membership)
        return result

    def topic_words(self, top_n: int = 10) -> list[list[str]]:
        return _topic_words(self.components_, self.vocabulary_, top_n)
