# NMF baseline

## SCCD adapter

`cst_mil/src/cst_mil/bench/native_topics.py::NMFTopicModel` uses the third-party scikit-learn NMF implementation. It is not presented as code maintained by the original paper's authors.

The adapter uses TF-IDF input, up to 32 topics, `init="nndsvda"`, `solver="mu"`, `max_iter=300`, `tol=1e-4` and seed 42. Topic-factor L1 scales correct the NMF weights before row normalization. Empty text and entirely unknown vocabulary produce a zero distribution. Small synthetic smoke tests record the effective factor count and pad the output to the configured width.

From the repository root, inspect the research command and prepare the required datasets and split manifests separately:

```powershell
cd cst_mil
uv run --frozen --group benchmark cst-bench --help
```

Dependency versions are pinned in the Python lockfile. A new run records the installed scikit-learn version in local metadata. Experimental predictions, metrics and run artifacts are not included in this repository. See [research preparation](../../README.md#research-code-and-reproducibility).

## Implementation and attribution

This directory does not vendor an implementation maintained by Daniel D. Lee or H. Sebastian Seung. The adapter uses [`sklearn.decomposition.NMF`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html), which provides the nonnegative matrix factorization used by the benchmark code.

The multiplicative-update solver is configured explicitly:

```python
from sklearn.decomposition import NMF

model = NMF(
    n_components=K,
    init="nndsvda",
    solver="mu",
    beta_loss="frobenius",
    random_state=SEED,
)
```

For text topic discovery, the input is a nonnegative TF-IDF matrix. The original papers' image and encyclopedia corpora are not distributed here.

## Original papers

1. Daniel D. Lee and H. Sebastian Seung. [Learning the parts of objects by non-negative matrix factorization](https://www.nature.com/articles/44565). *Nature*, 1999.
2. Daniel D. Lee and H. Sebastian Seung. [Algorithms for Non-negative Matrix Factorization](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html). *NeurIPS*, 2000.
