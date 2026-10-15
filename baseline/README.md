# Optional research baselines

The default semantic demo does not require native backends. They remain
separate from the released model and are used for research comparisons.

| Method | Backend | Preparation |
|---|---|---|
| LDA | Original LDA-C plus deterministic/portability edits | Pinned download below; C compiler |
| BTM | Apache-2.0 C++ source with project numerical/reproducibility edits | `BTM/`; C++ compiler |
| GSDMM | Original Java model plus frozen-count bridge | Pinned download below; JDK |
| NMF | scikit-learn | Python dependency |
| BERTopic | MIT tag v0.17.4 source | `BERTopic/`; benchmark dependency group |

From the repository root:

```bash
python baseline/setup.py --methods lda gsdmm --accept-upstream-terms
```

The explicit flag acknowledges that you reviewed upstream terms; it does not
create an upstream license. LDA-C has conflicting LGPL/GPL notices, and pinned
GSDMM has no LICENSE. Archives come from fixed commits, are SHA-256 checked,
patched locally, and saved under ignored `LDA/` and `GSDMM/` paths. Pinned
metadata is in [`sources.json`](sources.json). The installer runs no compiler.

Offline installation from previously obtained pinned archives:

```bash
python baseline/setup.py --methods lda gsdmm --accept-upstream-terms --archives-dir /path/to/archives
```

Expected archive names are `<method>-<commit>.zip`. Existing installations are
verified; modified files are refused rather than silently overwritten.

Install Python comparison dependencies with `uv sync --group benchmark` in
`cst_mil/`. Datasets and prepared experimental splits are not shipped.
See the research CLI help and main README for preparation and training stages.
Component terms are in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).
