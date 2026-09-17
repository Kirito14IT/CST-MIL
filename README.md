# CST-MIL

A CPU-only session-level cyberbullying risk detection prototype for the **SCCD** (Situation-aware Cyberbullying Conversation Dataset) Weibo conversations, with a unified benchmark of five classical topic-discovery baselines.

The repository contains:

- **CST-MIL** — the proposed method (Variable-topic Character/Structure Topic-aware Multiple-Instance Learning), small (~12 MiB), no pre-trained encoders, no GPU.
- **Five baselines** — LDA, BTM, GSDMM, NMF, BERTopic — adapted to the same SCCD data protocol.
- A reproducible **benchmark** workflow that runs all six methods on either a 300-session pilot or the full 677-session SCCD corpus.

> Research scope: session-level **CB / Non-CB** binary classification. Comment labels are used only for evaluation evidence, never as training features. CST-MIL learns discussion topics from the corpus and produces variable per-session topic counts; risk categories are not pre-defined.

---

## 1. Repository layout

```
intern/
├── README.md                       # this file
├── .gitignore
├── notes.md                        # working notes / engineering log
├── task_plan.md                    # implementation plan
├── 待办事项.md                       # TODO list
├── 任务进度报告20260901.md           # formal progress report (Sep 1, 2026)
│
├── baseline/                       # baseline original-code snapshots
│   ├── LDA/                        # David Blei LDA-C
│   ├── BTM/                        # Yan BTM
│   ├── GSDMM/                      # Yin GSDMM (Java)
│   ├── NMF/                        # scikit-learn NMF (no canonical author repo)
│   └── BERTopic/                   # Grootendorst BERTopic v0.17.4
│
└── cst_mil/                        # CST-MIL project + benchmark
    ├── README.md                   # CST-MIL specific README
    ├── notes.md
    ├── task_plan.md
    ├── report_evidence.md          # detailed run_001 evidence
    ├── pyproject.toml              # uv-managed project
    ├── uv.lock
    ├── src/cst_mil/                # source code
    │   └── bench/                  # six-method unified adapter
    ├── tests/                      # 100+ pytest tests
    ├── configs/
    ├── artifacts/run_001/          # original 300-session CST-MIL run (metrics only)
    └── benchmark/
        ├── 操作指南.md              # operational guide (Chinese)
        ├── 验收记录.md              # acceptance record
        ├── READY.json              # pre-launch freeze certificate
        ├── protocol.json
        ├── selection.json          # frozen dev selection
        ├── comparison/             # pilot300 & full677 comparison reports
        ├── verification/           # per-method verification logs
        ├── logs/                   # per-method driver logs
        └── revisions/              # CST r1/r2/r3 attempt records
```

**Excluded from this repository** (re-downloadable / too large / not source-of-truth):

- Raw SCCD data and prepared datasets (`data/`, `cst_mil/data/`, `cst_mil/benchmark/data/`)
- BERTopic weights and embedding cache (`benchmark/models/`, `benchmark/cache/`)
- Full experiment run artifacts, including trained models (`benchmark/runs/`, `benchmark/quarantine/`)
- JDK / native tools (`benchmark/tools/`)
- Virtual envs and pytest temp directories
- Documents and templates related to the host university internship

---

## 2. Methods at a glance

| Method | Core mechanism | Pretrained? | GPU? | Topic count |
|---|---|---|---|---|
| LDA | Document–word count + Dirichlet topic mixtures | No | No | Fixed (32) |
| BTM | Corpus-level biterm modeling | No | No | Fixed (32) |
| GSDMM | Per-doc single-topic multinomial-Dirichlet mixture (Gibbs) | No | No | Auto |
| NMF | TF-IDF + non-negative factorization | No | No | Fixed (32) |
| BERTopic | Multilingual MiniLM embedding + UMAP + HDBSCAN + c-TF-IDF | Yes (encoder) | Optional | Auto + outliers |
| **CST-MIL r0** | NMF topics + character risk + reply structure + session classifier (MIL aggregation) | No | No | **Variable per session** |

---

## 3. Key results

### Full 677-session SCCD run (406 train / 135 dev / 136 test + 76 fresh test)

**Test-set overall (136 sessions):**

| Method | Macro-F1 | CB Recall | Precision | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| LDA | 0.8088 | 0.7887 | **0.8358** | 0.9174 | 0.9224 |
| BTM | 0.7983 | 0.8873 | 0.7683 | 0.9161 | 0.9077 |
| GSDMM | 0.8078 | 0.8451 | 0.8000 | 0.8599 | 0.8789 |
| NMF | 0.7868 | 0.7606 | 0.8182 | 0.9265 | 0.9183 |
| BERTopic | 0.8239 | **0.9859** | 0.7609 | 0.8530 | 0.8769 |
| **CST-MIL r0** | **0.8304** | 0.8451 | 0.8333 | **0.9465** | **0.9413** |

**Highlights of CST-MIL r0 on full-677:**

- Best AP and ROC-AUC overall — strongest **risk ranking**.
- Balanced precision/recall (11 missed, 12 false alarms) vs BERTopic's 1 missed / 22 false alarms.
- **Variable topic counts**: 0–11 topics per session, mean ≈ 3.93. A single comment can belong to multiple topics, with segment + speaker + original-text evidence preserved.
- Best **Top-3 hit rate** on sessions that contain at least one risky comment (66.34%, 67/101).
- Moderate cost: 67.68 s train, 102.26 s for all-677 inference, 11.93 MiB model, **no pre-trained encoder**. ~10.5× faster test inference than BERTopic, ~64.6% lower peak memory.

Full statistics, fresh-76 / seen-60 breakdowns, paired-confidence intervals, topic metrics and cost tables:
[`cst_mil/benchmark/comparison/full677/analysis-report.md`](cst_mil/benchmark/comparison/full677/analysis-report.md).

> CST vs dev-best NMF paired F1 delta = +.0437, 95% CI [−.0159, .1027]. The CI crosses zero and a single seed was used, so **no superiority / significance claim** is made.

---

## 4. Quick start (pilot300 smoke)

```powershell
# from repo root
cd cst_mil

uv sync --frozen
uv run cst-mil download-data      # SCCD from STAIR-BUPT/SCCD pinned commit
uv run cst-mil prepare --seed 42 --sessions 300
uv run cst-mil train --config configs/pilot.yaml
uv run cst-mil evaluate --run artifacts/run_001
```

Or one-shot:

```powershell
uv run cst-mil run-pilot --config configs/pilot.yaml
```

---

## 5. Benchmark (six-method SCCD comparison)

Operational guide (Chinese): [`cst_mil/benchmark/操作指南.md`](cst_mil/benchmark/操作指南.md).
Acceptance record: [`cst_mil/benchmark/验收记录.md`](cst_mil/benchmark/验收记录.md).
Per-method verification logs: [`cst_mil/benchmark/verification/`](cst_mil/benchmark/verification/).

Six independent full-677 commands are listed in the operational guide. Each driver writes its results to `benchmark/runs/full677/<method>/` (excluded from git).

The benchmark freezes:

- Single seed (`42`), CPU only, ≤ 4 threads, OS-level affinity enforced.
- 32 topics (where the method requires a fixed K); 5-message window / stride 2 + tail.
- Same data protocol (`build_windows`, `normalise_rows`, `topic_tokens`, `hash`) across all six methods.

---

## 6. Engineering status

- 103 tests pass, `ruff check` and `uv lock --check --offline` pass.
- Six methods each completed 677/677 sessions (136/136 test, 76/76 fresh-test).
- All three full-677 figures (PNG + vector PDF) visually inspected.
- Pre-launch freeze certificate: `cst_mil/benchmark/READY.json` (`ready=true`).
- Dev selection: `cst_mil/benchmark/selection.json` — CST-MIL **r0** frozen on 2026-09-13 03:17:49.
- Final source fingerprint: `d157911c2e395e6afa04513c146e36c4305678d57b7f05d41b325de0dc2ed973`.

Notable engineering issues resolved during the project are documented in [`notes.md`](notes.md) and [`cst_mil/notes.md`](cst_mil/notes.md) (CPU affinity, BERTopic multilingual preprocessing, fast-tokenizer compatibility, integrity hardening, etc.).

---

## 7. Data license reminder

SCCD is consumed from [STAIR-BUPT/SCCD](https://github.com/STAIR-BUPT/SCCD), pinned to commit `cf4015b802cabd651885211dc382152c1a270c31`. As of the time of this work, the upstream repository does not publish a clear standalone LICENSE. Therefore the raw dataset is **not redistributed in this repository** and any derived data release must confirm upstream authorization first.

---

## 8. Limitations and next steps

- One fixed seed, one train/dev/test split. Generalization across seeds, user / event splits and time splits has not been demonstrated.
- SCCD covers Weibo cyberbullying only — does not generalize to multi-turn group chat, fast drift, or other risk domains.
- CST-MIL's topic metrics equal NMF exactly (same topic SHA); the variable interface works but semantic accuracy and over-fragmentation remain unresolved.
- Evidence localization (comment-level AP) is still weak — Top-3 should be read as "high-risk context candidates", not strict single-sentence attribution.
- Next steps: clarifying AP / ROC-AUC definitions, comparing variable-topic support across methods, listing real Top-3 data per method, comprehensive cost analysis, ablation on the linear risk head, and writing a topic-discovery survey.

See [`待办事项.md`](待办事项.md) for the current TODO list and [`任务进度报告20260901.md`](任务进度报告20260901.md) for the most recent formal progress report.