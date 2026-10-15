# Third-party components and data

The top-level MIT license covers original CST-MIL code and original visual
assets. It does not replace the following component or model terms.

| Component | Pinned source | Distribution here | Terms |
|---|---|---|---|
| BERTopic 0.17.4 | [MaartenGr/BERTopic, `75f2910562f0e372beee29acbbd2a2835ba72cf2`](https://github.com/MaartenGr/BERTopic/tree/75f2910562f0e372beee29acbbd2a2835ba72cf2) | Python source and packaging; upstream docs/images omitted | [Upstream MIT](baseline/BERTopic/LICENSE), copyright Maarten P. Grootendorst |
| BTM | [xiaohuiyan/BTM, `66cc9b475afec81f3e74bb393b874b3fe5d5a148`](https://github.com/xiaohuiyan/BTM/tree/66cc9b475afec81f3e74bb393b874b3fe5d5a148) | Native source with project reproducibility/numerical adaptations; no example dataset | [Upstream Apache-2.0](baseline/BTM/LICENSE.txt) |
| LDA-C | [blei-lab/lda-c, `0c46575e89092683010db077f713f8aa3b3594a2`](https://github.com/blei-lab/lda-c/tree/0c46575e89092683010db077f713f8aa3b3594a2) | Original source **not redistributed**; optional download plus edit manifest | `license.txt` contains LGPL-2.1, while source headers/readme state GPL-2.0-or-later. No conflict resolution is asserted. |
| GSDMM | [jackyin12/GSDMM, `1b344d9de28faee43ec0df0c3cf7ea57737fe14e`](https://github.com/jackyin12/GSDMM/tree/1b344d9de28faee43ec0df0c3cf7ea57737fe14e) | Original source/JAR **not redistributed**; optional download, edit manifest and original project bridge | No upstream LICENSE found at this commit; public access alone does not grant redistribution permission. |
| Multilingual MiniLM | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Downloaded separately by deployment setup, not committed | Upstream model card lists Apache-2.0; exact revision/files fixed by release catalog. |

BTM files modified by CST-MIL in 2026: `src/infer.cpp`, `src/infer.h`,
`src/main.cpp`, `src/model.cpp` and `src/pvec.h`. Their original Apache-2.0
terms remain in force.

Optional source downloads are explicit: `python baseline/setup.py --help`.
They are not run during `npm start`. The installer preserves upstream notices,
verifies archive and adapter hashes, and records local file identity.
Project adapters do not grant additional rights in upstream code.

Python and npm dependencies are installed under their respective licenses.
Lockfiles identify versions; no dependency environment is vendored.

Research uses SCCD, CONDA, ToxiCN, ScamSpot and HateBR. Raw data, downloaded
dataset files, personal analysis history and review notes are not distributed.
Full reproduction requires independently obtaining the indicated sources and
any necessary permissions. Campus demo inputs are authored functional examples,
not research test examples.

Three aggregate historical SCCD result files are retained only to render the
demo's explicitly labelled historical comparison page; they contain numerical
summaries, not original comments or per-message predictions.

Separately released CST-MIL weights have [MODEL_LICENSE.md](MODEL_LICENSE.md)
terms, which do not relicense MiniLM or data.
