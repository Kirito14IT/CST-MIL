# CST-MIL benchmark implementation

## Goal
Implement the approved variable-topic CST-MIL and five faithful SCCD baselines; run one pilot300 experiment per baseline and r0 plus at most three development-selected CST revisions; prepare full677 without starting its formal training; verify all six methods on at most five smoke sessions and hand off individual resumable commands.

## Phases
- [x] Inspect workspace, approved plan and old provenance.
- [x] Preserve legacy source/data/report integrity; prepare tools and dependencies.
- [x] Implement shared data protocol, native baselines, BERTopic, CST and resume runner.
- [x] Run targeted tests and six-method smoke checks.
- [x] Run baseline pilot300 and CST r0; execute up to r1-r3 using dev-only gates.
- [x] Freeze winner, evaluate retained models, create comparison report and figures.
- [x] Verify full677 manifests, final smoke/restart checks, commands and preservation.

## Locked decisions
- Pilot300 = original 180 train / 60 dev / 60 test, 17,312 comments.
- Full677 = nested 406 train / 135 dev / 136 test, 38,872 comments. New test76 separate.
- Seed42; CPU only, 4 threads maximum. Standard BERTopic with cached multilingual MiniLM; CST no pretrained models.
- Window5 stride2 plus final tail; model capacity32 is not per-session topic count.
- Original raw/prepared/run_001/root reports and DOCX unchanged. New work in bench modules, baseline adapters and benchmark workspace.
- Formal full677 fitting was reserved for the user's manual launch after the pre-launch handoff; the user subsequently completed all six methods.
- Fixed-model inference resumes atomically by complete post_id; limit is cumulative prediction count.
- Formal tests unavailable for selection until revision freeze. Revisions r1 word features; r2 risk-weighted topics; r3 window10 equal fusion, based on current best.

## Status
Implementation, pre-launch delivery, and the user's subsequent full677 runs are complete. LDA/BTM/GSDMM/NMF/BERTopic/CST-r0 each completed677/677, including136/136 test and76/76 fresh-test sessions. The validated full comparison is under benchmark/comparison/full677. CST-r0 test F1/AP/AUC=.8304/.9465/.9413; fresh76 F1/AP=.8137/.9460. It has the highest overall test F1/AP/AUC point estimates and Top-3 hit rate, while BERTopic has higher recall and the best fresh76 F1. CST-vs-dev-best-NMF paired F1 delta=.04368,95%CI[-.01594,.10275], so no significant/stable superiority claim. Final post-report suite:103tests pass and Ruff passes; all three full figures visually inspected. READY.json and benchmark/验收记录.md remain the dated pre-launch certificate/record and are marked historical in the guide/record.

## Errors and resolutions
- Existing system pytest temp ACL failure: use unique workspace-local --basetemp paths.
- BERTopic English-default preprocessing removed Chinese text: fixed multilingual configuration.
- Fast-tokenizer prepare_for_model incompatibility: explicit token-ID chunking preserves every token.
- Synthetic NMF small-sample convergence warnings are retained in test logs, not hidden.
- Build default initially scanned local assets; explicit source-distribution include list now excludes datasets/models.
- Final integrity review requires revalidating a reloaded external encoder and recomputing cohort manifest digest. Initial smoke runs are retained as pre-hardening engineering evidence; repeat final smoke after these guards, before formal pilot.
- Windows numerical libraries exceeded environment-only CPU thread limits. First pilot retained as non-reportable resource-policy trial; OS-level affinity will enforce the same resource ceiling for every method. Do not report this trial as a second random-seed replicate.
- CPU guard integration initially exposed an outdated mocked monitor width4 in 1-thread test fixtures; 13 mock-test failures retained in logs. Fixture corrected to width1; actual over-limit rejection has its own regression test.
