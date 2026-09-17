# Implementation notes

## Confirmed environment
- Windows PowerShell, Python project under cst_mil; MinGW gcc/g++ available.
- Java runtime present but javac missing; portable JDK needed.
- Original run_001 source fingerprint includes pyproject/uv.lock; archive original runtime before editing.
- SCCD only677 sessions; old balanced sampler cannot select all677 and changing size reassigns splits.
- Full split additions: train122CB+104NonCB; dev41CB+34NonCB; test41CB+35NonCB.
- Exact duplicate roots uG75nkrjg-BbeS8D and gTwPptSnukhvQ67e must join existing train roots.
- No topic-count or topic-boundary labels; NPMI/diversity/coverage are proxies, constructed fixtures engineering checks only.

## Interface contract
- New package cst_mil.bench; legacy modules remain unchanged.
- root common.py supplies topic_tokens, build_windows, normalise_rows, JSON/hash helpers.
- Topic backends implement fit(texts, sample_weight=None), transform(texts), topic_words(top_n=10); .info dictionary records fitted metadata. Transform returns nonnegative normalized rows, zeros for unassigned; stable topic column order.
- NativeTopicModel(method, config, work_dir); NMFTopicModel(config); BERTopicModel(config, work_dir).
- model.py provides train_model(sessions:list[dict], messages:list[dict], method:str, config:dict, work_dir:Path) -> BenchmarkModel; predict_session(model, session, messages) -> JSON-safe dict.
- BenchmarkModel includes method, config, threshold, training_summary, dev_predictions, topic_model. Prediction fields post_id, split, label, prediction, risk_probability, topics, evidence, message_scores, coverage; metadata copied by root runner.
- config shared keys seed=42, n_topics=32, threads=4, window_size=5, stride=2, word_features=False, risk_weighted_topics=False, multiscale=False. Method-specific overrides in config[method].
- build_windows(session,messages,size=5,stride=2) returns list of dict window_id,post_id,text,message_ids; fallback root ID is post_id+':root'.
- topic_tokens uses independent topic-only cleaning; risk text normalization remains legacy normalize_text.

## Verified engineering progress
- 61 tests passed before final integrity hardening; 16 additional isolated resume-guard tests passed.
- All six methods really fit the same smoke5 (2 train/2 dev/1 test;256 comments) and passed independent-process 2->5->5 plus same-model prediction comparisons.
- Initial smoke evidence is preserved at cst_mil/benchmark/quarantine/smoke_pre_integrity_hardening_20260913 after an audit required stronger dataset/encoder integrity checks. Repeat final smoke before formal pilot.
- Cohort load now recomputes full manifest+sessions+messages hash and validates raw source/original split hashes.
- NPMI/diversity use common split-window corpus; coverage is micro message count, with macro session coverage reported separately.
- runtime source binding includes evaluation, data, runner, campaign, CLI, verification, pyproject and lock; only reporting is intentionally excluded.
- BERTopic external snapshot/cache live under D:/github/intern/benchmark/models and benchmark/cache, while native/JDK resources live under cst_mil/benchmark/tools.
- Original40 protected file hashes verified unchanged. No full677 formal model trained.
- Final integrity-hardened runtime fingerprint: d748ff50d4dead47007928726c61a03d4726b56dd88c15037da079cc7b0f974c.
- 86 final tests passed; all six final smoke5 methods passed. Initial smoke remains archived, not mixed with final artifacts.
- Formal pilot300 started2026-09-13 local02:38, sequential LDA->BTM->GSDMM->NMF->BERTopic->CSTr0 and dev-gated revisions. Driver logs cst_mil/benchmark/logs.

## CPU resource-policy repair (supersedes earlier current-runtime note)
- First pilot interrupted before BERTopic training committed because measured CPU activity was11.9034core-equivalents despite4-thread library settings. Initial four completed baselines and all partial evidence retained at benchmark/quarantine/pre_cpu_affinity_20260913, excluded from final comparison/selection.
- New resources.py sets passive waits and OS-level process-tree affinity to logicalCPUs[0,1,2,3]; runner stores policy and rejects observed affinity width>4.
- 101 final tests and all six CPU-capped smoke5 runs pass. Current runtime hash d157911c2e395e6afa04513c146e36c4305678d57b7f05d41b325de0dc2ed973.
- Fair replacement pilot auto driver running in exec session40085. Source/core is frozen again; reporting-only changes remain allowed. Data/seed/method hyperparameters unchanged. Test locked until selection; full677 never run.

## Final dev selection (test was not used)
- Baselines dev macroF1: LDA.7147, BTM.8500, GSDMM.8141, NMF.8333, BERTopic.8833. ReferenceBERTopic.
- CSTr0 F1.8666667/NPMI-.7246508/coverage.8394631; r1 sameF1andtopics (APslightlybetter, not selectioncriterion); r2 sameF1/NPMI-.7154363/coverage.7310067 (fails coverage gate); r3 F1.8653199/sameNPMI/coverage.8416107 (no required improvement).
- selection.json frozen2026-09-13 local03:17:49 ->r0, stop_target_met=false, test_used_for_selection=false. All3additional attempts retained. Actual F1gap about.016634 (referenceF1.883301, not exact.883333) exceeds.01 target.
- All300 predictions for fivebaselines+r0/r1/r2/r3 complete; driver40085 generating comparison. No full677 training calls.
- Independent audit:79checks /9validruns /61evidencefilehashes pass. Six user commands checked with17source/schema checks; full677status reports677sessions and methods{}. Protected40files unchanged, uv offline lock check passes72packages.
- Frozenr0 testF1.8316498, CBrecall.7333333; BERTopicF1.8331479, CBrecall.8666667. Testtopiccounts:0=0,1=11,2=16,3=12,>3=21,max7,mean3.0333. Variableoutputs do not establish correct semantic topic counts or three named risk categories.

## Delivery
- cst_mil/benchmark/操作指南.md and cst_mil/benchmark/comparison/ reports contain commands, numerical results, limitations and smoke evidence.
- Final ready2026-09-13: benchmark/READY.json ready=true;102tests/0fail/0error/0skip,79iterationchecks,17commandchecks,6smokes,40protectedfiles unchanged. Finalcodehash remains d157911c2e395e6afa04513c146e36c4305678d57b7f05d41b325de0dc2ed973.
- Reporting regenerated after final wording/denominator changes. Actualrisk/topic/costfigures visually inspected;3PNGs+3vectorPDFs,CSVandJSON contain9configurations. Finalpytestlogs tests_1789241743877630000.{xml,log};ruffcheck/uvlock--check--offline/uvbuild--offline pass. Wheel28entries includesCPUresourcesmodule and noexperimentmodels/nativebinaries/data.
- Frozenr0vsBERTopictestF1diff-.0014981105,paired95%CI[-.1065,.1004]. CSTr1/r2/r3testF1.8490/.8661/.8667 areexploratory, do notoverridefrozendevselectionr0. r0time train16.871s/all300inference23.496s,peak326.844MiB. BERTopiccacheconditionsandnativeI/Ooverheadsdisclosed.
- Full677 manifestready,canonicalfullrunabsent,formaltraininguser-only. Finalhandoffguideand验收记录.md retained;READYfileUTF8needsnormalJSONreadratherthaninterpretingconsoleencoding.

## Full677 completed by user and compared (2026-09-15)
- All six runs complete677/677,test136/136,fresh76/76. Comparison regenerated at benchmark/comparison/full677 after report-schema review; CSV/statistics/raw metrics consistency audit passed65numeric checks.
- Overall test F1/Recall/AP: LDA .8088/.7887/.9174; BTM .7983/.8873/.9161; GSDMM .8078/.8451/.8599; NMF .7868/.7606/.9265; BERTopic .8239/.9859/.8530; CST-r0 .8304/.8451/.9465. CST AUC .9413 highest.
- Fresh76 F1/Recall/AP: BERTopic .8476/1.0/.9058; CST .8137/.8537/.9460. Seen60 best F1 GSDMM .8661; CST .8500 and AP .9489. Report allmethods subgroup tables, not only CST-vs-NMF.
- Full dev referenceNMF .8738; CST .9256. Test paired CST-NMF F1delta+.04367996,95%CI[-.01594345,.10274802]; freshdelta+.03775878 CI[-.02774816,.10511170]; seen+.05018079 CI[-.04899477,.15026490]. All cross0; single seed, no superiority/significance claim.
- CST topic metrics equalNMF exactly (same topics SHA):NPMI-.4105,diversity.8813,coverage.7838. It cannot claim bettertopic discovery thanNMF. Topiccounts0..11,mean3.926,75/136>3; topic0generic appears112/136;1022/8028messages multiassigned;1391segments including265singletons. Variabletopic interface works but semantic accuracy/overfragmentation unresolved.
- CST evidence commentAP.3701(notbest),Top3Hit.6634(best;67/101sessions),Recall@3.0844(notbest). bN5... correctsession/highrisk but top3 all goldnonrisk illustrates evidence misalignment.
- Costs train/allinfer/testinfer seconds:CST67.68/102.26/21.24,NMF41.81/80.91/15.23,BER378.58/311.18/222.56. CST peak409.66MiB/model11.93MiB,noencoder;BER1157.25MiB/model71.29MiB+476.42MiBencoder,cachedtiming disclosed.
- Retained full errors: BTM one training and one inference CPU-affinity rejection, then complete; BER one userKeyboardInterrupt duringdevencoding, attempt2complete. Valid-result times exclude abortedattemptcosts. Report now separates these from oldpilot quarantine.
- Reporting-only patch adds per-method fresh/seen statistics and correct fullnextstep/history wording.11 reportingtests+ruff passed; final allsuite103tests pass with expected syntheticNMF warnings. Full risk/topic/cost PNGs visually inspected.
