# Modules 8 to 14: what exists, where, and what is still open

Mapping of every sub-task in *Module Split v2: Fatigue Software with ML + LLM/RAG Layer* to the repository.
Status words: **done** (built and tested), **done, with a caveat**, **open** (not finished, and why).
Numbers quoted here are reproduced by the commands in the last column; they are not hand-entered.

The design rule behind all of it: *ML measures, the LLM explains.* The fatigue estimate comes from the pipeline
models (`ml/dataset_pipeline`, Modules 1 to 7). The language model only turns already-computed numbers into
sentences, and every number in its text is checked against those computed values.

## Module 8: knowledge base for retrieval

| Task | What | Status | Check |
|---|---|---|---|
| 8.1a physiology notes written by the team | `knowledge/physiology/*.md` (8 notes), `knowledge/scales/`, `knowledge/clinical/`, `knowledge/safety/` | done | `GET /api/rag/sources` |
| 8.1b data dictionary and protocol | `knowledge/protocol/wsd4fedsrm_protocol.md`, `shoulder_biomechanics.md`, `knowledge/protocol/dataset_audit_findings.md` (what is wrong with which recordings) | done | |
| 8.1c paper summaries with source and licence | `knowledge/literature/papers_summary.md` | done, with a caveat: abstracts and snippets were read, not the full papers | open each paper before citing |
| 8.1d per-person facts from past sessions | `ml/rag/person_facts.py`: usual time to maximal effort, usual build-up rate, starting muscle frequency and pulse, strength. Built from the pipeline tables and offered to the coach as chunk `PERSON-Pxx`; the numbers are registered with the summary so the checker can verify them | done | `tests/test_live_pipeline.py` |
| 8.2 short self-contained chunks with title, source, date | markdown front matter (`chunk_id`, `title`, `category`, `source`, `date`, `tags`) | done | |
| 8.3 embedding model and local vector store | sentence-transformers `all-MiniLM-L6-v2` when installed, TF-IDF otherwise; the index is an in-memory matrix (about 30 chunks), not FAISS or Chroma, which would add a dependency without a benefit at this size | done, with a caveat | `GET /api/health` shows the backend |
| 8.4 search returning chunks with source tags | `LocalRAGKnowledgeBase.search`, `KnowledgeChunk.format_citation` | done | |
| 8.5 20 test questions with expected chunk | `LLMLayerEvaluator.RETRIEVAL_TEST_QUERIES` | done, with a caveat: query-expansion terms were tuned after seeing misses on these questions, so the score is optimistic, not held out | `python -m ml.dataset_pipeline.evaluate_llm_layer` |

## Module 9: structured summaries and explanation engine

| Task | What | Status |
|---|---|---|
| 9.1 summary format | `ml/rag/session_summary.py::SessionSummary`: person, session, time span, fatigue index, trend, change from the person's baseline per feature, quality flags, calibration status, model used | done |
| 9.2 builder from pipeline outputs | `build_session_summary_from_pipeline` (generic) and `PipelineStore.build_summary` (from the session tables: smoothed estimate, drivers, percent change from the person's own start, quality flags, calibration status) | done |
| 9.3 prompt templates | `ml/rag/hybrid_router.py`: live explanation (2 to 4 sentences), answer-a-question (intent-shaped), session report (`GET /api/pipeline/report`) | done |
| 9.4 prompt rules | use only numbers from the facts, cite retrieved notes by ID, say what cannot be told from the reading | done |
| 9.5 swappable model | one call interface: custom callable, Claude (`ANTHROPIC_API_KEY`), local Ollama, rule-based template | done |
| 9.6 explanations for 10 recorded sessions, read by a person | `reports/llm_eval/ten_session_explanations.md` (20 explanations next to their numbers, with the number check result) | done, with a caveat: rule-based text unless a model is configured |

Trend wording is judged against **the person's own usual build-up** (`usual_rate_per_min`): in this protocol effort
climbs about 10 Borg points a minute for everybody, so the earlier fixed "2 points a minute = sharp" rule marked
every window as a sharp rise.

## Module 10: hybrid routing and guardrails

| Task | What | Status |
|---|---|---|
| 10.1 triggers | `HybridRouter.should_trigger_explanation`: warning (15) or stop (18) level, sharp rise, the person asks why, session end; everything else stays machine-only | done |
| 10.2 the loop never waits for the model | `LivePipeline` hands windows to `ExplanationService` (worker thread); the API starts a job and the page polls (`POST /api/pipeline/explain`, `GET .../explain/{job}`) | done; `test_the_live_loop_never_waits_for_the_language_model` |
| 10.3 number checker | `ml/rag/number_checker.py`: every number in the text must be in the summary (or a scale constant). Wrong text is sent back once, then replaced by the template | done, with a caveat: whole numbers 0 to 20, 50 and 100 are always allowed, so an invented count in that range would pass |
| 10.4 safety rules | no diagnosis or treatment words (whole-word match), standard note on every answer, out-of-scope questions (blood pressure, injury, sleep, diet, ...) are declined without calling a model | done |
| 10.5 fallback | deterministic template built from the numbers when the model is off, errors, or fails the check | done |
| 10.6 log of every call | `HybridRouter(call_log_path=...)`: JSONL with inputs (the summary), query, output, retrieved chunk IDs, number-check result, attempts, delay. The API writes `reports/llm_call_log.jsonl` (`FATIGUE_CALL_LOG` overrides) | done |

## Module 11: software core

| Task | What | Status |
|---|---|---|
| 11.1 data-source interface | `DataSource` in `ml/dataset_pipeline/live.py`: `FileReplaySource` (now), `SyntheticSource` (simulator), `SerialSource` (placeholder for hardware) | done; the hardware reader itself is **open** (needs a device) |
| 11.2 streaming window processor | `LivePipeline`: 1 s chunks, 10 s windows, the same `window_feats` as the offline tables | done |
| 11.3 inference, smoothed index, alerts | generalised and personalised estimates, exponential smoothing, warning and stop alerts once each | done |
| 11.4 calibration flow for a new person | first 2 windows form the baseline; after `K` windows (default 4, 40 s) the personalised estimate starts; `calibration_state` reports progress | done |
| 11.5 session logging | JSONL per session (`reports/sessions/`) | done |
| 11.6 explanation as a background service | `ExplanationService` and the API jobs | done |
| 11.7 replay equals offline | `python -m ml.dataset_pipeline.verify_replay`: six recordings, worst difference between live replay and offline estimates is written to `m11_replay_equality.csv` | done |

## Module 12: dashboard

All of it is the **Session lab** tab (key `6`) of the existing dashboard, which stays as it was. It reads
`/api/pipeline/*`, so it needs only the committed pipeline tables, not the raw dataset.

| Task | Where | Status |
|---|---|---|
| 12.1 sketch the pages first | `docs/dashboard_wireframe.md` (a text wireframe; a paper sketch is for a person to do) | done, with a caveat |
| 12.2 session picker, live signal strips | picker and replay controls in the Session lab; the raw EMG, IMU and PPG strips are in the existing *Live monitor* tab, which replays the raw recording and therefore needs the dataset | done, with a caveat |
| 12.3 gauge with colour bands and a trend line | gauge (green, amber at 15, red at 18) and the effort chart with the zones | done |
| 12.4 predicted versus self-reported | chart of the athlete's own rating, generalised and personalised estimates | done |
| 12.5 feature trends against the person's own baseline | percent change from the first 20 seconds, 8 signals, switchable | done |
| 12.6 profile card and data-quality panel | profile list, quality score, per-window channel flags | done |
| 12.7 model toggle and top features | Generalised / Personalised switch; top three drivers per window (occlusion against the saved model, `explain_windows.py`). Next to the switch the page states the measured result of personalising on the first 40 s (no significant gain over the generalised estimate), so the toggle is an alternative view, not an upgrade. Elapsed time is the top driver in most windows, which the page shows as it is | done |
| 12.8 explain panel with sources and a "generated" label | "Generated text" badge, who wrote it, trigger, delay, numbers it was allowed to use, sources with snippets, disclaimer | done |
| 12.9 session report with charts, exportable | `GET /api/pipeline/report/{sid}/{task}` (HTML with an inline chart, or Markdown) | done |
| 12.10 ask your data from fixed read-only lookups | `POST /api/pipeline/ask`: eight lookups (peak effort, threshold times, accuracy, model comparison, biggest change, data quality, duration, exercise comparison); anything else is refused and the supported questions are listed | done |
| 12.11 report export | the two buttons in the explanation card | done |
| "a non-technical person can read it without explanation" | plain-language labels throughout; **not tested with people** | **open** |

## Module 13: evaluation of the explanation layer

Run `python -m ml.dataset_pipeline.evaluate_llm_layer`; results in `reports/llm_eval/llm_layer_evaluation.md`.

| Task | Status |
|---|---|
| 13.1 faithfulness on 50 explanations | done on 50 **real** windows (the earlier test used random synthetic summaries). Plus a guardrail test: a model that always invents a pulse and a percentage must be caught and replaced |
| 13.2 questions the data cannot answer | done (10 questions) |
| 13.3 retrieval quality | done (optimistic, see 8.5); plus the share of answers that cite a retrieved note |
| 13.4 usefulness ratings from 3 to 5 people | **open**: `usefulness_rating_sheet.csv` is generated (12 numbers-only versus numbers-plus-explanation items); `python -m ml.dataset_pipeline.aggregate_ratings` summarises the returned sheets. No ratings were invented |
| 13.5 cost and speed, always-call versus selective | done for call counts over all 204 recorded sessions and for the rule-based delay. Language-model delay and cost are **open**: no model was configured when this was run |
| 13.6 ablation ML-only versus ML + LLM | readability proxy only (Flesch score, length). The usefulness comparison depends on 13.4 and is **open** |

## Module 14: testing and hardware readiness

| Task | What | Status |
|---|---|---|
| 14.1 unit tests for filters, windowing, features, number checker | `tests/test_dataset_pipeline.py` (run detection, gap repair, time split, baseline, z-score, features), `tests/test_live_pipeline.py`, `tests/test_session_lab.py`, and the existing suite (`test_preprocessing`, `test_feature_engineering`, `test_rag_and_guardrails`, ...) | done |
| 14.2 processing delay of the ML loop | `window_processing_ms` in `python -m ml.dataset_pipeline.e2e_run` (feature extraction plus both models per 10 s window) | done |
| 14.3 hardware contract | `docs/hardware_contract.md`, with section 4 checked against the recordings (EMG reaches 3.9 mV, which the original +-2.5 mV range would clip) and `CONTRACT` / `validate_chunk` in code | done |
| 14.4 hardware simulator | `SyntheticSource` (spectrum, amplitude, tempo and pulse drift with a known effort curve; can inject dropouts and spikes) and the older `ml/hardware/simulator.py` | done; synthetic, so it validates plumbing and not accuracy |
| 14.5 whole system with the simulator, with and without the model | `python -m ml.dataset_pipeline.e2e_run`: model off, a scripted stand-in model, and a stand-in that invents numbers; report in `reports/dataset_modules_1_7/m14_e2e_report.md` | done with a stand-in model; a real model needs a key or a local server (`--real`) |

## What is not finished, in one place

1. Real hardware: `SerialSource.read_chunk` and a causal front-end filter validated on a device.
2. Human evaluation: usefulness ratings (13.4), the human ablation (13.6), and a read-without-explanation check of the dashboard (12).
3. Language-model measurements: delay, cost and faithfulness of a real model (13.1, 13.5).
4. Cross-day personalisation: the dataset has one session per person, so only within-session calibration could be tested.
5. The estimate is a research guide, not a measurement: leave-one-person-out error about 1.7 Borg points, and every sensor-based gain over "elapsed time alone" is modest (see `docs/dataset_pipeline_modules_1_7.md`).
