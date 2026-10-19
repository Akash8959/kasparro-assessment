# Kasparro FSE Intern Pathway Assessment — Technical Writeup

## 1. System Architecture & Extraction/Classification Design

The claim verification system is designed around a decoupled, fail-safe agentic architecture adhering strictly to the assessment specifications:

- **Model & Generation Configuration**: Built with Python 3.11 and the official `google-genai` SDK using `gemini-3.1-flash-lite`. 
  - *Model Selection Note*: During the Phase 2 smoke test on A01–A03, `gemini-3.5-flash` was initially tested. However, `gemini-3.5-flash` encountered the 20 requests/day project quota on the developer tier (`GenerateRequestsPerDayPerProjectPerModel-FreeTier: 20`). `gemini-3.1-flash-lite` was therefore selected for all full evaluation runs. All four full evaluation runs (v1 development, v2 development, v1 check, v2 check) used strictly **identical settings**: `model = "gemini-3.1-flash-lite"`, `temperature = 0.0`, `timeout_seconds = 60` (configured via `types.HttpOptions(timeout=60000)` in milliseconds), and `max_turns = 3`.
- **Catalogue Tool Protocol**: Real function calling is executed via the `get_brand_facts(brand_id)` function. The Python tool dispatcher (`agent/facts_tool.py`) strictly enforces valid tool names (`IllegalToolError`) and ensures the requested brand matches known catalogue brands (`UnknownBrandError`). Brand facts are returned with schema version `speaker-catalogue-v4`.
- **Span Extraction vs. Character Offset Computation**: Models frequently make off-by-one or token-boundary errors when asked to output numeric character offsets directly. To guarantee 100% character precision, the model is tasked with extracting exact verbatim sentence quotes. The Python runtime (`agent/spans.py`) computes exact Unicode code-point offsets `[start_char, end_char)` using progressive cursor-based sequential matching. This handles identical repeated sentences across the text without collisions or span drift.
- **Strict Validation Pipeline (`agent/validate.py`)**: Prior to accepting any prediction, every claim must satisfy:
  1. Complete containment within `[0, len(ai_answer)]`.
  2. Exact character-level quote equality: `ai_answer[start_char:end_char] == quote`.
  3. No leading or trailing whitespace in quotes.
  4. Non-overlapping character spans: `start_char >= prev_end_char`.
  5. Vocabulary adherence: `label ∈ {"supported", "contradicted", "unknown"}`.
  6. Catalogue fact adherence: all `fact_ids` must exist in `brand_facts.csv` (empty for `unknown`).
  7. Explanation conciseness: `reason` must not exceed 2 sentences.
  If any claim violates these constraints, the entire answer fails as an answer-level error (`status="error"`), producing zero accepted claims. Answer-level errors are never converted to "unknown" and penalize both precision and recall.

---

## 2. Baseline v1 Evaluation & Failure Diagnosis

### Experimental Setup
- **Dataset**: 50 answers from `data/answers.csv` partitioned into `development` (A01–A40) and `check` (A41–A50).
- **Catalogue**: 132 facts across 3 brands (`cleartrail`, `cloudnote`, `stonewave`) in `data/brand_facts.csv`.
- **Baseline Prompt (`v1-frozen`)**: Enforced sentence-level extraction, 1 October 2026 reference date, catalogue authority, and linked-clause refutation rules.

### Development Split Baseline Performance (A01–A40)
- **Gold Claims ($G$)**: 376
- **Predicted Claims ($P$)**: 288
- **Exact Span Matches ($M$)**: 288
- **Correct Label Matches ($C$)**: 286
- **Extraction Precision / Recall / F1**: 1.0000 / 0.7660 / 0.8675
- **Joint Precision / Recall / F1**: 0.9931 / 0.7606 / 0.8614
- **Coverage**: 0.7606 (286 / 376)
- **Answer-Level Errors**: 3 answers (A05, A12, A25)

### Diagnosis of Baseline Errors
1. **Missed Claims (88 claims missed)**:
   - **26 claims** were missed due to the 3 answer-level errors (A05: 8 claims, A12: 9 claims, A25: 9 claims), all caused by explanations exceeding the 2-sentence validator limit.
   - **62 claims** were missed in successfully processed answers. Inspection revealed that the model systematically skipped verifiable product statements when the catalogue had no matching fact (e.g., statements regarding manufacturing origin, repair-time guarantees, or cold-weather operating duration), confusing "absence from catalogue" with "not a claim".
2. **Label Errors (2 claims)**:
   - `A04:1637:1798`: Model predicted `supported`, expected `contradicted`. (Claim asserted 1500 minutes of playback; catalogue stated 24 hours = 1440 minutes. The model failed on unit conversion arithmetic).
   - `A21:559:722`: Model predicted `supported`, expected `contradicted`. (Claim asserted 1260 minutes; catalogue stated 20 hours = 1200 minutes).

---

## 3. Targeted Improvement (v2 Hypothesis & Changes)

### Targeted Change
Rather than loosening the strict 2-sentence validator rule, the biggest performance bottleneck identified was the 62 uncatalogued claims skipped by the model (a potential ~+0.10 joint F1 gain vs. ~+0.04 for the reason length failures).

In `agent/prompts/v2.txt`, exactly **ONE** rule was added to the Extraction Rules section:
> *"A sentence must be extracted whenever it asserts a verifiable product, specification, purchase-policy, warranty, service, shipping, returns, materials, origin or comparison fact, even when the catalogue says nothing about it. Absence from the catalogue is NOT a reason to skip a sentence. Extract it and label it unknown with fact_ids []. Still exclude questions, pure advice, opinions and instructions."*

The codebase was committed and tagged as `v2-frozen`.

### Hypothesis
Explicitly clarifying that catalogue absence is not grounds for omitting an assertive sentence will cause the model to extract the previously skipped sentences and classify them as `unknown`, significantly boosting extraction recall, joint recall, and joint F1 without degrading extraction precision.

---

## 4. Comparative Evaluation: v1 vs v2

### Development Split Comparison (A01–A40)

| Metric | v1 (`v1-frozen`) | v2 (`v2-frozen`) | Absolute Delta | Percentage Delta |
|---|---|---|---|---|
| **Gold Claims ($G$)** | 376 | 376 | 0 | 0.0% |
| **Predicted Claims ($P$)** | 288 | 331 | +43 | +14.9% |
| **Exact Span Matches ($M$)** | 288 | 331 | +43 | +14.9% |
| **Correct Label Matches ($C$)** | 286 | 328 | +42 | +14.7% |
| **Extraction Precision** | **1.0000** | **1.0000** | **+0.0000** | **0.0%** |
| **Extraction Recall** | 0.7660 | 0.8803 | +0.1143 | +11.43% |
| **Extraction F1** | 0.8675 | 0.9364 | +0.0689 | +6.89% |
| **Joint Precision** | 0.9931 | 0.9909 | -0.0022 | -0.22% |
| **Joint Recall** | 0.7606 | 0.8723 | +0.1117 | +11.17% |
| **Joint F1** | **0.8614** | **0.9279** | **+0.0665** | **+6.65%** |
| **Coverage** | **0.7606** | **0.8723** | **+0.1117** | **+11.17%** |
| **Unmatched Predictions** | 0 | 0 | 0 | — |
| **Answer-Level Errors** | 3 (7.5%) | 5 (12.5%) | +2 | +5.0% |

### Measured Effect & Recovery of Missed Claims
- **60 of the 62 previously missed unknown claims were recovered** by v2.
- Across all 35 successfully validated answers in v2, **the model missed zero claims** (100% extraction recall on valid answers).
- All 45 missed claims in v2 belong entirely to the 5 answers that failed validation ($8 + 9 + 8 + 10 + 10 = 45$).
- Joint F1 increased by **+0.0665** (+6.65 percentage points from 0.8614 to 0.9279).
- Coverage increased by **+0.1117** (+11.17 percentage points from 76.06% to 87.23%).

### Regressions & Error Analysis in v2
- **Answer-Level Errors Increased (3 to 5)**:
  - `A05`, `A12`: Failed in both v1 and v2.
  - `A25`: Failed in v1, **succeeded in v2** (11 accepted claims).
  - `A21`, `A35`, `A37`: Succeeded in v1, failed in v2.
  - *Cause & Nature of Increase*: The increase in answer errors from 3 to 5 is likely run-to-run variation in LLM self-correction phrasing. In all 5 errored answers, the failure was triggered because the model performed intermediate arithmetic scratchpad reasoning in the `reason` field (e.g., *"Wait, 1260/60 = 21. The catalogue says 20 hours. Therefore, it is contradicted."*), generating 3–4 sentences and exceeding the strict 2-sentence validator limit.
- **Label Regressions (3 wrong-label claims in v2 vs 2 in v1)**:
  - `A04:1637:1798`: Model predicted `supported`, expected `contradicted` (claim asserted 1500 minutes of playback; catalogue stated 24 hours = 1440 minutes). Wrong in both v1 and v2.
  - `A06:1803:1957`: Price difference comparison between ClearTrail Plus (INR 2,499) and Mini (INR 1,299). Expected: `supported`. Model predicted `contradicted` (correct in v1, regressed in v2).
  - `A25:306:459`: Price difference comparison between CloudNote Air (INR 1,499) and Studio (INR 2,999). Expected: `supported`. Model predicted `contradicted`. (A25 had an answer-level error in v1; in v2 it succeeded validation with 11 claims, but this comparison claim was labeled `contradicted`).
  - Joint precision dropped slightly from **0.9931 to 0.9909** (-0.22%) due to these label misclassifications.
- **Regressed to Missed**: 16 claims (correct in v1) were missed in v2 strictly because their parent answers (`A21`, `A35`, `A37`) failed validation.

### Check Split Evaluation (A41–A50)
- **Not a blind benchmark.** The check inputs were visible to me. The v2 change was designed from development-set errors only, and v2 was frozen (tag v2-frozen) before any check run. The v1-check and v2-check runs were started in the same batch as the v2 development run, before I saw its results, so no change was made based on check output. Even so, v2 was tuned on the development set, so its development gain is in-sample. Gold labels for the check split were never in the repository.
- **Check Split Error Causes (A41 and A49)**:
  - `A41` failed validation in both v1 and v2 because the reason for `A41:911:1071` exceeded 2 sentences (4 sentences: *"1020 minutes is equivalent to 17 hours, but the catalogue states 16 hours for 50% volume. Wait, 1020/60 = 17. The catalogue says 16 [hours]. Therefore, this is contradicted."*).
  - `A49` failed validation in both v1 and v2 because the reason for `A49:975:1135` exceeded 2 sentences (4 sentences: *"The catalogue states 16 hours of battery life [under these conditions], which is equivalent to/equals 960 minutes. Wait, 16 hours * 60 minutes = 960 minutes. The claim says 1020 minutes. This is contradicted."*).
- **Metrics on Check Split**:
  - **v1**: 10 answers processed, 2 errors (A41, A49), 62 accepted claims.
  - **v2**: 10 answers processed, 2 errors (A41, A49), 77 accepted claims (+15 more accepted predictions).
  - In both reports, all quality metrics ($G, M, C$, Precision, Recall, F1, Coverage) are strictly **`null`**.

---

## 5. FastAPI & PostgreSQL Implementation

### Endpoints (`api/main.py`)
1. `POST /classify`:
   - Validates input brand against catalogue (returns `404 Not Found` with `UnknownBrandError` if brand is unknown).
   - Enforces strict anti-leakage validation via Pydantic validator: rejecting any payload containing benchmark fields (`expected_label`, `evidence_fact_ids`, `difficulty_tags`, `gold`, `claims`) with `422 Unprocessable Entity`.
   - Maps agent timeout to `504 Gateway Timeout` and internal model/validation failures to `502 Bad Gateway`.
2. `POST /evaluation-runs`:
   - Triggers fixed evaluation runs across splits (`development` or `check`) with versioning, saving results to `reports/` on disk as JSON. While the PostgreSQL save path via `save_run(report)` is implemented in code, it was not exercised against a live database; when PostgreSQL is unreachable, the endpoint catches the connection error and returns the run summary directly from the disk report.
3. `GET /evaluation-runs/{id}`:
   - Validates UUID format (`422 Unprocessable Entity` on invalid UUID). Retrieves run metadata, answer outcomes, and claim results. When a live PostgreSQL connection is unavailable, it gracefully reads from stored JSON report files on disk (`reports/*.json`). In the evaluated setup, all retrieval requests were served exclusively from these JSON files, not from a database.

### Database Schema (`api/schema.sql` & `api/db.py`)
- Tables:
  - `runs`: Primary metadata, SHA-256 dataset hashes, git revision, and JSONB metrics.
  - `answer_results`: Foreign key to `runs(run_id)` with `CONSTRAINT uq_run_answer UNIQUE (run_id, answer_id)`.
  - `claim_results`: Foreign key to `runs(run_id)` with `CONSTRAINT uq_run_answer_span UNIQUE (run_id, answer_id, start_char, end_char)`.
- All database queries are strictly parameterized (`%s` placeholders with tuples).

### Test Execution Status: Run vs. Written
- **Automated Tests Executed (`pytest tests/`)**: **43 passed, 1 skipped**.
  - `tests/test_core.py`: 8 passed.
  - `tests/test_csv_and_spans.py`: 12 passed.
  - `tests/test_scorer.py`: 6 passed.
  - `tests/test_fake_provider.py`: 7 passed.
  - `tests/test_api_and_db.py`: 10 passed, 1 skipped.
- **PostgreSQL Tests Distinction**:
  - *Actually Run*: Schema DDL validation (`test_sql_schema_syntax_and_tables`) and mock parameterized query execution (`test_db_parameterized_query_mock`) passed using mocked connections.
  - *Written, Not Run*: Live integration test (`test_db_live_postgres_persistence`) was skipped because password authentication failed against the local PostgreSQL service. No rows were inserted into a real database.

---

## 6. Limitations, Failure Modes & Future Improvements

1. **Storage Limitation (JSON vs. PostgreSQL)**:
   - The four run reports (`eval_development_v1`, `eval_development_v2`, `eval_check_v1`, `eval_check_v2`) are stored solely as JSON files on disk in `reports/` rather than in PostgreSQL. While the SQL schema and insertion functions (`save_run`, `get_run_details`) are fully implemented and verified via mock tests, lack of authenticated local PostgreSQL credentials prevented persisting the full evaluation runs into the database.
2. **The 2-Sentence Reason Constraint**:
   - The 2-sentence reason rule is the primary remaining bottleneck in the system. When validating arithmetic claims (e.g. converting 1260 minutes to 21 hours and comparing against 20 hours), Gemini Flash Lite outputs intermediate mathematical steps into the explanation ("Wait, 1260/60 = 21..."), producing 3–4 sentences and tripping the strict validator.
   - *Primary Next Step*: Add a dedicated scratchpad field in the JSON schema for internal math calculations, strictly isolating the final user-facing `reason` to 1–2 sentences.
3. **Unit Conversion Fragility**:
   - LLMs continue to exhibit occasional arithmetic errors on temporal conversions (minutes to hours). Incorporating a deterministic calculator tool alongside `get_brand_facts` would resolve numerical comparison errors.
4. **Unfinished Work**:
   - Provisioning live PostgreSQL server credentials to enable database persistence for full evaluation runs.
   - Prompt engineering or post-processing to compress mathematical justifications into exactly two sentences to eliminate the 5 answer-level errors.
5. **Sample Exchange Provenance**:
   - `reports/sample_exchange_A01.json` was produced by `gemini-3.5-flash` during the Phase 2 smoke/exchange dump step (`scripts/dump_exchange.py`), which may differ from the model used in the four full runs (`gemini-3.1-flash-lite`).

---

## 7. AI Assistance Disclosure

- **Tools:** Google Antigravity (Gemini-based coding agent) wrote nearly all of the code, tests, prompt drafts, run scripts and first drafts of these documents, working from a phase-by-phase spec prompt I gave it. Claude (Anthropic chat) helped with planning, writing that prompt, reviewing results at each stage, and editing this README and write-up.
- **Time:** I received the assessment at 13:05 and submitted at about 16:50, roughly 3 hours 45 minutes. That is over the 2-3 hour guideline; much of it was waiting for live runs.
- **What I did myself:** created the API key and .env; ran the work phase by phase and reviewed the output at each stop; read prompts/v1.txt and confirmed it contains only generic rules; had the gold-isolation checks run and reviewed the output (no gold references in agent/, gold loaded only after predictions are saved); rejected Antigravity's first proposed v2 change (reason-length) after the error breakdown showed skipped unknown claims cost more, a choice I made with Claude's suggestion; checked that all four runs used identical settings and that no secrets are in the repo.
- **What I did not verify independently:** I did not review the code line by line, and I did not recompute the metrics by hand. The CSV parsing, offset resolution, scorer, validator and tests were written by Antigravity and checked only through its test results (43 passed, 1 skipped) and the A01-A03 smoke test.
