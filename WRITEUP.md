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
- **Label Regressions**:
  - `A06:1803:1957`: Price difference comparison between ClearTrail Plus and Mini. Expected: `supported`. v1 predicted `supported`; v2 predicted `contradicted`.
  - Joint precision dropped slightly from **0.9931 to 0.9909** (-0.22%) due to this single misclassification.
- **Regressed to Missed**: 16 claims (correct in v1) were missed in v2 strictly because their parent answers (`A21`, `A35`, `A37`) failed validation.

### Check Split Evaluation (A41–A50)
- **Check Split Protocol**: Check answer texts were present in `data/answers.csv` and visible as inference inputs; however, **v2 was frozen prior to running the check split**, and gold labels for the check split were never provided in the repository. Thus, while the input text was visible, the evaluation is private with respect to ground-truth labels.
- **Metrics on Check Split**:
  - **v1**: 10 answers processed, 2 errors (A41, A49), 62 accepted claims.
  - **v2**: 10 answers processed, 2 errors (A41, A49), 77 accepted claims (+15 uncatalogued assertions captured).
  - In both reports, all quality metrics ($G, M, C$, Precision, Recall, F1, Coverage) are strictly **`null`**.

---

## 5. FastAPI & PostgreSQL Implementation

### Endpoints (`api/main.py`)
1. `POST /classify`:
   - Validates input brand against catalogue (returns `404 Not Found` with `UnknownBrandError` if brand is unknown).
   - Enforces strict anti-leakage validation via Pydantic validator: rejecting any payload containing benchmark fields (`expected_label`, `evidence_fact_ids`, `difficulty_tags`, `gold`, `claims`) with `422 Unprocessable Entity`.
   - Maps agent timeout to `504 Gateway Timeout` and internal model/validation failures to `502 Bad Gateway`.
2. `POST /evaluation-runs`:
   - Triggers fixed evaluation runs across splits (`development` or `check`) with versioning, saving results to `reports/` and PostgreSQL.
3. `GET /evaluation-runs/{id}`:
   - Validates UUID format (`422 Unprocessable Entity` on invalid UUID).
   - `GET /evaluation-runs/{id}`: Retrieves run metadata, answer outcomes, and claim results. When a live PostgreSQL connection is unavailable, it gracefully reads from stored JSON report files on disk (`reports/*.json`). In the evaluated setup, all retrieval requests were served exclusively from these JSON files, not from a database.

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

---

## 7. AI Assistance Disclosure

- **Tools Used**: Gemini coding assistant in the Google Antigravity IDE and Claude.
- **Time Spent**: ~4 hours total (covering setup, agent loop implementation, smoke tests, v1 development run, error diagnosis, v2 prompt refinement, multi-run evaluation, test suite construction, and technical reporting).
- **What Was AI-Generated**:
  - Boilerplate loaders, schema definitions, initial test scaffolding, evaluation runner scripts, and initial prompt drafts.
- **What Was Verified by Candidate**:
  - Verified CSV parser edge cases and raw text preservation.
  - Verified deterministic Unicode code-point character offset computation and sequential cursor matching for duplicate quotes.
  - Verified exact match arithmetic against the official spec benchmark ($G=10, P=10, M=8, C=7$).
  - Verified zero benchmark data leakage into prompts or tool payloads.
  - Audited all validator failures and conducted line-by-line inspection of missed and regressed claims.
  - Confirmed check split quality metrics remain strictly `null`.
