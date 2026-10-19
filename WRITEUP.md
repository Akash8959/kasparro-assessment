# Kasparro FSE Intern Pathway Assessment — Technical Writeup

## 1. System Architecture & Extraction/Classification Design
- **Architecture Overview**: Agentic extraction and verification pipeline built with Python 3.11, `google-genai` SDK, and Gemini Flash (`gemini-3.8-flash`).
- **Tool Protocol**: Real function calling with `get_brand_facts(brand_id)`. The model issues structured tool calls, which the Python runtime validates against known brands and executes strictly against `brand_facts.csv` (version `speaker-catalogue-v4`).
- **Span Extraction & Alignment**: The model extracts exact claim quotes; Python deterministically locates zero-based Unicode code-point offsets `[start_char, end_char)` using progressive cursor matching to resolve repeated sentences.
- **Validation Pipeline**: Strict validation ensures non-overlapping spans, valid claim IDs, exact quote equality, valid label vocabulary (`supported`, `contradicted`, `unknown`), existing catalogue fact IDs, and concise explanations.

## 2. Baseline v1 Evaluation & Failure Diagnosis
- **Baseline Prompt (`v1`)**: Initial system prompt enforcing sentence-level extraction, 1 October 2026 reference date, strict catalogue authority, and linked-clause refutation rules.
- **Development Split Performance (A01–A40)**:
  - *G* (Gold claims): 376
  - *P* (Predicted claims): [TBD upon full run]
  - *M* (Exact span matches): [TBD]
  - *C* (Correct label matches): [TBD]
  - *Joint F1*: [TBD]
- **Error Analysis & Failure Classification**:
  - Missed claims:
  - Extra non-claim sentences:
  - Label misclassifications:
  - Span boundary / punctuation issues:

## 3. Targeted Improvement (v2 Hypothesis & Changes)
- **Diagnosed Bottleneck**: [To be identified after v1 development run]
- **Targeted Single Change**: [Single precise modification to prompt / agent logic]
- **Hypothesis**: [Expected impact on extraction or joint F1]

## 4. Comparative Evaluation: v1 vs v2
- **Development Split Comparison (A01–A40)**:
  - Metric breakdown table (G, P, M, C, Extraction P/R/F1, Joint P/R/F1, Coverage).
  - Regression analysis (cases where v1 was correct but v2 erred).
- **Check Split Evaluation (A41–A50)**:
  - Predictions and error accounting for private evaluation.

## 5. FastAPI & PostgreSQL Implementation
- **REST Endpoints**:
  - `POST /classify`: Thin inference route with validation against gold-data leaks, unknown brands, and provider errors.
  - `POST /evaluation-runs`: Automated split evaluation runner.
  - `GET /evaluation-runs/{id}`: Joined relational query for run metadata, answer outcomes, and ordered claims.
- **Data Model**: Relational schema in PostgreSQL with foreign keys and uniqueness constraints on `(run_id, answer_id)` and `(run_id, answer_id, start_char, end_char)`.

## 6. Reliability, Limitations, and Future Work
- **Reliability Features**: 60s request timeout, exponential backoff on transient errors, persistent checkpoints for failure recovery.
- **Limitations & Future Work**:
  - Complex multi-hop numerical reasoning.
  - Granular clause decomposition while maintaining exact sentence boundaries.

## 7. AI Assistance Disclosure
- **Tools Used**: Gemini assistant in Antigravity.
- **What Was Generated**: Boilerplate loaders, schema definitions, test suites, evaluation scripts, and prompt scaffolding.
- **What Was Verified by Candidate**: Verified CSV parser edge cases, deterministic span indexing, exact scorer arithmetic checks, API constraints, and live error analysis.
