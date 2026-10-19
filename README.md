# Kasparro FSE Intern Pathway Assessment

Repo link: <PASTE GITHUB URL HERE>

An agentic factual claim extraction and verification pipeline that processes long AI-generated answers about speaker products, extracts factual claim sentences, fetches catalogue facts via live function calling, and classifies each claim as `supported`, `contradicted`, or `unknown`.

---

## 1. Setup & Installation

### Prerequisites
- Python 3.11+
- Git
- PostgreSQL (optional for local DB persistence, required for full DB test suite)

### Clone & Virtual Environment Setup
```bash
git clone <repo-url>
cd kasparro-assessment

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env` and fill in your Gemini API key and PostgreSQL connection string:
```bash
cp .env.example .env
```
In `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/kasparro_eval
```
> **Security Note:** Never commit `.env` or log API keys. `.env` is ignored in `.gitignore`.

---

## 2. Configuration & Model Settings

- **Provider**: Google Gemini via `google-genai` SDK
- **Model**: `gemini-3.1-flash-lite`
- **Temperature**: `0.0`
- **Provider Request Timeout**: `60.0` seconds (configured via `types.HttpOptions(timeout=60000)` in milliseconds)
- **Max Model Turns**: 3 turns per answer
- **Tool Protocol**: Function calling with `get_brand_facts(brand_id)`
- **Fact Version**: `speaker-catalogue-v4`
- **Dataset Version**: `long-brand-answers-50-v4`
- **Evaluator Version**: `sentence-spans-and-labels-v4`

---

## 3. Running the Evaluation Harness

The evaluation harness processes answers sequentially, saves checkpoints to `reports/` after every answer, and produces a complete JSON report.

### Run Baseline v1 on Development Split (A01–A40)
```bash
python -m harness.run_eval --version v1 --split development
```

### Run Candidate v2 on Development Split (A01–A40)
```bash
python -m harness.run_eval --version v2 --split development
```

### Run Check Split (A41–A50)
```bash
python -m harness.run_eval --version v1 --split check
python -m harness.run_eval --version v2 --split check
```
*(Check split outputs predictions and error records only; quality metrics remain null).*

### Run All Evaluations Sequentially
```bash
python scripts/run_all_evals.py
```

---

## 4. Running the FastAPI Service

Start the FastAPI server using Uvicorn:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints
- `POST /classify`: Classify a single answer. Input: `answer_id`, `brand_id`, `question`, `ai_answer`, `agent_version`. Returns 422 if any gold/benchmark fields are supplied.
- `POST /evaluation-runs`: Trigger an evaluation run across a CSV split (`development` or `check`).
- `GET /evaluation-runs/{id}`: Retrieve evaluation run details. Reads from stored JSON reports in `reports/` (or PostgreSQL if connected).

### Database Persistence & Script
Evaluation runs are saved as JSON files in `reports/`. To optionally load them into a PostgreSQL database once credentials are configured:
```bash
python scripts/load_reports_to_db.py
```

---

## 5. Running Tests

Run the complete test suite via pytest:
```bash
pytest tests/ -v
```
Output is saved to `reports/test_output.txt`. All tests except the live PostgreSQL test use deterministic mocks and fake providers to verify edge cases without incurring API usage.
