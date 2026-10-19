"""
Tests for FastAPI endpoints and PostgreSQL persistence/queries.
Validates HTTP status codes (200, 404, 422, 502, 504), input rejection of forbidden gold fields,
and database schema constraints, parameterized queries, and joined retrieval.
"""

import json
import os
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.db import save_run, get_run_details, init_db, get_db_connection

client = TestClient(app)


# ---------------------------------------------------------
# FastAPI Route Tests
# ---------------------------------------------------------

def test_api_classify_rejects_forbidden_gold_fields():
    """
    CRITICAL SECURITY CHECK:
    Verify that POST /classify returns HTTP 422 if ANY gold/benchmark fields
    are passed in the request body (e.g., expected_label, evidence_fact_ids, gold).
    """
    forbidden_payloads = [
        {
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "What is the return policy?",
            "ai_answer": "Returns within 30 days.",
            "expected_label": "supported",  # FORBIDDEN
        },
        {
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "What is the return policy?",
            "ai_answer": "Returns within 30 days.",
            "evidence_fact_ids": ["ct.return.policy"],  # FORBIDDEN
        },
        {
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "What is the return policy?",
            "ai_answer": "Returns within 30 days.",
            "difficulty_tags": "hard",  # FORBIDDEN
        },
        {
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "What is the return policy?",
            "ai_answer": "Returns within 30 days.",
            "claims": [{"quote": "Returns within 30 days."}],  # FORBIDDEN
        },
    ]

    for p in forbidden_payloads:
        resp = client.post("/classify", json=p)
        assert resp.status_code == 422
        assert "forbidden" in resp.text.lower() or "validation error" in resp.text.lower()


def test_api_classify_unknown_brand_returns_404():
    """Verify unknown primary brand returns HTTP 404 Not Found."""
    resp = client.post("/classify", json={
        "answer_id": "A99",
        "brand_id": "totally_fictional_brand_xyz",
        "question": "What is the price?",
        "ai_answer": "It costs 50 dollars.",
    })
    assert resp.status_code == 404
    body = resp.json()
    assert "UnknownBrandError" in body["detail"]["error_type"]


def test_api_classify_valid_mocked_success():
    """Verify successful classification returns HTTP 200 with claims."""
    with patch("api.main.ClaimAgent") as MockAgentClass:
        mock_instance = MockAgentClass.return_value
        mock_instance.process_answer.return_value = {
            "answer_id": "A01",
            "status": "success",
            "claims": [
                {
                    "claim_id": "A01:0:22",
                    "start_char": 0,
                    "end_char": 22,
                    "quote": "ClearTrail Mini is $80.",
                    "label": "supported",
                    "fact_ids": ["ct.mini.price"],
                    "reason": "Matches catalogue.",
                }
            ],
            "latency_ms": 120,
            "error": None,
        }

        resp = client.post("/classify", json={
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "What is the price of ClearTrail Mini?",
            "ai_answer": "ClearTrail Mini is $80.",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer_id"] == "A01"
        assert len(data["claims"]) == 1
        assert data["claims"][0]["claim_id"] == "A01:0:22"
        assert data["latency_ms"] == 120


def test_api_classify_provider_error_maps_to_502():
    """Verify model/provider failure maps to HTTP 502 Bad Gateway."""
    with patch("api.main.ClaimAgent") as MockAgentClass:
        mock_instance = MockAgentClass.return_value
        mock_instance.process_answer.return_value = {
            "answer_id": "A01",
            "status": "error",
            "claims": [],
            "error": {
                "error_type": "InvalidOutputError",
                "message": "Model returned invalid JSON",
            },
            "latency_ms": 50,
        }

        resp = client.post("/classify", json={
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "What is the price?",
            "ai_answer": "Price is 80.",
        })

        assert resp.status_code == 502
        assert resp.json()["detail"]["error_type"] == "InvalidOutputError"


def test_api_classify_timeout_maps_to_504():
    """Verify provider timeout maps to HTTP 504 Gateway Timeout."""
    with patch("api.main.ClaimAgent") as MockAgentClass:
        mock_instance = MockAgentClass.return_value
        mock_instance.process_answer.return_value = {
            "answer_id": "A01",
            "status": "error",
            "claims": [],
            "error": {
                "error_type": "TimeoutError",
                "message": "Request deadline exceeded: provider timeout after 60s",
            },
            "latency_ms": 60000,
        }

        resp = client.post("/classify", json={
            "answer_id": "A01",
            "brand_id": "cleartrail",
            "question": "What is the price?",
            "ai_answer": "Price is 80.",
        })

        assert resp.status_code == 504
        assert resp.json()["detail"]["error_type"] == "GatewayTimeout"


def test_api_get_run_invalid_uuid_returns_422():
    """Verify non-UUID parameter returns HTTP 422."""
    resp = client.get("/evaluation-runs/not-a-valid-uuid")
    assert resp.status_code == 422


def test_api_get_run_nonexistent_uuid_returns_404():
    """Verify valid but non-existent UUID returns HTTP 404."""
    random_uuid = str(uuid.uuid4())
    resp = client.get(f"/evaluation-runs/{random_uuid}")
    assert resp.status_code == 404


def test_api_get_run_existing_from_disk_fallback():
    """Verify existing run report saved to reports/ can be retrieved via GET /evaluation-runs/{id}."""
    # Find any existing evaluation report in reports/
    report_files = list(Path("reports").glob("eval_*.json"))
    if report_files:
        with open(report_files[0], "r", encoding="utf-8") as f:
            report_data = json.load(f)
        run_id = report_data["run_id"]
        resp = client.get(f"/evaluation-runs/{run_id}")
        assert resp.status_code == 200
        assert resp.json()["run_id"] == run_id


# ---------------------------------------------------------
# PostgreSQL Schema & Parameterized Queries Tests
# ---------------------------------------------------------

def test_sql_schema_syntax_and_tables():
    """Verify api/schema.sql file exists and defines all required tables, constraints, and indexes."""
    schema_path = Path("api/schema.sql")
    assert schema_path.exists()
    content = schema_path.read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS runs" in content
    assert "CREATE TABLE IF NOT EXISTS answer_results" in content
    assert "CREATE TABLE IF NOT EXISTS claim_results" in content
    assert "CONSTRAINT uq_run_answer UNIQUE (run_id, answer_id)" in content
    assert "CONSTRAINT uq_run_answer_span UNIQUE (run_id, answer_id, start_char, end_char)" in content
    assert "CREATE INDEX IF NOT EXISTS idx_answer_results_run_id" in content
    assert "CREATE INDEX IF NOT EXISTS idx_claim_results_run_id" in content


def test_db_parameterized_query_mock():
    """Verify save_run generates strictly parameterized queries and safe serialization."""
    sample_report = {
        "run_id": str(uuid.uuid4()),
        "split": "development",
        "agent_version": "v1",
        "prompt_version": "v1",
        "dataset_version": "sha256-test",
        "evaluator_version": "eval-v1",
        "fact_version": "speaker-catalogue-v4",
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
        "settings": {"temperature": 0.0},
        "timestamp": "2026-10-19T10:00:00Z",
        "git_revision": "abc1234",
        "file_hashes": {"answers.csv": "hash1"},
        "metrics": {"extraction_f1": 0.85},
        "answer_outcomes": [
            {"answer_id": "A01", "status": "success", "latency_ms": 1500}
        ],
        "predictions": [
            {
                "answer_id": "A01",
                "claim_id": "A01:0:20",
                "start_char": 0,
                "end_char": 20,
                "quote": "ClearTrail Mini is 80.",
                "label": "supported",
                "fact_ids": ["ct.mini.price"],
                "reason": "Catalogue fact.",
            }
        ],
        "error_records": [],
    }

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    with patch("api.db.get_db_connection", return_value=mock_conn):
        saved_id = save_run(sample_report)
        assert saved_id == sample_report["run_id"]
        # Verify execute was called with parameterized query tuples
        assert mock_cursor.execute.call_count >= 3
        for call_args in mock_cursor.execute.call_args_list:
            sql, params = call_args[0]
            assert "%s" in sql
            assert isinstance(params, tuple)


def test_db_live_postgres_persistence():
    """
    Test live database execution against a running PostgreSQL instance if reachable.
    If database is not reachable, the test reports clear status and is skipped.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not configured; skipping live PostgreSQL integration test.")

    try:
        init_db(db_url)
    except Exception as e:
        pytest.skip(f"Live PostgreSQL instance not reachable ({e}); skipping live test.")

    # Generate isolated test run
    test_run_id = str(uuid.uuid4())
    sample_report = {
        "run_id": test_run_id,
        "split": "development",
        "agent_version": "v1",
        "prompt_version": "v1",
        "dataset_version": "sha256-test",
        "evaluator_version": "eval-v1",
        "fact_version": "speaker-catalogue-v4",
        "provider": "gemini",
        "model": "gemini-3.1-flash-lite",
        "settings": {"temperature": 0.0},
        "timestamp": "2026-10-19T10:00:00Z",
        "git_revision": "test-git",
        "file_hashes": {"answers.csv": "abc"},
        "metrics": {"extraction_f1": 1.0},
        "answer_outcomes": [
            {"answer_id": "A01", "status": "success", "latency_ms": 100},
            {"answer_id": "A02", "status": "error", "latency_ms": 50},
        ],
        "predictions": [
            {
                "answer_id": "A01",
                "claim_id": "A01:0:15",
                "start_char": 0,
                "end_char": 15,
                "quote": "Sample claim 1.",
                "label": "supported",
                "fact_ids": ["ct.mini.price"],
                "reason": "Test reason.",
            }
        ],
        "error_records": [
            {
                "answer_id": "A02",
                "error": {"error_type": "InvalidOutputError", "message": "Failed"},
            }
        ],
    }

    # Save
    save_run(sample_report, db_url=db_url)

    # Retrieve and verify joined query and stable ordering
    retrieved = get_run_details(test_run_id, db_url=db_url)
    assert retrieved is not None
    assert retrieved["run"]["run_id"] == test_run_id
    assert len(retrieved["answers"]) == 2
    assert retrieved["answers"][0]["answer_id"] == "A01"
    assert retrieved["answers"][1]["answer_id"] == "A02"
    assert len(retrieved["claims"]) == 1
    assert retrieved["claims"][0]["claim_id"] == "A01:0:15"
