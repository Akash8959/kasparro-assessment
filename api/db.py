"""
Database access module using psycopg (PostgreSQL 3.x driver).
Provides parameterized SQL execution for runs, answer_results, and claim_results.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/kasparro_eval")


def get_db_connection(db_url: Optional[str] = None) -> psycopg.Connection:
    """Open a psycopg database connection."""
    url = db_url or os.environ.get("DATABASE_URL") or DEFAULT_DB_URL
    return psycopg.connect(url, row_factory=dict_row)


def init_db(db_url: Optional[str] = None, schema_path: str | Path = "api/schema.sql") -> None:
    """Initialize the database schema from api/schema.sql."""
    schema_file = Path(schema_path)
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found at {schema_path}")

    with open(schema_file, "r", encoding="utf-8") as f:
        sql = f.read()

    with get_db_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def save_run(
    report: Dict[str, Any],
    db_url: Optional[str] = None
) -> str:
    """
    Persist an evaluation run report and its answers/claims to PostgreSQL.
    Enforces foreign keys, parameterized queries, and unique constraints.
    """
    run_id = report["run_id"]

    with get_db_connection(db_url) as conn:
        with conn.cursor() as cur:
            # 1. Insert into runs
            cur.execute(
                """
                INSERT INTO runs (
                    run_id, split, agent_version, prompt_version,
                    dataset_version, evaluator_version, fact_version,
                    provider, model, settings, timestamp,
                    git_revision, file_hashes, metrics, status
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (run_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    metrics = EXCLUDED.metrics;
                """,
                (
                    run_id,
                    report["split"],
                    report["agent_version"],
                    report["prompt_version"],
                    report["dataset_version"],
                    report["evaluator_version"],
                    report["fact_version"],
                    report["provider"],
                    report["model"],
                    json.dumps(report["settings"]),
                    report["timestamp"],
                    report.get("git_revision"),
                    json.dumps(report["file_hashes"]),
                    json.dumps(report.get("metrics")),
                    "completed",
                ),
            )

            # 2. Insert into answer_results
            for outcome in report.get("answer_outcomes", []):
                ans_id = outcome["answer_id"]
                error_obj = None
                for err in report.get("error_records", []):
                    if err["answer_id"] == ans_id:
                        error_obj = err.get("error")
                        break

                cur.execute(
                    """
                    INSERT INTO answer_results (
                        run_id, answer_id, status, latency_ms, error
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, answer_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        latency_ms = EXCLUDED.latency_ms,
                        error = EXCLUDED.error;
                    """,
                    (
                        run_id,
                        ans_id,
                        outcome["status"],
                        outcome["latency_ms"],
                        json.dumps(error_obj) if error_obj else None,
                    ),
                )

            # 3. Insert into claim_results
            for pred in report.get("predictions", []):
                cur.execute(
                    """
                    INSERT INTO claim_results (
                        run_id, answer_id, claim_id, start_char, end_char,
                        quote, label, fact_ids, reason
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, answer_id, start_char, end_char) DO UPDATE SET
                        label = EXCLUDED.label,
                        fact_ids = EXCLUDED.fact_ids,
                        reason = EXCLUDED.reason;
                    """,
                    (
                        run_id,
                        pred["answer_id"],
                        pred["claim_id"],
                        pred["start_char"],
                        pred["end_char"],
                        pred["quote"],
                        pred["label"],
                        json.dumps(pred["fact_ids"]),
                        pred["reason"],
                    ),
                )

        conn.commit()

    return run_id


def get_run_details(run_id: str, db_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve joined run metadata, answer outcomes, and predicted claims.
    Returns stable ordering: answers sorted by answer_id, claims sorted by start_char.
    """
    with get_db_connection(db_url) as conn:
        with conn.cursor() as cur:
            # Fetch run metadata
            cur.execute(
                """
                SELECT run_id, split, agent_version, prompt_version,
                       dataset_version, evaluator_version, fact_version,
                       provider, model, settings, timestamp,
                       git_revision, file_hashes, metrics, status
                FROM runs
                WHERE run_id = %s;
                """,
                (run_id,),
            )
            run_row = cur.fetchone()
            if not run_row:
                return None

            # Fetch answer results
            cur.execute(
                """
                SELECT answer_id, status, latency_ms, error
                FROM answer_results
                WHERE run_id = %s
                ORDER BY answer_id ASC;
                """,
                (run_id,),
            )
            answers = cur.fetchall()

            # Fetch claim results
            cur.execute(
                """
                SELECT answer_id, claim_id, start_char, end_char,
                       quote, label, fact_ids, reason
                FROM claim_results
                WHERE run_id = %s
                ORDER BY answer_id ASC, start_char ASC;
                """,
                (run_id,),
            )
            claims = cur.fetchall()

    return {
        "run": run_row,
        "answers": answers,
        "claims": claims,
    }
