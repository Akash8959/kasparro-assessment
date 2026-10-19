"""
FastAPI application for the Kasparro FSE Assessment.
Implements thin routes:
  - POST /classify
  - POST /evaluation-runs
  - GET /evaluation-runs/{id}
"""

import uuid
from typing import Any, Dict, List, Literal, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from agent.agent import ClaimAgent, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS
from agent.facts_tool import UnknownBrandError
from agent.loader import get_valid_brand_ids
from harness.run_eval import run_evaluation
from api.db import get_run_details, save_run

app = FastAPI(title="Kasparro FSE Assessment API", version="1.0.0")

# Prohibited fields that must never appear in /classify input
FORBIDDEN_INPUT_FIELDS = {
    "claims", "claim_id", "start_char", "end_char", "quote",
    "expected_label", "evidence_fact_ids", "explanation", "difficulty_tags",
    "gold", "gold_claims", "labels"
}


class ClassifyRequest(BaseModel):
    answer_id: str
    brand_id: str
    question: str
    ai_answer: str
    agent_version: Literal["v1", "v2"] = "v1"

    @model_validator(mode="before")
    @classmethod
    def check_forbidden_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            found_forbidden = [f for f in values if f in FORBIDDEN_INPUT_FIELDS]
            if found_forbidden:
                raise ValueError(
                    f"Forbidden gold/claim-list fields present in inference payload: {found_forbidden}"
                )
        return values


class EvaluationRunRequest(BaseModel):
    agent_version: Literal["v1", "v2"] = "v1"
    split: Literal["development", "check"] = "development"
    limit: Optional[int] = None


@app.post("/classify", status_code=status.HTTP_200_OK)
async def classify_answer(req: ClassifyRequest):
    """
    Run claim extraction and classification for a single answer.
    422: Invalid input or presence of forbidden gold/claim-list fields.
    404: Unknown primary brand.
    502: Provider failure or invalid model output.
    504: Request timeout.
    """
    valid_brands = get_valid_brand_ids("data/brand_facts.csv")
    clean_brand = req.brand_id.strip().lower()
    if clean_brand not in valid_brands:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_type": "UnknownBrandError",
                "message": f"Brand '{req.brand_id}' is not in the catalogue. Known brands: {sorted(valid_brands)}"
            }
        )

    try:
        agent = ClaimAgent(
            prompt_version=req.agent_version,
            model_name=DEFAULT_MODEL,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error_type": "AgentInitializationError", "message": "Failed to initialize agent"}
        )

    result = agent.process_answer(
        answer_id=req.answer_id,
        brand_id=req.brand_id,
        question=req.question,
        ai_answer=req.ai_answer,
    )

    if result["status"] == "error":
        err = result.get("error") or {}
        err_type = err.get("error_type", "AgentError")
        err_msg = err.get("message", "An error occurred during claim processing")

        if "timeout" in err_msg.lower() or "deadline" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"error_type": "GatewayTimeout", "message": "Provider request timed out"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"error_type": err_type, "message": err_msg}
            )

    return {
        "answer_id": req.answer_id,
        "claims": result["claims"],
        "latency_ms": result["latency_ms"],
    }


@app.post("/evaluation-runs", status_code=status.HTTP_201_CREATED)
async def create_evaluation_run(req: EvaluationRunRequest):
    """
    Execute an evaluation run across a fixed CSV split and persist to database if available.
    """
    report = run_evaluation(
        split=req.split,
        version=req.agent_version,
        limit=req.limit,
    )

    # Attempt PostgreSQL save if configured
    try:
        save_run(report)
    except Exception as e:
        # DB may not be configured in non-DB mode; report is still saved to disk
        pass

    return {
        "run_id": report["run_id"],
        "status": "completed",
        "split": req.split,
        "agent_version": req.agent_version,
        "total_answers": report["total_answers"],
        "successful_answers": report["successful_answers"],
        "error_answers": report["error_answers"],
        "total_accepted_predictions": report["total_accepted_predictions"],
        "metrics": report["metrics"],
    }


@app.get("/evaluation-runs/{run_id}")
async def get_evaluation_run(run_id: str):
    """
    Retrieve joined run metadata, answer outcomes, and predicted claims.
    422 if run_id is not a valid UUID.
    404 if run not found.
    """
    try:
        val_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error_type": "InvalidUUID", "message": f"'{run_id}' is not a valid UUID."}
        )

    # First try PostgreSQL
    try:
        data = get_run_details(str(val_uuid))
        if data:
            return data
    except Exception:
        pass

    # Fallback to local report files in reports/
    from pathlib import Path
    import json
    for p in Path("reports").glob(f"eval_*_{run_id[:8]}*.json"):
        with open(p, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            if report_data.get("run_id") == str(val_uuid) or report_data.get("run_id", "").startswith(run_id[:8]):
                return report_data

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error_type": "RunNotFound", "message": f"Run '{run_id}' was not found."}
    )
