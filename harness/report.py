"""
Evaluation reporting module.
Assembles run metadata, answer outcomes, accepted predictions, error records, and scorer metrics.
Saves JSON reports in reports/ following assessment specifications.
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.loader import get_dataset_hashes, DATASET_VERSION, FACT_VERSION, EVALUATOR_VERSION


def get_git_revision() -> Optional[str]:
    """Retrieve current git revision hash if available."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        return rev
    except Exception:
        return None


def generate_run_report(
    run_id: str,
    split: str,
    agent_version: str,
    model_name: str,
    answer_results: List[Dict[str, Any]],
    score_summary: Dict[str, Any],
    reports_dir: str | Path = "reports",
) -> Dict[str, Any]:
    """
    Generate and save evaluation report.
    For check split: G, M, C, matched IDs, expected labels, correctness, and all quality metrics are null.
    """
    reports_path = Path(reports_dir)
    reports_path.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    git_rev = get_git_revision()
    file_hashes = get_dataset_hashes()

    # Predictions and errors
    predictions = []
    error_records = []
    answer_outcomes = []

    for a in answer_results:
        ans_id = a["answer_id"]
        status = a.get("status", "unknown")
        latency = a.get("latency_ms", 0)
        err = a.get("error")

        answer_outcomes.append({
            "answer_id": ans_id,
            "status": status,
            "latency_ms": latency,
            "claim_count": len(a.get("claims", [])),
            "has_error": status == "error",
            "error_type": err.get("error_type") if err else None,
        })

        if status == "success":
            for c in a.get("claims", []):
                predictions.append({
                    "answer_id": ans_id,
                    "claim_id": c["claim_id"],
                    "start_char": c["start_char"],
                    "end_char": c["end_char"],
                    "quote": c["quote"],
                    "label": c["label"],
                    "fact_ids": c["fact_ids"],
                    "reason": c["reason"],
                })
        else:
            error_records.append({
                "answer_id": ans_id,
                "error": err,
                "latency_ms": latency,
            })

    report = {
        "run_id": run_id,
        "split": split,
        "agent_version": agent_version,
        "prompt_version": agent_version,
        "dataset_version": DATASET_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "fact_version": FACT_VERSION,
        "provider": "google-genai",
        "model": model_name,
        "settings": {
            "temperature": 0.0,
            "timeout_seconds": 60,
            "max_turns": 3,
        },
        "timestamp": timestamp,
        "git_revision": git_rev,
        "file_hashes": file_hashes,
        "answer_outcomes": answer_outcomes,
        "total_answers": len(answer_results),
        "successful_answers": len([a for a in answer_results if a.get("status") == "success"]),
        "error_answers": len(error_records),
        "total_accepted_predictions": len(predictions),
        "error_records": error_records,
        "predictions": predictions,
        "metrics": score_summary,
    }

    report_filename = f"eval_{split}_{agent_version}_{run_id[:8]}.json"
    report_file = reports_path / report_filename
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
