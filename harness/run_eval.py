"""
Evaluation runner module.
Executes evaluation runs sequentially over development (A01-A40) or check (A41-A50) splits.
Features:
  - Checkpoint saving and resumption to prevent re-running completed answers
  - Short inter-answer delay to preserve rate limits
  - Automatic scoring via harness/scorer.py
  - Report serialization via harness/report.py
"""

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.agent import ClaimAgent, DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS
from agent.loader import load_answers
from harness.gold_loader import load_development_gold
from harness.report import generate_run_report
from harness.scorer import score_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("reports")


def get_checkpoint_path(split: str, version: str) -> Path:
    return CHECKPOINT_DIR / f"checkpoint_{split}_{version}.json"


def load_checkpoint(split: str, version: str) -> Dict[str, Any]:
    path = get_checkpoint_path(split, version)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read checkpoint at {path}: {e}")
    return {"run_id": str(uuid.uuid4()), "results": {}}


def save_checkpoint(split: str, version: str, data: Dict[str, Any]) -> None:
    path = get_checkpoint_path(split, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp_path.replace(path)


def clear_checkpoint(split: str, version: str) -> None:
    path = get_checkpoint_path(split, version)
    if path.exists():
        try:
            path.unlink()
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {path}: {e}")


def run_evaluation(
    split: str = "development",
    version: str = "v1",
    model_name: str = DEFAULT_MODEL,
    limit: Optional[int] = None,
    delay_between_answers: float = 1.0,
    resume: bool = True,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run evaluation for specified split and agent version.
    """
    answers = load_answers(split=split)
    if limit:
        answers = answers[:limit]

    logger.info(f"Loaded {len(answers)} answers for split '{split}' (version '{version}', model '{model_name}').")

    # Load or initialize checkpoint
    if resume:
        checkpoint = load_checkpoint(split, version)
        active_run_id = checkpoint.get("run_id") or run_id or str(uuid.uuid4())
        results_by_id = checkpoint.get("results", {})
        logger.info(f"Resuming run {active_run_id}. Already completed: {len(results_by_id)} answers.")
    else:
        active_run_id = run_id or str(uuid.uuid4())
        results_by_id = {}

    agent = ClaimAgent(
        prompt_version=version,
        model_name=model_name,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )

    for idx, ans in enumerate(answers):
        ans_id = ans["answer_id"]
        if ans_id in results_by_id:
            logger.info(f"[{idx+1}/{len(answers)}] Skipping {ans_id} (already in checkpoint).")
            continue

        logger.info(f"[{idx+1}/{len(answers)}] Processing {ans_id} ({ans['brand_id']})...")
        res = agent.process_answer(
            answer_id=ans_id,
            brand_id=ans["brand_id"],
            question=ans["question"],
            ai_answer=ans["ai_answer"],
        )

        results_by_id[ans_id] = res

        # Save checkpoint after each answer
        save_checkpoint(split, version, {"run_id": active_run_id, "results": results_by_id})

        if delay_between_answers > 0 and idx < len(answers) - 1:
            time.sleep(delay_between_answers)

    # Order results to match input answers
    ordered_results = [results_by_id[a["answer_id"]] for a in answers if a["answer_id"] in results_by_id]

    # Score results
    is_check = (split == "check")
    gold_claims = []
    if not is_check:
        gold_claims = load_development_gold()
        # Filter gold claims to only scheduled answers
        scheduled_ids = {a["answer_id"] for a in answers}
        gold_claims = [g for g in gold_claims if g["answer_id"] in scheduled_ids]

    score_summary = score_predictions(
        gold_claims=gold_claims,
        answer_results=ordered_results,
        is_check_split=is_check,
    )

    # Generate full report
    report = generate_run_report(
        run_id=active_run_id,
        split=split,
        agent_version=version,
        model_name=model_name,
        answer_results=ordered_results,
        score_summary=score_summary,
    )

    # If run completed fully, clear checkpoint
    if len(ordered_results) == len(answers):
        clear_checkpoint(split, version)

    logger.info(f"Run {active_run_id} completed successfully.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Run Kasparro FSE evaluation harness.")
    parser.add_argument("--version", choices=["v1", "v2"], default="v1", help="Agent prompt version")
    parser.add_argument("--split", choices=["development", "check"], default="development", help="Evaluation split")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on answers processed")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between answers in seconds")

    args = parser.parse_args()
    report = run_evaluation(
        split=args.split,
        version=args.version,
        model_name=args.model,
        limit=args.limit,
        delay_between_answers=args.delay,
        resume=not args.no_resume,
    )

    # Print summary
    metrics = report.get("metrics", {})
    print("\n" + "=" * 50)
    print(f"EVALUATION SUMMARY: Split={args.split}, Version={args.version}")
    print(f"Total Answers: {report['total_answers']}, Successful: {report['successful_answers']}, Errors: {report['error_answers']}")
    print(f"Total Accepted Predictions: {report['total_accepted_predictions']}")
    if not metrics.get("is_check_split"):
        print(f"Gold Claims (G): {metrics.get('G')}")
        print(f"Accepted Predictions (P): {metrics.get('P')}")
        print(f"Exact Matches (M): {metrics.get('M')}")
        print(f"Correct Matches (C): {metrics.get('C')}")
        print(f"Extraction Precision: {metrics.get('extraction_precision')}")
        print(f"Extraction Recall:    {metrics.get('extraction_recall')}")
        print(f"Extraction F1:        {metrics.get('extraction_f1')}")
        print(f"Joint Precision:      {metrics.get('joint_precision')}")
        print(f"Joint Recall:         {metrics.get('joint_recall')}")
        print(f"Joint F1:             {metrics.get('joint_f1')}")
        print(f"Coverage (C/G):       {metrics.get('coverage')}")
    else:
        print("Check split: All quality metrics are null (private evaluation).")
    print("=" * 50)


if __name__ == "__main__":
    main()
