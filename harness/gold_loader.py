"""
Harness reference data loader.
Loads development_gold.csv exclusively for post-inference evaluation in the harness.
Never imported by the agent package.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_development_gold(
    filepath: str | Path = "data/development_gold.csv",
    answer_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Load gold reference claims from development_gold.csv.
    Used ONLY in the evaluation harness after predictions are saved.
    """
    gold_records = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if answer_id is None or row["answer_id"] == answer_id:
                gold_records.append({
                    "answer_id": row["answer_id"],
                    "claim_id": row["claim_id"],
                    "start_char": int(row["start_char"]),
                    "end_char": int(row["end_char"]),
                    "quote": row["quote"],
                    "expected_label": row["expected_label"],
                    "evidence_fact_ids": [fid for fid in row["evidence_fact_ids"].split("|") if fid],
                    "explanation": row["explanation"],
                    "difficulty_tags": [t for t in row["difficulty_tags"].split("|") if t],
                })
    return gold_records
