"""
Data loader module for agent runtime.
Reads answers.csv and brand_facts.csv using standard UTF-8 CSV parsing.
Preserves raw strings, handles embedded commas/newlines, and computes SHA-256 hashes.
"""

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set

DATASET_VERSION = "long-brand-answers-50-v4"
FACT_VERSION = "speaker-catalogue-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"


def get_file_hash(filepath: str | Path) -> str:
    """Compute the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_dataset_hashes(data_dir: str | Path = "data") -> Dict[str, str]:
    """Compute SHA-256 hashes for all CSV dataset files in data_dir."""
    data_path = Path(data_dir)
    return {p.name: get_file_hash(p) for p in sorted(data_path.glob("*.csv"))}


def load_answers(
    filepath: str | Path = "data/answers.csv",
    split: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Load answers from CSV.
    Each record contains: answer_id, split, brand_id, question, ai_answer.
    ai_answer is preserved exactly as stored with no whitespace normalization or line splitting.
    """
    answers = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is None or row["split"] == split:
                answers.append({
                    "answer_id": row["answer_id"],
                    "split": row["split"],
                    "brand_id": row["brand_id"],
                    "question": row["question"],
                    "ai_answer": row["ai_answer"],
                })
    return answers


def load_brand_facts(
    filepath: str | Path = "data/brand_facts.csv"
) -> List[Dict[str, str]]:
    """
    Load brand facts from CSV.
    Columns: fact_id, brand_id, model_id, field, value, value_type, unit, condition, valid_from, valid_to.
    Booleans are kept as raw text 'true'/'false'.
    Blank units and conditions are preserved as empty strings.
    """
    facts = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append({
                "fact_id": row["fact_id"],
                "brand_id": row["brand_id"],
                "model_id": row["model_id"],
                "field": row["field"],
                "value": row["value"],
                "value_type": row["value_type"],
                "unit": row["unit"] or "",
                "condition": row["condition"] or "",
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
            })
    return facts


def get_valid_fact_ids(filepath: str | Path = "data/brand_facts.csv") -> Set[str]:
    """Return the set of all valid fact_ids from the catalogue."""
    facts = load_brand_facts(filepath)
    return {f["fact_id"] for f in facts}


def get_valid_brand_ids(filepath: str | Path = "data/brand_facts.csv") -> Set[str]:
    """Return the set of all known brand_ids from the catalogue."""
    facts = load_brand_facts(filepath)
    return {f["brand_id"] for f in facts}
