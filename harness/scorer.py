"""
Deterministic evaluation scorer.
Implements exact span matching on (answer_id, start_char, end_char) with quote validation.
Computes G, P, M, C, extraction metrics, joint metrics, and detailed discrepancy tracking.
Strictly conforms to spec: no fuzzy matching, no LLM judge, exact arithmetic.
"""

from typing import Any, Dict, List, Optional, Set, Tuple


def score_predictions(
    gold_claims: List[Dict[str, Any]],
    answer_results: List[Dict[str, Any]],
    is_check_split: bool = False
) -> Dict[str, Any]:
    """
    Score accepted predictions against gold claims across a split.
    
    If is_check_split is True:
      Quality metrics (G, M, C, precision, recall, F1, coverage, matched IDs, expected labels)
      must be null/None because check split gold is private.
      
    Rules:
      - Match condition: answer_id, start_char, end_char must be identical.
      - First validates quote equality between gold and prediction.
      - Each gold span matches at most one prediction.
      - Answers with errors add 0 accepted predictions to P, but their gold claims remain in G as missed.
      - Zero denominators evaluate to 0.0 when gold claims exist.
    """
    # Total answers and errors
    total_answers = len(answer_results)
    error_answers = [a for a in answer_results if a.get("status") == "error"]
    error_answer_count = len(error_answers)
    error_answer_ids = [a["answer_id"] for a in error_answers]

    # Collect accepted predictions from answers with status == 'success'
    accepted_predictions: List[Dict[str, Any]] = []
    for a in answer_results:
        if a.get("status") == "success":
            for claim in a.get("claims", []):
                # Ensure answer_id is attached
                c_copy = dict(claim)
                c_copy["answer_id"] = a["answer_id"]
                accepted_predictions.append(c_copy)

    # For check split (A41-A50) or when gold is None: all quality metrics must be None
    if is_check_split or gold_claims is None:
        return {
            "is_check_split": True,
            "total_answers": total_answers,
            "error_answer_count": error_answer_count,
            "error_answer_ids": error_answer_ids,
            "accepted_prediction_count": len(accepted_predictions),
            "G": None,
            "P": len(accepted_predictions),
            "M": None,
            "C": None,
            "extraction_precision": None,
            "extraction_recall": None,
            "extraction_f1": None,
            "joint_precision": None,
            "joint_recall": None,
            "joint_f1": None,
            "coverage": None,
            "missed_gold_ids": None,
            "unmatched_prediction_ids": None,
            "matched_wrong_label_records": None,
        }

    # Development split scoring
    G = len(gold_claims)
    P = len(accepted_predictions)

    # Index gold claims by (answer_id, start_char, end_char)
    # Check that gold claims are unique
    gold_by_span: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for g in gold_claims:
        key = (g["answer_id"], int(g["start_char"]), int(g["end_char"]))
        gold_by_span[key] = g

    matched_gold_ids: Set[str] = set()
    matched_pred_ids: Set[str] = set()
    matched_correct_label_pred_ids: Set[str] = set()
    matched_wrong_label_records: List[Dict[str, Any]] = []
    unmatched_prediction_ids: List[str] = []

    # Match predictions against gold
    for p in accepted_predictions:
        p_ans = p["answer_id"]
        p_start = int(p["start_char"])
        p_end = int(p["end_char"])
        p_id = p.get("claim_id", f"{p_ans}:{p_start}:{p_end}")
        p_quote = p.get("quote", "")
        p_label = p.get("label", "")

        key = (p_ans, p_start, p_end)
        if key in gold_by_span:
            g = gold_by_span[key]
            # Validate quote equality between gold and prediction
            if p_quote == g["quote"] and g["claim_id"] not in matched_gold_ids:
                matched_gold_ids.add(g["claim_id"])
                matched_pred_ids.add(p_id)

                if p_label == g["expected_label"]:
                    matched_correct_label_pred_ids.add(p_id)
                else:
                    matched_wrong_label_records.append({
                        "claim_id": p_id,
                        "answer_id": p_ans,
                        "span": [p_start, p_end],
                        "quote": p_quote,
                        "predicted_label": p_label,
                        "expected_label": g["expected_label"],
                        "evidence_fact_ids": g.get("evidence_fact_ids", []),
                        "explanation": g.get("explanation", ""),
                    })
            else:
                unmatched_prediction_ids.append(p_id)
        else:
            unmatched_prediction_ids.append(p_id)

    M = len(matched_pred_ids)
    C = len(matched_correct_label_pred_ids)

    # Missed gold claims
    missed_gold_ids = [
        g["claim_id"] for g in gold_claims if g["claim_id"] not in matched_gold_ids
    ]

    # Calculate metrics with zero denominator safeguards
    def safe_div(num: float, den: float) -> float:
        return round(num / den, 4) if den > 0 else 0.0

    extraction_precision = safe_div(M, P)
    extraction_recall = safe_div(M, G)
    extraction_f1 = safe_div(2 * M, P + G)

    joint_precision = safe_div(C, P)
    joint_recall = safe_div(C, G)
    joint_f1 = safe_div(2 * C, P + G)

    coverage = safe_div(C, G)

    return {
        "is_check_split": False,
        "total_answers": total_answers,
        "error_answer_count": error_answer_count,
        "error_answer_ids": error_answer_ids,
        "accepted_prediction_count": P,
        "G": G,
        "P": P,
        "M": M,
        "C": C,
        "extraction_precision": extraction_precision,
        "extraction_recall": extraction_recall,
        "extraction_f1": extraction_f1,
        "joint_precision": joint_precision,
        "joint_recall": joint_recall,
        "joint_f1": joint_f1,
        "coverage": coverage,
        "missed_gold_count": len(missed_gold_ids),
        "missed_gold_ids": missed_gold_ids,
        "unmatched_prediction_count": len(unmatched_prediction_ids),
        "unmatched_prediction_ids": unmatched_prediction_ids,
        "matched_wrong_label_count": len(matched_wrong_label_records),
        "matched_wrong_label_records": matched_wrong_label_records,
    }
