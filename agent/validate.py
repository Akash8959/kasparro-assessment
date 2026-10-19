"""
Claim validation module.
Validates:
  - Bounds: 0 <= start_char < end_char <= len(ai_answer)
  - Exact quote equality: ai_answer[start_char:end_char] == quote
  - No leading or trailing whitespace on quote
  - Unique claim IDs formatted as answer_id:start_char:end_char
  - Non-overlapping spans sorted in increasing start_char order
  - Allowed labels: 'supported', 'contradicted', 'unknown'
  - Existing catalogue fact_ids only (empty list is valid)
  - Reason: at most 2 sentences
"""

import re
from typing import Any, Dict, List, Optional, Set

VALID_LABELS = {"supported", "contradicted", "unknown"}


class ClaimValidationError(Exception):
    """Raised when a claim or claim set violates validation rules."""
    pass


def count_sentences(text: str) -> int:
    """Approximate sentence count using punctuation boundary splitting."""
    if not text or not text.strip():
        return 0
    # Split by period, exclamation mark, question mark followed by space or end
    sentences = [s for s in re.split(r'[.!?]+(?:\s+|$)', text.strip()) if s]
    return len(sentences)


def validate_claims(
    answer_id: str,
    ai_answer: str,
    claims: List[Dict[str, Any]],
    valid_fact_ids: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Validate a list of claims for a given answer.
    Sorts claims by start_char, verifies bounds, non-overlap, label correctness, and fact_id existence.
    Returns the validated, sorted claims list.
    Raises ClaimValidationError on any violation.
    """
    if not isinstance(claims, list):
        raise ClaimValidationError(f"Claims must be a list, got {type(claims)}")

    # Empty list is structurally valid
    if len(claims) == 0:
        return []

    # Sort claims by start_char
    sorted_claims = sorted(claims, key=lambda c: c.get("start_char", 0))

    seen_ids: Set[str] = set()
    prev_end: int = -1

    validated = []
    for idx, c in enumerate(sorted_claims):
        start_char = c.get("start_char")
        end_char = c.get("end_char")
        quote = c.get("quote")
        label = c.get("label")
        fact_ids = c.get("fact_ids", [])
        reason = c.get("reason", "")
        claim_id = c.get("claim_id")

        # 1. Type and bounds checks
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            raise ClaimValidationError(
                f"Claim {idx} has invalid integer offsets: start={start_char}, end={end_char}"
            )

        if not (0 <= start_char < end_char <= len(ai_answer)):
            raise ClaimValidationError(
                f"Claim {idx} offsets out of bounds [0, {len(ai_answer)}]: start={start_char}, end={end_char}"
            )

        # 2. Quote equality check
        expected_substring = ai_answer[start_char:end_char]
        if quote != expected_substring:
            raise ClaimValidationError(
                f"Claim {idx} quote does not match ai_answer slice. Expected {expected_substring!r}, got {quote!r}"
            )

        # 3. No leading or trailing whitespace on quote
        if quote != quote.strip():
            raise ClaimValidationError(
                f"Claim {idx} quote contains leading or trailing whitespace: {quote!r}"
            )

        # 4. claim_id format and uniqueness
        expected_claim_id = f"{answer_id}:{start_char}:{end_char}"
        if claim_id != expected_claim_id:
            raise ClaimValidationError(
                f"Claim {idx} id '{claim_id}' does not match expected '{expected_claim_id}'"
            )

        if claim_id in seen_ids:
            raise ClaimValidationError(
                f"Duplicate claim_id detected: {claim_id}"
            )
        seen_ids.add(claim_id)

        # 5. Non-overlapping check
        if start_char < prev_end:
            raise ClaimValidationError(
                f"Overlapping claim detected at index {idx}: start {start_char} is before previous end {prev_end}"
            )
        prev_end = end_char

        # 6. Label validation
        if label not in VALID_LABELS:
            raise ClaimValidationError(
                f"Invalid label '{label}' for claim {claim_id}. Must be one of {sorted(VALID_LABELS)}"
            )

        # 7. Fact IDs validation
        if not isinstance(fact_ids, list):
            raise ClaimValidationError(
                f"fact_ids must be a list for claim {claim_id}, got {type(fact_ids)}"
            )

        if valid_fact_ids is not None:
            for fid in fact_ids:
                if fid not in valid_fact_ids:
                    raise ClaimValidationError(
                        f"Unknown fact_id '{fid}' in claim {claim_id}. Not found in catalogue."
                    )

        # 8. Reason length check (at most 2 sentences)
        if count_sentences(reason) > 2:
            raise ClaimValidationError(
                f"Reason for claim {claim_id} exceeds 2 sentences: {reason}"
            )

        validated.append({
            "claim_id": claim_id,
            "start_char": start_char,
            "end_char": end_char,
            "quote": quote,
            "label": label,
            "fact_ids": list(fact_ids),
            "reason": reason,
        })

    return validated
