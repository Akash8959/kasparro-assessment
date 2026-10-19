"""
Span locator module for resolving exact character offsets in ai_answer.
Computes 0-based, end-exclusive Unicode code-point offsets (start_char, end_char).
Resolves repeated quote occurrences sequentially using source position context.
Guarantees ai_answer[start_char:end_char] == quote.
"""

from typing import Any, Dict, List, Optional, Tuple


class SpanError(Exception):
    """Raised when a quote cannot be resolved to a valid span."""
    pass


def find_quote_span(
    ai_answer: str,
    quote: str,
    search_start: int = 0
) -> Optional[Tuple[int, int]]:
    """
    Find start_char and end_char of quote in ai_answer.
    Tries searching from search_start first (for ordered source progression).
    If not found after search_start, searches the entire text from position 0.
    Returns (start_char, end_char) or None if not found.
    """
    clean_quote = quote.strip()
    if not clean_quote:
        return None

    # First attempt: search forward from current cursor
    idx = ai_answer.find(clean_quote, search_start)
    if idx != -1:
        return idx, idx + len(clean_quote)

    # Second attempt: search from beginning of ai_answer
    idx = ai_answer.find(clean_quote)
    if idx != -1:
        return idx, idx + len(clean_quote)

    return None


def locate_claims(
    answer_id: str,
    ai_answer: str,
    raw_claims: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert model output quotes into validated span records with:
      - claim_id: f"{answer_id}:{start_char}:{end_char}"
      - start_char: int
      - end_char: int
      - quote: ai_answer[start_char:end_char]
      - label: str
      - fact_ids: list[str]
      - reason: str
    Tracks the source search cursor across sequential claims to properly resolve duplicate sentences.
    Raises SpanError if a quote cannot be found in ai_answer.
    """
    located = []
    cursor = 0

    for idx, raw in enumerate(raw_claims):
        quote = raw.get("quote", "")
        if not isinstance(quote, str) or not quote.strip():
            raise SpanError(f"Claim index {idx} in answer {answer_id} has empty or non-string quote: {raw}")

        span = find_quote_span(ai_answer, quote, search_start=cursor)
        if span is None:
            raise SpanError(
                f"Could not locate quote in answer {answer_id}: {quote[:60]!r}..."
            )

        start_char, end_char = span
        # Update cursor to after this span
        cursor = end_char

        # Quote must equal the exact substring
        exact_quote = ai_answer[start_char:end_char]

        # Extract other fields
        label = str(raw.get("label", "")).strip().lower()
        fact_ids = raw.get("fact_ids", [])
        if not isinstance(fact_ids, list):
            fact_ids = [fact_ids] if fact_ids else []
        fact_ids = [str(fid).strip() for fid in fact_ids if fid]

        reason = str(raw.get("reason", "")).strip()

        located.append({
            "claim_id": f"{answer_id}:{start_char}:{end_char}",
            "start_char": start_char,
            "end_char": end_char,
            "quote": exact_quote,
            "label": label,
            "fact_ids": fact_ids,
            "reason": reason,
        })

    return located
