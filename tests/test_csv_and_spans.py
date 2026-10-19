"""
Tests for CSV parsing, Unicode character offsets, span resolution, and strict claim validation.
"""

import pytest
from pathlib import Path
from agent.loader import (
    load_answers,
    load_brand_facts,
    get_dataset_hashes,
    get_valid_fact_ids,
    get_valid_brand_ids,
)
from agent.spans import locate_claims, find_quote_span, SpanError
from agent.validate import validate_claims, ClaimValidationError


def test_csv_preserves_raw_ai_answer_formatting():
    """Verify raw ai_answer preserves internal newlines, quotation marks, and spacing."""
    answers = load_answers("data/answers.csv")
    assert len(answers) == 50

    # Ensure answers have required fields
    for a in answers:
        assert "answer_id" in a
        assert "brand_id" in a
        assert "question" in a
        assert "ai_answer" in a
        assert "split" in a
        assert a["split"] in ("development", "check")

    # A01 has multi-paragraph structure
    a01 = next(a for a in answers if a["answer_id"] == "A01")
    assert "\n" in a01["ai_answer"]
    assert len(a01["ai_answer"]) > 100

    # Check split partitioning
    dev_split = load_answers("data/answers.csv", split="development")
    check_split = load_answers("data/answers.csv", split="check")
    assert len(dev_split) == 40
    assert len(check_split) == 10
    assert {a["answer_id"] for a in dev_split}.isdisjoint({a["answer_id"] for a in check_split})


def test_brand_facts_schema_and_types():
    """Verify brand facts table contains valid rows and types."""
    facts = load_brand_facts("data/brand_facts.csv")
    assert len(facts) == 132

    valid_facts = get_valid_fact_ids("data/brand_facts.csv")
    assert len(valid_facts) == 132

    valid_brands = get_valid_brand_ids("data/brand_facts.csv")
    assert valid_brands == {"cleartrail", "cloudnote", "stonewave"}

    for fact in facts:
        assert fact["brand_id"] in valid_brands
        assert fact["fact_id"] in valid_facts
        assert len(fact["model_id"]) > 0
        assert len(fact["field"]) > 0
        assert fact["value_type"] in ("string", "number", "boolean", "list", "currency", "date", "json", "text")


def test_unicode_code_point_span_offsets():
    """Verify locate_claims handles Unicode characters (multibyte code points) correctly."""
    # Text with multibyte Unicode characters (e.g., em-dash —, curly quotes “ ”, euro €)
    text = "ClearTrail™ offers 30—day returns “free of charge” for €50."
    claims_raw = [
        {"quote": "ClearTrail™ offers 30—day returns", "label": "supported", "fact_ids": [], "reason": "Doc."},
        {"quote": "“free of charge” for €50.", "label": "unknown", "fact_ids": [], "reason": "Unknown."},
    ]

    located = locate_claims("A99", text, claims_raw)
    assert len(located) == 2
    # Verify exact character slicing
    for c in located:
        assert text[c["start_char"]:c["end_char"]] == c["quote"]


def test_spans_duplicate_quote_sequential_tracking():
    """Verify repeated identical sentences in an answer are resolved sequentially without span collision."""
    sentence = "The battery lasts up to 10 hours."
    text = f"{sentence} It comes with USB-C. {sentence}"
    claims_raw = [
        {"quote": sentence, "label": "supported", "fact_ids": [], "reason": "First mention."},
        {"quote": sentence, "label": "supported", "fact_ids": [], "reason": "Second mention."},
    ]

    located = locate_claims("A99", text, claims_raw)
    assert len(located) == 2
    assert located[0]["start_char"] == 0
    assert located[0]["end_char"] == len(sentence)
    assert located[1]["start_char"] > located[0]["end_char"]
    assert located[1]["end_char"] == len(text)
    assert located[0]["start_char"] != located[1]["start_char"]


def test_spans_missing_quote_raises_span_error():
    """Verify SpanError is raised when model hallucinated a quote not in raw text."""
    text = "The quick brown fox jumps over the lazy dog."
    claims_raw = [
        {"quote": "The fast brown fox leaps over the sleeping dog.", "label": "supported", "fact_ids": [], "reason": "x"}
    ]
    with pytest.raises(SpanError) as exc_info:
        locate_claims("A99", text, claims_raw)
    assert "could not locate" in str(exc_info.value).lower()


def test_validate_claims_valid_cases():
    """Verify validate_claims accepts properly formatted, sorted, non-overlapping claims."""
    text = "Sentence one. Sentence two. Sentence three."
    valid_facts = {"ct.fact.1", "ct.fact.2"}

    claims = [
        {
            "claim_id": "A01:0:13",
            "start_char": 0,
            "end_char": 13,
            "quote": "Sentence one.",
            "label": "supported",
            "fact_ids": ["ct.fact.1"],
            "reason": "Found in catalogue.",
        },
        {
            "claim_id": "A01:14:27",
            "start_char": 14,
            "end_char": 27,
            "quote": "Sentence two.",
            "label": "contradicted",
            "fact_ids": ["ct.fact.2"],
            "reason": "Directly contradicted by fact 2.",
        },
        {
            "claim_id": "A01:28:43",
            "start_char": 28,
            "end_char": 43,
            "quote": "Sentence three.",
            "label": "unknown",
            "fact_ids": [],
            "reason": "No mention in catalogue.",
        },
    ]

    res = validate_claims("A01", text, claims, valid_facts)
    assert len(res) == 3


def test_validate_claims_empty_is_valid():
    """Verify empty claims list is valid for an answer with no claims."""
    text = "Thank you for contacting customer support!"
    res = validate_claims("A01", text, [], {"fact.1"})
    assert res == []


def test_validate_claims_overlapping_spans_rejected():
    """Verify overlapping character spans are strictly rejected."""
    text = "This is a single long sentence with multiple parts."
    valid_facts = {"f1"}

    # Overlap between 0-25 and 20-40
    claims = [
        {"claim_id": "A01:0:25", "start_char": 0, "end_char": 25, "quote": text[0:25], "label": "supported", "fact_ids": ["f1"], "reason": "ok."},
        {"claim_id": "A01:20:40", "start_char": 20, "end_char": 40, "quote": text[20:40], "label": "supported", "fact_ids": ["f1"], "reason": "ok."},
    ]
    with pytest.raises(ClaimValidationError) as exc:
        validate_claims("A01", text, claims, valid_facts)
    assert "overlapping" in str(exc.value).lower()


def test_validate_claims_leading_trailing_whitespace_quote():
    """Verify quote with leading or trailing whitespace is rejected."""
    text = " Sentence with space. "
    valid_facts = {"f1"}

    claims = [
        {"claim_id": "A01:0:22", "start_char": 0, "end_char": 22, "quote": " Sentence with space. ", "label": "unknown", "fact_ids": [], "reason": "ok."}
    ]
    with pytest.raises(ClaimValidationError) as exc:
        validate_claims("A01", text, claims, valid_facts)
    assert "whitespace" in str(exc.value).lower()


def test_validate_claims_invalid_label_rejected():
    """Verify unsupported label values are rejected."""
    text = "Product is waterproof."
    valid_facts = {"f1"}

    for bad_label in ["true", "false", "verified", "unsupported", "SUPPORTED", "none"]:
        claims = [
            {"claim_id": "A01:0:22", "start_char": 0, "end_char": 22, "quote": text, "label": bad_label, "fact_ids": [], "reason": "ok."}
        ]
        with pytest.raises(ClaimValidationError) as exc:
            validate_claims("A01", text, claims, valid_facts)
        assert "label" in str(exc.value).lower()


def test_validate_claims_unknown_fact_id_rejected():
    """Verify invented or misspelled fact IDs are strictly rejected."""
    text = "Product has Bluetooth 5.0."
    valid_facts = {"bt.spec.version"}

    claims = [
        {"claim_id": "A01:0:26", "start_char": 0, "end_char": 26, "quote": text, "label": "supported", "fact_ids": ["invented.fact.id"], "reason": "ok."}
    ]
    with pytest.raises(ClaimValidationError) as exc:
        validate_claims("A01", text, claims, valid_facts)
    assert "catalogue" in str(exc.value).lower()


def test_validate_claims_excessive_reason_sentences_rejected():
    """Verify reasons exceeding 2 sentences are rejected according to spec constraint."""
    text = "The product is red."
    valid_facts = {"color.red"}

    three_sentence_reason = "This is sentence one. This is sentence two. This is sentence three."
    claims = [
        {"claim_id": "A01:0:19", "start_char": 0, "end_char": 19, "quote": text, "label": "supported", "fact_ids": ["color.red"], "reason": three_sentence_reason}
    ]
    with pytest.raises(ClaimValidationError) as exc:
        validate_claims("A01", text, claims, valid_facts)
    assert "sentence" in str(exc.value).lower()
