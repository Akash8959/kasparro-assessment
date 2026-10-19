"""
Unit tests for Phase 1 core pieces: loader, facts_tool, spans, validate, and scorer arithmetic.
"""

import pytest
from agent.loader import (
    load_answers,
    load_brand_facts,
    get_dataset_hashes,
    get_valid_fact_ids,
    get_valid_brand_ids,
)
from harness.gold_loader import load_development_gold
from agent.facts_tool import (
    get_brand_facts,
    execute_tool_call,
    UnknownBrandError,
    IllegalToolError,
    ToolExecutionError,
)
from agent.spans import locate_claims, find_quote_span, SpanError
from agent.validate import validate_claims, ClaimValidationError
from harness.scorer import score_predictions


def test_loader_csv_hashes():
    """Verify SHA-256 hashes for data files are computed and non-empty."""
    hashes = get_dataset_hashes("data")
    assert "answers.csv" in hashes
    assert "brand_facts.csv" in hashes
    assert "development_gold.csv" in hashes
    assert len(hashes["answers.csv"]) == 64
    assert hashes["answers.csv"] == "fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724"


def test_loader_preserves_raw_ai_answer():
    """Verify answers loader preserves exact ai_answer without line splitting or stripping."""
    answers = load_answers("data/answers.csv")
    assert len(answers) == 50
    a01 = next(a for a in answers if a["answer_id"] == "A01")
    assert "\n" in a01["ai_answer"]
    assert a01["split"] == "development"
    assert a01["brand_id"] == "cleartrail"


def test_loader_brand_facts():
    """Verify brand facts have correct types and fact_version compatibility."""
    facts = load_brand_facts("data/brand_facts.csv")
    assert len(facts) == 132
    f0 = facts[0]
    assert "fact_id" in f0
    assert "brand_id" in f0
    assert "value_type" in f0


def test_facts_tool_valid_and_invalid():
    """Verify tool execution with known brand, unknown brand, and illegal tool name."""
    res = execute_tool_call("get_brand_facts", {"brand_id": "cleartrail"})
    assert res["fact_version"] == "speaker-catalogue-v4"
    assert res["brand_id"] == "cleartrail"
    assert len(res["facts"]) > 0

    with pytest.raises(UnknownBrandError):
        execute_tool_call("get_brand_facts", {"brand_id": "nonexistent_brand"})

    with pytest.raises(IllegalToolError):
        execute_tool_call("arbitrary_tool", {"brand_id": "cleartrail"})


def test_spans_offset_resolution():
    """Verify Unicode code point offset resolution and duplicate quote sequential cursor tracking."""
    text = "ClearTrail Mini is 1200 INR. Another sentence. ClearTrail Mini is 1200 INR."
    raw = [
        {"quote": "ClearTrail Mini is 1200 INR.", "label": "supported", "fact_ids": [], "reason": ""},
        {"quote": "Another sentence.", "label": "unknown", "fact_ids": [], "reason": ""},
        {"quote": "ClearTrail Mini is 1200 INR.", "label": "supported", "fact_ids": [], "reason": ""},
    ]
    located = locate_claims("A01", text, raw)
    assert len(located) == 3
    # Check first occurrence
    assert located[0]["start_char"] == 0
    assert located[0]["end_char"] == 28
    # Check second occurrence resolved after cursor
    assert located[2]["start_char"] == 47
    assert located[2]["end_char"] == 75
    assert text[located[2]["start_char"]:located[2]["end_char"]] == "ClearTrail Mini is 1200 INR."


def test_spans_missing_quote_raises():
    """Verify SpanError when a quote is not present in text."""
    text = "Some answer text."
    raw = [{"quote": "Nonexistent text in answer.", "label": "supported", "fact_ids": []}]
    with pytest.raises(SpanError):
        locate_claims("A01", text, raw)


def test_validate_claims_strict():
    """Verify validation of bounds, quote equality, whitespace, overlap, labels, and fact IDs."""
    text = "First claim sentence. Second claim sentence."
    valid_facts = {"ct.mini.price"}

    claims = [
        {
            "claim_id": "A01:0:21",
            "start_char": 0,
            "end_char": 21,
            "quote": "First claim sentence.",
            "label": "supported",
            "fact_ids": ["ct.mini.price"],
            "reason": "Verified from catalogue.",
        },
        {
            "claim_id": "A01:22:44",
            "start_char": 22,
            "end_char": 44,
            "quote": "Second claim sentence.",
            "label": "unknown",
            "fact_ids": [],
            "reason": "Not in catalogue.",
        },
    ]

    validated = validate_claims("A01", text, claims, valid_facts)
    assert len(validated) == 2

    # Test overlapping spans
    overlapping = [
        {"claim_id": "A01:0:25", "start_char": 0, "end_char": 25, "quote": text[0:25], "label": "supported"},
        {"claim_id": "A01:20:44", "start_char": 20, "end_char": 44, "quote": text[20:44], "label": "supported"},
    ]
    with pytest.raises(ClaimValidationError):
        validate_claims("A01", text, overlapping, valid_facts)

    # Test invalid label
    bad_label = [
        {"claim_id": "A01:0:21", "start_char": 0, "end_char": 21, "quote": "First claim sentence.", "label": "verified"}
    ]
    with pytest.raises(ClaimValidationError):
        validate_claims("A01", text, bad_label, valid_facts)

    # Test unknown fact id
    bad_fact = [
        {"claim_id": "A01:0:21", "start_char": 0, "end_char": 21, "quote": "First claim sentence.", "label": "supported", "fact_ids": ["invented.fact"]}
    ]
    with pytest.raises(ClaimValidationError):
        validate_claims("A01", text, bad_fact, valid_facts)


def test_scorer_arithmetic_spec_example():
    """Verify scorer arithmetic check from the specification: G=10, P=10, M=8, C=7 -> 0.80 and 0.70."""
    gold = [
        {
            "answer_id": "A01",
            "claim_id": f"A01:{i*10}:{(i+1)*10}",
            "start_char": i * 10,
            "end_char": (i + 1) * 10,
            "quote": f"claim_{i}",
            "expected_label": "supported" if i < 7 else "contradicted",
        }
        for i in range(10)
    ]

    # 8 matches: 7 correct label, 1 wrong label
    matched_preds = [
        {
            "claim_id": f"A01:{i*10}:{(i+1)*10}",
            "start_char": i * 10,
            "end_char": (i + 1) * 10,
            "quote": f"claim_{i}",
            "label": "supported",
            "fact_ids": [],
            "reason": "",
        }
        for i in range(8)
    ]
    # 2 extra predictions
    extra_preds = [
        {
            "claim_id": f"A01:{100+i*10}:{100+(i+1)*10}",
            "start_char": 100 + i * 10,
            "end_char": 100 + (i + 1) * 10,
            "quote": f"extra_{i}",
            "label": "unknown",
            "fact_ids": [],
            "reason": "",
        }
        for i in range(8, 10)
    ]

    answer_results = [
        {
            "answer_id": "A01",
            "status": "success",
            "claims": matched_preds + extra_preds,
        }
    ]

    res = score_predictions(gold, answer_results)
    assert res["G"] == 10
    assert res["P"] == 10
    assert res["M"] == 8
    assert res["C"] == 7
    assert res["extraction_precision"] == 0.80
    assert res["extraction_recall"] == 0.80
    assert res["extraction_f1"] == 0.80
    assert res["joint_precision"] == 0.70
    assert res["joint_recall"] == 0.70
    assert res["joint_f1"] == 0.70
    assert res["coverage"] == 0.70
