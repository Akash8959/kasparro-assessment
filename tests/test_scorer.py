"""
Unit tests for evaluation scorer arithmetic, edge cases, and report generation.
"""

import pytest
from harness.scorer import score_predictions


def test_scorer_arithmetic_spec_benchmark():
    """Verify standard arithmetic benchmark: G=10, P=10, M=8, C=7."""
    gold = [
        {"answer_id": "A01", "claim_id": f"A01:{i*10}:{(i+1)*10}", "start_char": i*10, "end_char": (i+1)*10, "quote": f"q{i}", "expected_label": "supported" if i < 7 else "contradicted"}
        for i in range(10)
    ]

    # 8 matches: 7 have correct label, 1 has wrong label
    matched_preds = [
        {"claim_id": f"A01:{i*10}:{(i+1)*10}", "start_char": i*10, "end_char": (i+1)*10, "quote": f"q{i}", "label": "supported", "fact_ids": [], "reason": ""}
        for i in range(8)
    ]
    # 2 extra predictions
    extra_preds = [
        {"claim_id": f"A01:{100+i*10}:{100+(i+1)*10}", "start_char": 100+i*10, "end_char": 100+(i+1)*10, "quote": f"ex{i}", "label": "unknown", "fact_ids": [], "reason": ""}
        for i in range(8, 10)
    ]

    answer_results = [{"answer_id": "A01", "status": "success", "claims": matched_preds + extra_preds}]
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


def test_scorer_perfect_evaluation():
    """Verify 100% precision, recall, F1, and coverage when P == M == C == G."""
    gold = [
        {"answer_id": "A01", "claim_id": "A01:0:10", "start_char": 0, "end_char": 10, "quote": "Sentence 1", "expected_label": "supported"},
        {"answer_id": "A01", "claim_id": "A01:15:25", "start_char": 15, "end_char": 25, "quote": "Sentence 2", "expected_label": "contradicted"},
        {"answer_id": "A02", "claim_id": "A02:0:12", "start_char": 0, "end_char": 12, "quote": "Sentence 3", "expected_label": "unknown"},
    ]

    answer_results = [
        {
            "answer_id": "A01",
            "status": "success",
            "claims": [
                {"claim_id": "A01:0:10", "start_char": 0, "end_char": 10, "quote": "Sentence 1", "label": "supported", "fact_ids": ["f1"], "reason": "r1"},
                {"claim_id": "A01:15:25", "start_char": 15, "end_char": 25, "quote": "Sentence 2", "label": "contradicted", "fact_ids": ["f2"], "reason": "r2"},
            ]
        },
        {
            "answer_id": "A02",
            "status": "success",
            "claims": [
                {"claim_id": "A02:0:12", "start_char": 0, "end_char": 12, "quote": "Sentence 3", "label": "unknown", "fact_ids": [], "reason": "r3"},
            ]
        }
    ]

    res = score_predictions(gold, answer_results)
    assert res["G"] == 3
    assert res["P"] == 3
    assert res["M"] == 3
    assert res["C"] == 3
    assert res["extraction_precision"] == 1.0
    assert res["extraction_recall"] == 1.0
    assert res["extraction_f1"] == 1.0
    assert res["joint_precision"] == 1.0
    assert res["joint_recall"] == 1.0
    assert res["joint_f1"] == 1.0
    assert res["coverage"] == 1.0
    assert len(res["missed_gold_ids"]) == 0
    assert len(res["unmatched_prediction_ids"]) == 0
    assert len(res["matched_wrong_label_records"]) == 0


def test_scorer_zero_predictions_all_missed():
    """Verify zero predictions resulting in 0 precision, recall, and F1."""
    gold = [
        {"answer_id": "A01", "claim_id": "A01:0:10", "start_char": 0, "end_char": 10, "quote": "q0", "expected_label": "supported"}
    ]
    answer_results = [{"answer_id": "A01", "status": "success", "claims": []}]

    res = score_predictions(gold, answer_results)
    assert res["G"] == 1
    assert res["P"] == 0
    assert res["M"] == 0
    assert res["C"] == 0
    assert res["extraction_precision"] == 0.0
    assert res["extraction_recall"] == 0.0
    assert res["extraction_f1"] == 0.0
    assert res["joint_f1"] == 0.0
    assert res["coverage"] == 0.0
    assert res["missed_gold_ids"] == ["A01:0:10"]


def test_scorer_wrong_labels_penalty():
    """Verify extraction is 1.0 but joint metrics drop when labels disagree."""
    gold = [
        {"answer_id": "A01", "claim_id": "A01:0:10", "start_char": 0, "end_char": 10, "quote": "q0", "expected_label": "supported"},
        {"answer_id": "A01", "claim_id": "A01:20:30", "start_char": 20, "end_char": 30, "quote": "q1", "expected_label": "contradicted"},
    ]
    answer_results = [
        {
            "answer_id": "A01",
            "status": "success",
            "claims": [
                {"claim_id": "A01:0:10", "start_char": 0, "end_char": 10, "quote": "q0", "label": "supported", "fact_ids": [], "reason": ""},
                {"claim_id": "A01:20:30", "start_char": 20, "end_char": 30, "quote": "q1", "label": "unknown", "fact_ids": [], "reason": ""},
            ]
        }
    ]

    res = score_predictions(gold, answer_results)
    assert res["G"] == 2
    assert res["P"] == 2
    assert res["M"] == 2
    assert res["C"] == 1  # only 1 correct label
    assert res["extraction_f1"] == 1.0
    assert res["joint_precision"] == 0.50
    assert res["joint_recall"] == 0.50
    assert res["joint_f1"] == 0.50
    assert res["coverage"] == 0.50
    assert len(res["matched_wrong_label_records"]) == 1
    assert res["matched_wrong_label_records"][0]["claim_id"] == "A01:20:30"
    assert res["matched_wrong_label_records"][0]["predicted_label"] == "unknown"
    assert res["matched_wrong_label_records"][0]["expected_label"] == "contradicted"


def test_scorer_answer_level_error_penalizes_metrics():
    """Verify that an answer marked as error contributes 0 claims and counts against recall/coverage."""
    gold = [
        {"answer_id": "A01", "claim_id": "A01:0:10", "start_char": 0, "end_char": 10, "quote": "q0", "expected_label": "supported"},
        {"answer_id": "A02", "claim_id": "A02:0:10", "start_char": 0, "end_char": 10, "quote": "q1", "expected_label": "supported"},
    ]
    # A01 succeeded, A02 failed with error
    answer_results = [
        {
            "answer_id": "A01",
            "status": "success",
            "claims": [
                {"claim_id": "A01:0:10", "start_char": 0, "end_char": 10, "quote": "q0", "label": "supported", "fact_ids": [], "reason": ""},
            ]
        },
        {
            "answer_id": "A02",
            "status": "error",
            "claims": []
        }
    ]

    res = score_predictions(gold, answer_results)
    assert res["G"] == 2
    assert res["P"] == 1
    assert res["M"] == 1
    assert res["C"] == 1
    assert res["extraction_precision"] == 1.0
    assert res["extraction_recall"] == 0.5
    assert res["joint_precision"] == 1.0
    assert res["joint_recall"] == 0.5
    assert res["coverage"] == 0.5
    assert res["missed_gold_ids"] == ["A02:0:10"]


def test_scorer_check_split_no_gold_returns_null():
    """Verify that when gold is empty or None (check split), all quality metrics are strictly None."""
    answer_results = [
        {
            "answer_id": "A41",
            "status": "success",
            "claims": [
                {"claim_id": "A41:0:10", "start_char": 0, "end_char": 10, "quote": "q0", "label": "supported", "fact_ids": [], "reason": ""},
            ]
        }
    ]

    res = score_predictions(None, answer_results)
    assert res is None or res.get("G") is None
