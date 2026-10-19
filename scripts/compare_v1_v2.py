"""
Detailed comparison between v1 and v2 evaluation runs on development split.
"""

import json
from pathlib import Path


def load_report(pattern: str):
    files = list(Path("reports").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern}")
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    v1 = load_report("eval_development_v1_*.json")
    v2 = load_report("eval_development_v2_*.json")
    check_v1 = load_report("eval_check_v1_*.json")
    check_v2 = load_report("eval_check_v2_*.json")

    m1 = v1["metrics"]
    m2 = v2["metrics"]

    print("=" * 60)
    print("DEVELOPMENT METRICS COMPARISON (v1 vs v2)")
    print("=" * 60)
    print(f"{'Metric':<25} {'v1':<15} {'v2':<15} {'Delta':<10}")
    print("-" * 60)
    for k in ["G", "P", "M", "C"]:
        val1 = m1[k]
        val2 = m2[k]
        diff = val2 - val1
        print(f"{k:<25} {val1:<15} {val2:<15} {diff:+d}")

    for k in [
        "extraction_precision",
        "extraction_recall",
        "extraction_f1",
        "joint_precision",
        "joint_recall",
        "joint_f1",
        "coverage",
    ]:
        val1 = m1[k]
        val2 = m2[k]
        diff = val2 - val1
        print(f"{k:<25} {val1:<15.4f} {val2:<15.4f} {diff:+.4f}")

    print("\n" + "=" * 60)
    print("ANSWER-LEVEL ERRORS")
    print("=" * 60)
    err1 = set(m1["error_answer_ids"])
    err2 = set(m2["error_answer_ids"])
    print(f"v1 error count: {len(err1)} ({sorted(err1)})")
    print(f"v2 error count: {len(err2)} ({sorted(err2)})")
    print(f"Errors in v1 only: {sorted(err1 - err2)}")
    print(f"Errors in v2 only: {sorted(err2 - err1)}")
    print(f"Errors in both: {sorted(err1 & err2)}")

    # Analyze predictions
    # Map (answer_id, start_char, end_char) -> pred
    preds1 = {(p["answer_id"], p["start_char"], p["end_char"]): p for p in v1["predictions"]}
    preds2 = {(p["answer_id"], p["start_char"], p["end_char"]): p for p in v2["predictions"]}

    # Missed in v1:
    missed_v1 = set(m1["missed_gold_ids"])
    missed_v2 = set(m2["missed_gold_ids"])

    recovered = missed_v1 - missed_v2
    still_missed = missed_v1 & missed_v2
    newly_missed = missed_v2 - missed_v1

    print("\n" + "=" * 60)
    print("MISSED CLAIMS ANALYSIS")
    print("=" * 60)
    print(f"Missed in v1: {len(missed_v1)}")
    print(f"Missed in v2: {len(missed_v2)}")
    print(f"Recovered by v2: {len(recovered)}")
    print(f"Still missed in v2: {len(still_missed)}")
    print(f"Newly missed in v2: {len(newly_missed)} (due to error answers {sorted(err2 - err1)})")

    # Check for regressions
    # A regression is a claim that was correctly predicted (in v1: matched and correct label)
    # but in v2 was either missed or had wrong label.
    wrong1 = {r["claim_id"]: r for r in m1["matched_wrong_label_records"]}
    wrong2 = {r["claim_id"]: r for r in m2["matched_wrong_label_records"]}

    correct1 = set()
    for p in v1["predictions"]:
        cid = p["claim_id"]
        if cid not in wrong1 and cid not in m1.get("unmatched_prediction_ids", []):
            correct1.add(cid)

    regressed_missed = [cid for cid in correct1 if cid in missed_v2]
    regressed_wrong = [cid for cid in correct1 if cid in wrong2]

    print("\n" + "=" * 60)
    print("REGRESSIONS ANALYSIS")
    print("=" * 60)
    print(f"Claims correctly predicted in v1: {len(correct1)}")
    print(f"Regressed to MISSED in v2: {len(regressed_missed)} (all from new error answers: {sorted(set(c.split(':')[0] for c in regressed_missed))})")
    print(f"Regressed to WRONG LABEL in v2: {len(regressed_wrong)} ({regressed_wrong})")
    if regressed_wrong:
        for cid in regressed_wrong:
            print(f"  {cid}: v2 predicted {wrong2[cid]['predicted_label']}, expected {wrong2[cid]['expected_label']}")

    print("\n" + "=" * 60)
    print("CHECK SPLIT SUMMARY")
    print("=" * 60)
    cm1 = check_v1["metrics"]
    cm2 = check_v2["metrics"]
    print(f"v1 Check Split: Total Answers={cm1['total_answers']}, Errors={cm1['error_answer_count']} ({cm1['error_answer_ids']}), Accepted Claims={cm1['accepted_prediction_count']}")
    print(f"v2 Check Split: Total Answers={cm2['total_answers']}, Errors={cm2['error_answer_count']} ({cm2['error_answer_ids']}), Accepted Claims={cm2['accepted_prediction_count']}")
    print(f"Quality metrics are strictly null: G={cm1['G']}, F1={cm1['joint_f1']}")


if __name__ == "__main__":
    main()
