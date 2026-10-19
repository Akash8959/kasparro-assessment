"""
Analyze v1 development run report.
Groups errors into:
  1. Missed claims (from answer-level errors vs extraction omissions)
  2. Extra non-claim sentences (unmatched predictions)
  3. Wrong labels
  4. Span / validation issues
"""

import json
from pathlib import Path


def main():
    reports = list(Path("reports").glob("eval_development_v1_*.json"))
    if not reports:
        print("No v1 development report found.")
        return

    report_file = sorted(reports)[-1]
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["metrics"]
    print("=" * 60)
    print("V1 DEVELOPMENT REPORT ANALYSIS")
    print("=" * 60)
    print(f"Report File: {report_file.name}")
    print(f"Total Answers: {data['total_answers']}")
    print(f"Successful Answers: {data['successful_answers']}")
    print(f"Error Answers: {data['error_answers']}")
    print(f"Accepted Predictions (P): {metrics['P']}")
    print(f"Gold Claims (G): {metrics['G']}")
    print(f"Exact Matches (M): {metrics['M']}")
    print(f"Correct Labels (C): {metrics['C']}")
    print("-" * 60)
    print(f"Extraction Precision: {metrics['extraction_precision']:.4f}")
    print(f"Extraction Recall:    {metrics['extraction_recall']:.4f}")
    print(f"Extraction F1:        {metrics['extraction_f1']:.4f}")
    print(f"Joint Precision:      {metrics['joint_precision']:.4f}")
    print(f"Joint Recall:         {metrics['joint_recall']:.4f}")
    print(f"Joint F1:             {metrics['joint_f1']:.4f}")
    print(f"Coverage (C/G):       {metrics['coverage']:.4f}")
    print("=" * 60)

    # 1. Answer-level Errors
    print("\n[GROUP 1: ANSWER-LEVEL ERRORS]")
    for err in data["error_records"]:
        print(f"Answer {err['answer_id']}:")
        print(f"  Type: {err['error']['error_type']}")
        print(f"  Message: {err['error']['message']}")

    # 2. Wrong Labels
    print(f"\n[GROUP 2: WRONG LABELS ({metrics['matched_wrong_label_count']} records)]")
    for wl in metrics["matched_wrong_label_records"]:
        print(f"Claim {wl['claim_id']}:")
        print(f"  Predicted: {wl['predicted_label']} | Expected: {wl['expected_label']}")
        print(f"  Quote: {wl['quote']}")
        print(f"  Explanation: {wl['explanation']}")

    # 3. Unmatched Extra Predictions
    unmatched = metrics.get("unmatched_prediction_ids", [])
    print(f"\n[GROUP 3: EXTRA / UNMATCHED PREDICTIONS ({len(unmatched)} records)]")
    if unmatched:
        for uid in unmatched:
            print(f"  - {uid}")
    else:
        print("  None! (Zero extra non-claim predictions, 100% precision)")

    # 4. Missed Gold Claims
    missed = metrics.get("missed_gold_ids", [])
    error_answer_ids = set(metrics.get("error_answer_ids", []))
    missed_from_errors = [gid for gid in missed if gid.split(":")[0] in error_answer_ids]
    missed_from_success = [gid for gid in missed if gid.split(":")[0] not in error_answer_ids]

    print(f"\n[GROUP 4: MISSED GOLD CLAIMS ({len(missed)} total)]")
    print(f"  - Missed due to 3 Answer Errors ({sorted(error_answer_ids)}): {len(missed_from_errors)} claims")
    for ans in sorted(error_answer_ids):
        count = sum(1 for gid in missed_from_errors if gid.split(":")[0] == ans)
        print(f"      {ans}: {count} missed gold claims")

    print(f"  - Missed from Successful Answers: {len(missed_from_success)} claims")
    ans_missed_map = {}
    for gid in missed_from_success:
        ans = gid.split(":")[0]
        ans_missed_map[ans] = ans_missed_map.get(ans, 0) + 1
    for ans, c in sorted(ans_missed_map.items()):
        print(f"      {ans}: {c} missed claims")


if __name__ == "__main__":
    main()
