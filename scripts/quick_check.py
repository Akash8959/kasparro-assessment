"""
Quick check script: inspects 5 missed claims across 5 different answers from v1 report.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.gold_loader import load_development_gold


def main():
    report_file = list(Path("reports").glob("eval_development_v1_*.json"))[0]
    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("=" * 60)
    print("V1 GENERATION SETTINGS")
    print("=" * 60)
    print(f"Provider: {report['provider']}")
    print(f"Model: {report['model']}")
    print(f"Settings: {json.dumps(report['settings'], indent=2)}")

    gold_map = {g["claim_id"]: g for g in load_development_gold()}
    missed = report["metrics"]["missed_gold_ids"]

    sample_answers = ["A08", "A10", "A11", "A17", "A19"]

    print("\n" + "=" * 60)
    print("5 MISSED CLAIMS FROM 5 DISTINCT ANSWERS")
    print("=" * 60)

    for ans_id in sample_answers:
        ans_missed = [gold_map[gid] for gid in missed if gid.startswith(ans_id + ":")]
        first_missed = ans_missed[0]

        saved_preds = [p for p in report["predictions"] if p["answer_id"] == ans_id]
        pred_quotes = [p["quote"] for p in saved_preds]

        print(f"\n--- Answer {ans_id} | Missed Gold Claim: {first_missed['claim_id']} ---")
        print(f"Gold Quote:\n  \"{first_missed['quote']}\"")
        print(f"Expected Label: {first_missed['expected_label']}")
        print(f"Explanation: {first_missed['explanation']}")
        print(f"Predictions saved for {ans_id}: {len(saved_preds)}")

        if first_missed["quote"] in pred_quotes:
            print("Model action: Quote was extracted but matched differently")
        else:
            overlapping = [q for q in pred_quotes if first_missed["quote"] in q or q in first_missed["quote"]]
            if overlapping:
                print(f"Model action: PARTIALLY OVERLAPPING ({len(overlapping)}): {overlapping}")
            else:
                print("Model action: COMPLETELY SKIPPED (The model did not extract this sentence at all)")


if __name__ == "__main__":
    main()
