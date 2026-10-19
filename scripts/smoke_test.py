"""
Smoke test script for Phase 2: Runs agent v1 on A01-A03, prints answer text, tool calls, and predictions.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.agent import ClaimAgent
from agent.loader import load_answers
from harness.gold_loader import load_development_gold
from harness.scorer import score_predictions


def main():
    answers = load_answers("data/answers.csv", split="development")[:3]
    agent = ClaimAgent(prompt_version="v1")

    results = []
    print("\n" + "=" * 70)
    print("PHASE 2 SMOKE TEST — A01 to A03")
    print("=" * 70)

    for a in answers:
        ans_id = a["answer_id"]
        brand_id = a["brand_id"]
        question = a["question"]
        ai_answer = a["ai_answer"]

        print(f"\n--- Answer {ans_id} [Brand: {brand_id}] ---")
        print(f"Question: {question}\n")
        print("Full ai_answer text:")
        print("-" * 50)
        print(ai_answer)
        print("-" * 50)

        res = agent.process_answer(
            answer_id=ans_id,
            brand_id=brand_id,
            question=question,
            ai_answer=ai_answer,
        )

        results.append(res)
        print(f"Status: {res['status']} | Latency: {res['latency_ms']} ms")
        print(f"Tool Calls Executed ({len(res['tool_calls'])} calls):")
        for tc in res["tool_calls"]:
            print(f"  Turn {tc['turn']}: {tc['tool_name']}({tc['args']})")

        print(f"\nPredictions ({len(res['claims'])} claims extracted):")
        for idx, c in enumerate(res["claims"], 1):
            print(f"  {idx}. [{c['start_char']}:{c['end_char']}] ({c['label']})")
            print(f"     Quote: {c['quote']}")
            print(f"     Fact IDs: {c['fact_ids']}")
            print(f"     Reason: {c['reason']}")

    # Evaluation against gold
    gold = load_development_gold()
    gold_subset = [g for g in gold if g["answer_id"] in {"A01", "A02", "A03"}]
    scores = score_predictions(gold_subset, results)

    print("\n" + "=" * 70)
    print("SMOKE TEST SCORER SUMMARY (A01-A03)")
    print("=" * 70)
    print(f"Gold Claims (G):             {scores['G']}")
    print(f"Accepted Predictions (P):    {scores['P']}")
    print(f"Exact Span Matches (M):      {scores['M']}")
    print(f"Correct Label Matches (C):   {scores['C']}")
    print(f"Extraction Precision:        {scores['extraction_precision']:.4f}")
    print(f"Extraction Recall:           {scores['extraction_recall']:.4f}")
    print(f"Extraction F1:               {scores['extraction_f1']:.4f}")
    print(f"Joint Precision:             {scores['joint_precision']:.4f}")
    print(f"Joint Recall:                {scores['joint_recall']:.4f}")
    print(f"Joint F1:                    {scores['joint_f1']:.4f}")
    print(f"Coverage (C/G):              {scores['coverage']:.4f}")
    print(f"Wrong-label count:           {scores['matched_wrong_label_count']}")
    if scores['matched_wrong_label_records']:
        print("Wrong-label details:")
        for r in scores['matched_wrong_label_records']:
            print(f"  - {r['claim_id']}: predicted '{r['predicted_label']}', expected '{r['expected_label']}'")
            print(f"    Quote: {r['quote']}")
            print(f"    Explanation: {r['explanation']}")


if __name__ == "__main__":
    main()
