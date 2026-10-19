"""
Sequential runner for Phase 3:
1. v2 on development (A01-A40)
2. v1 on check (A41-A50)
3. v2 on check (A41-A50)
Runs sequentially with checkpointing to respect rate limits and save reports to reports/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.run_eval import run_evaluation


def main():
    print("\n" + "=" * 70)
    print("STARTING RUN 1/3: v2 on DEVELOPMENT split (A01-A40)")
    print("=" * 70)
    rep_v2_dev = run_evaluation(
        split="development",
        version="v2",
        model_name="gemini-3.1-flash-lite",
        delay_between_answers=2.0,
        resume=True,
    )
    print(f"RUN 1 COMPLETE: {rep_v2_dev['run_id']}")

    print("\n" + "=" * 70)
    print("STARTING RUN 2/3: v1 on CHECK split (A41-A50)")
    print("=" * 70)
    rep_v1_check = run_evaluation(
        split="check",
        version="v1",
        model_name="gemini-3.1-flash-lite",
        delay_between_answers=2.0,
        resume=True,
    )
    print(f"RUN 2 COMPLETE: {rep_v1_check['run_id']}")

    print("\n" + "=" * 70)
    print("STARTING RUN 3/3: v2 on CHECK split (A41-A50)")
    print("=" * 70)
    rep_v2_check = run_evaluation(
        split="check",
        version="v2",
        model_name="gemini-3.1-flash-lite",
        delay_between_answers=2.0,
        resume=True,
    )
    print(f"RUN 3 COMPLETE: {rep_v2_check['run_id']}")
    print("\nALL 3 EVALUATION RUNS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
