"""
Script to load all existing JSON evaluation reports from reports/ into PostgreSQL
using api.db.save_run.
Does not make any LLM/model calls.
"""

import glob
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.db import save_run, get_run_details, init_db, get_db_connection

load_dotenv()


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set in environment or .env.")
        sys.exit(1)

    print("Attempting to connect to database and apply schema...")
    try:
        init_db(db_url)
        print("Schema successfully applied.")
    except Exception as e:
        print(f"Failed to connect or apply schema: {e}")
        sys.exit(1)

    report_files = sorted(glob.glob("reports/eval_*.json"))
    print(f"Found {len(report_files)} report files to import.")

    for path in report_files:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)

        run_id = report["run_id"]
        split = report["split"]
        version = report["agent_version"]

        print(f"\nImporting {path} (run_id={run_id}, split={split}, version={version})...")
        save_run(report, db_url=db_url)
        print(f"Successfully saved run {run_id}.")

    # Verify row counts per run
    print("\n" + "=" * 60)
    print("DATABASE ROW COUNTS PER RUN")
    print("=" * 60)
    with get_db_connection(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.run_id, r.split, r.agent_version,
                       COUNT(DISTINCT a.id) as answer_count,
                       COUNT(DISTINCT c.id) as claim_count
                FROM runs r
                LEFT JOIN answer_results a ON r.run_id = a.run_id
                LEFT JOIN claim_results c ON r.run_id = c.run_id
                GROUP BY r.run_id, r.split, r.agent_version
                ORDER BY r.split DESC, r.agent_version ASC;
            """)
            rows = cur.fetchall()
            for row in rows:
                print(f"Run: {row['run_id'][:8]} | Split: {row['split']:<11} | Version: {row['agent_version']} | Answers: {row['answer_count']:<2} | Claims: {row['claim_count']}")

    # Demonstrate joined retrieval
    dev_run_id = rows[0]["run_id"] if rows else None
    if dev_run_id:
        print("\n" + "=" * 60)
        print(f"JOINED RETRIEVAL SAMPLE FOR RUN {dev_run_id}")
        print("=" * 60)
        details = get_run_details(str(dev_run_id), db_url=db_url)
        # Trim output for display
        sample_output = {
            "run_id": details["run"]["run_id"],
            "split": details["run"]["split"],
            "agent_version": details["run"]["agent_version"],
            "metrics": details["run"]["metrics"],
            "answers_sample": details["answers"][:2],
            "claims_sample": details["claims"][:2],
            "total_answers": len(details["answers"]),
            "total_claims": len(details["claims"]),
        }
        print(json.dumps(sample_output, indent=2, default=str))


if __name__ == "__main__":
    main()
