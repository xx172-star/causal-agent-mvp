"""
scripts/plan_csv.py

Generate an explainable plan.json for a CSV file:
CSV -> robust load report -> planner -> plan.json

Usage examples:
  python scripts/plan_csv.py --csv "data/ihdp_data.csv"
  python scripts/plan_csv.py --csv "data/PBC_agent01.csv"
  python scripts/plan_csv.py --csv "data/PBC_agent01.csv" --treatment trt01
  python scripts/plan_csv.py --csv "data/PBC_agent01.csv" --time time --event event --treatment trt01

Outputs:
  out/plans/<safe_name>.plan.json
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
import argparse

# Ensure project root is on sys.path so `import src...` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils_csv import load_csv_robust
from src.planner import plan_from_csv_report


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an explainable plan.json for a CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV file.")
    parser.add_argument("--out_dir", default="out", help="Output directory root (default: out).")

    # Optional overrides
    parser.add_argument("--treatment", default="", help="Override: treatment column name.")
    parser.add_argument("--outcome", default="", help="Override: outcome column name.")
    parser.add_argument("--time", default="", help="Override: time column name (survival).")
    parser.add_argument("--event", default="", help="Override: event column name (survival).")

    parser.add_argument("--no_parse_dates", action="store_true", help="Disable datetime parsing.")
    parser.add_argument("--user_request", default="", help="Optional user instruction text (stored for future LLM planner).")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df, rep = load_csv_robust(csv_path, parse_dates=(not args.no_parse_dates))

    overrides = {}
    if args.treatment:
        overrides["treatment"] = args.treatment
    if args.outcome:
        overrides["outcome"] = args.outcome
    if args.time:
        overrides["time"] = args.time
    if args.event:
        overrides["event"] = args.event

    plan = plan_from_csv_report(
        rep,
        user_request=(args.user_request or None),
        overrides=(overrides or None),
    )

    out_root = Path(args.out_dir)
    plans_dir = out_root / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    out_path = plans_dir / (safe_name(csv_path.name) + ".plan.json")
    payload = {
        "csv": str(csv_path),
        "plan": plan.to_json(),
        "ingestion_summary": {
            "success": rep.success,
            "n_rows": rep.n_rows,
            "n_cols": rep.n_cols,
            "used_sep": rep.used_sep,
            "used_encoding": rep.used_encoding,
            "n_warnings": len(rep.warnings or []),
            "n_errors": len(rep.errors or []),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("[OK] wrote:", out_path)
    print(json.dumps(plan.to_json(), indent=2))


if __name__ == "__main__":
    main()
