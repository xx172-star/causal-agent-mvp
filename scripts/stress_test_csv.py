"""
scripts/stress_test_csv.py

Batch stress test for robust CSV ingestion.

Outputs:
- out/reports/<file>.report.json   (full report per csv)
- out/summary.csv                  (compact table for quick review)
- out/summary.json                 (same as JSON)

Usage:
  python scripts/stress_test_csv.py --data_dir data --out_dir out

Notes:
- This version matches the CURRENT utils_csv.load_csv_robust signature (no required_columns).
- It also ensures the project root is on sys.path so `from src.utils_csv ...` works.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src...` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import traceback
import datetime
from typing import Any, Dict, List

import pandas as pd

# Import your robust loader
from src.utils_csv import load_csv_robust


def safe_filename(name: str) -> str:
    """Make a filesystem-safe filename."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test robust CSV ingestion.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing CSV files.")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory.")
    parser.add_argument("--no_parse_dates", action="store_true", help="Disable datetime parsing.")
    parser.add_argument("--max_files", type=int, default=0, help="If >0, limit number of CSV files processed.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted([p for p in data_dir.rglob("*.csv") if p.is_file()])
    if args.max_files and args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    if not csv_files:
        print(f"[WARN] No CSV files found under: {data_dir.resolve()}")
        return

    rows: List[Dict[str, Any]] = []
    started = datetime.datetime.now().isoformat(timespec="seconds")

    print(f"[INFO] Found {len(csv_files)} CSV file(s). Starting stress test...")
    print(f"[INFO] data_dir={data_dir.resolve()}")
    print(f"[INFO] out_dir={out_dir.resolve()}")

    for i, path in enumerate(csv_files, start=1):
        rel = str(path.relative_to(data_dir))
        print(f"\n[{i}/{len(csv_files)}] Testing: {rel}")

        entry: Dict[str, Any] = {
            "file": rel,
            "abs_path": str(path.resolve()),
            "success": False,
            "n_rows": None,
            "n_cols": None,
            "used_encoding": None,
            "used_sep": None,
            "n_warnings": None,
            "n_errors": None,
            "parsed_datetime_cols": None,
            "report_path": None,
            "exception": None,
        }

        try:
            df, report = load_csv_robust(
                path,
                parse_dates=(not args.no_parse_dates),
                verbose_warnings=False,  # avoid console spam in batch mode
            )

            report_dict = report.to_json_dict()
            report_name = safe_filename(rel.replace("/", "__")) + ".report.json"
            report_path = reports_dir / report_name

            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2, default=str)

            entry.update({
                "success": bool(getattr(report, "success", False)),
                "n_rows": int(getattr(report, "n_rows", 0)),
                "n_cols": int(getattr(report, "n_cols", 0)),
                "used_encoding": getattr(report, "used_encoding", None),
                "used_sep": getattr(report, "used_sep", None),
                "n_warnings": len(getattr(report, "warnings", []) or []),
                "n_errors": len(getattr(report, "errors", []) or []),
                "parsed_datetime_cols": ",".join(getattr(report, "parsed_datetime_cols", []) or []),
                "report_path": str(report_path.relative_to(out_dir)),
            })

            # Optional quick surface: top warnings
            top_warns = (getattr(report, "warnings", []) or [])[:3]
            if top_warns:
                print("[INFO] Top warnings:")
                for w in top_warns:
                    print(f"  - {w}")

            status = "OK" if entry["success"] else "FAIL"
            print(f"[{status}] rows={entry['n_rows']} cols={entry['n_cols']} warnings={entry['n_warnings']}")

        except Exception as e:
            entry["success"] = False
            entry["exception"] = f"{type(e).__name__}: {str(e)}"
            print(f"[FAIL] {entry['exception']}")

            tb_name = safe_filename(rel.replace("/", "__")) + ".traceback.txt"
            tb_path = reports_dir / tb_name
            tb_path.write_text(traceback.format_exc(), encoding="utf-8")
            entry["report_path"] = str(tb_path.relative_to(out_dir))

        rows.append(entry)

    # Summary outputs
    df_sum = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "summary.csv"
    df_sum.to_csv(summary_csv, index=False)

    summary_json = out_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "started": started,
                "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                "n_files": len(csv_files),
                "n_success": int(df_sum["success"].sum()) if "success" in df_sum else 0,
                "results": rows,
            },
            f,
            indent=2,
            default=str,
        )

    # Print quick rollup
    n_success = int(df_sum["success"].sum())
    print("\n==============================")
    print("[DONE] Stress test complete.")
    print(f"Success: {n_success}/{len(csv_files)}")
    print(f"Summary: {summary_csv.resolve()}")
    print(f"Reports: {reports_dir.resolve()}")
    print("==============================\n")

    # Optional: show failures
    fails = df_sum[df_sum["success"] == False]
    if len(fails) > 0:
        print("[FAILURES]")
        for _, r in fails.iterrows():
            print(f"- {r['file']} -> {r.get('exception')} ({r.get('report_path')})")


if __name__ == "__main__":
    main()
