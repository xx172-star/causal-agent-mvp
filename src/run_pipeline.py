# src/run_pipeline.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.planner import choose_capability, inspect_csv, load_capabilities
from src.utils_csv import load_csv_robust

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


def get_csv_columns(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path, nrows=5).columns.tolist()


def sanitize_covariates_for_survival(
    csv_path: Path,
    user_covariates: str,
    *,
    reserved: set[str] | None = None,
) -> tuple[str, list[str], list[str], list[str], list[str]]:
    """
    Returns:
      (covariates_csv_string, kept, dropped_missing, dropped_reserved, all_columns)
    """
    cols = get_csv_columns(csv_path)
    colset = set(cols)

    reserved = reserved or {"id", "time", "event", "group"}

    raw = [c.strip() for c in (user_covariates or "").split(",") if c.strip()]
    dropped_missing = [c for c in raw if c not in colset]

    present = [c for c in raw if c in colset]
    dropped_reserved = [c for c in present if c in reserved]
    kept = [c for c in present if c not in reserved]

    return ",".join(kept), kept, dropped_missing, dropped_reserved, cols


def run_script(script_path: Path, out_path: Path, extra_args: list[str] | None = None) -> dict[str, Any]:
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd += extra_args

    print(f"[RUN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)

    # Save stdout/stderr for reproducibility
    out_path.write_text(proc.stdout + "\n\n=== STDERR ===\n" + proc.stderr)
    print(f"[SAVED] {out_path}")

    # Also print a short preview to terminal
    preview = "\n".join(proc.stdout.splitlines()[:30])
    print("\n--- Preview (first 30 lines) ---")
    print(preview if preview else "(no stdout)")
    print("--- End Preview ---\n")

    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "saved_output_txt": str(out_path),
    }


def infer_outcome_type_from_csv(csv_path: Path, outcome_col: str | None) -> str:
    """
    Keep your robust inference (used when planner needs outcome typing).
    """
    df, _ = load_csv_robust(
        csv_path,
        parse_dates=True,
        verbose=False,
    )

    cols_lower = set(str(c).strip().lower() for c in df.columns)

    if ("time" in cols_lower) and ("event" in cols_lower):
        return "survival"

    if outcome_col is None:
        return "continuous"

    if outcome_col not in df.columns:
        raise ValueError(
            f"Outcome column '{outcome_col}' not found in CSV columns: {list(df.columns)[:20]} ..."
        )

    y = df[outcome_col].dropna()
    uniq = set(pd.unique(y))

    if y.dtype == "object":
        lower = set(str(v).strip().lower() for v in uniq)
        if lower.issubset({"0", "1", "true", "false", "t", "f", "yes", "no"}):
            return "binary"

    try:
        nums = set(float(v) for v in uniq)
        if nums.issubset({0.0, 1.0}):
            return "binary"
    except Exception:
        pass

    return "continuous"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "route",
        choices=["auto", "survival", "continuous", "binary"],
        help="auto uses planner; others override planner",
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV dataset")

    # only needed for continuous/binary
    parser.add_argument("--outcome", type=str, default=None, help="Outcome column name")
    parser.add_argument("--treatment", type=str, default=None, help="Treatment column name")

    # optional for all
    parser.add_argument("--covariates", type=str, default="", help="Comma-separated covariates")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        raise SystemExit(2)

    # Load capability list (JSON is the "logic of what it can do")
    caps_path = ROOT / "pipeline_capabilities.json"
    caps = load_capabilities(caps_path)

    # Inspect CSV for planning
    csv_info = inspect_csv(csv_path)

    # planner chooses an action (capability) with reasons
    chosen, reasons = choose_capability(
        caps,
        csv_info,
        user_route=args.route,
        outcome=args.outcome,
    )

    print(f"[PLAN] chosen capability: {chosen['name']}")
    for r in reasons:
        print(f"[PLAN] {r}")

    # Prepare script + output path by chosen name
    script_path = ROOT / chosen["script"]
    if not script_path.exists():
        print(f"Missing script: {script_path}")
        raise SystemExit(2)

    if chosen["name"] == "survival_iptw_km":
        out_txt = OUTDIR / "pipeline_survival.txt"

        covs, kept, dropped_missing, dropped_reserved, cols = sanitize_covariates_for_survival(
            csv_path,
            args.covariates,
            reserved={"id", "time", "event", "group"},
        )

        if args.covariates and dropped_missing:
            print(f"[WARN] Dropping missing covariates for survival: {dropped_missing}")
        if args.covariates and dropped_reserved:
            print(f"[WARN] Dropping reserved columns from covariates: {dropped_reserved}")
        if args.covariates and not kept:
            print(
                "[WARN] None of the provided covariates are usable for the propensity model. "
                "Falling back to intercept-only model (group ~ 1). "
                f"CSV columns are: {cols}"
            )

        extra_args = ["--csv", str(csv_path), "--covariates", covs]
        run_res = run_script(script_path, out_txt, extra_args=extra_args)

    else:
        # continuous or binary causal route requires outcome + treatment
        if args.outcome is None or args.treatment is None:
            print("For continuous/binary you must provide --outcome and --treatment.")
            raise SystemExit(2)

        # choose output filename by inferred/forced type for readability
        inferred = infer_outcome_type_from_csv(csv_path, args.outcome)
        if chosen["name"] == "causal_binary_dr" or inferred == "binary":
            out_txt = OUTDIR / "pipeline_binary.txt"
        else:
            out_txt = OUTDIR / "pipeline_continuous.txt"

        extra_args = [
            "--csv",
            str(csv_path),
            "--outcome",
            args.outcome,
            "--treatment",
            args.treatment,
            "--covariates",
            args.covariates,
        ]
        run_res = run_script(script_path, out_txt, extra_args=extra_args)

    # Build structured JSON report (planner + diagnostics)
    stderr_lower = (run_res.get("stderr") or "").lower()
    warnings: list[str] = []
    if "perfect separation" in stderr_lower or "separation" in stderr_lower:
        warnings.append("perfect_separation_hint")
    if "all weights" in stderr_lower:
        warnings.append("degenerate_weights_hint")
    if "warning" in stderr_lower:
        warnings.append("stderr_contains_warning")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "csv": {
            "path": str(csv_path),
            "columns": csv_info.get("columns", []),
            "dtypes_preview": csv_info.get("dtypes", {}),
            "missing_rate_preview": csv_info.get("missing_rate", {}),
        },
        "planner": {
            "capabilities_file": str(caps_path),
            "chosen_capability": chosen,
            "reasons": reasons,
        },
        "run": {
            "cmd": run_res.get("cmd", []),
            "returncode": run_res.get("returncode", None),
            "saved_output_txt": run_res.get("saved_output_txt", ""),
        },
        "warnings": warnings,
    }

    # save JSON next to txt
    out_json = Path(run_res["saved_output_txt"]).with_suffix(".json")
    out_json.write_text(json.dumps(report, indent=2))
    print(f"[SAVED] {out_json}")

    raise SystemExit(int(run_res.get("returncode", 1)))


if __name__ == "__main__":
    main()
