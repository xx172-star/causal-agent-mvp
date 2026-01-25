"""
scripts/run_from_plan.py

Run an analysis demo from a generated plan.json.

DISPATCH MAPPING (IMPORTANT):
- causal_models
    -> src/run_causalmodels_demo.py
       --csv
       --treatment
       --outcome
- adjusted_curves / survival_descriptive
    -> src/run_adjustedcurves_demo.py
       --csv
       --time
       --event
       --group        (mapped from plan.treatment)
       --covariates   (optional, not used yet)

NOTE:
- run_adjustedcurves_demo.py does NOT accept --out_dir
- run_adjustedcurves_demo.py uses --group instead of --treatment
"""

from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path
import argparse
import shlex
import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[RUN] {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write("[CMD] " + " ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()

        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        f.write(f"\n[EXIT_CODE] {proc.returncode}\n")
        return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo from plan.json")
    parser.add_argument("--plan", required=True, help="Path to plan.json produced by scripts/plan_csv.py")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = payload["plan"]
    csv_path = payload["csv"]

    capability = plan["chosen_capability"]
    detected = plan.get("detected_columns", {}) or {}

    log_path = Path("out") / "runs" / (safe_name(plan_path.name) + ".log")

    causal_demo = PROJECT_ROOT / "src" / "run_causalmodels_demo.py"
    adj_demo = PROJECT_ROOT / "src" / "run_adjustedcurves_demo.py"

    # ------------------------------------------------------------
    # causal_models
    # ------------------------------------------------------------
    if capability == "causal_models":
        treatment = (detected.get("treatment") or [None])[0]
        outcome = (detected.get("outcome") or [None])[0]

        if not treatment or not outcome:
            print("[FAIL] causal_models requires treatment and outcome.")
            print("detected_columns:", detected)
            sys.exit(2)

        cmd = [
            sys.executable,
            str(causal_demo),
            "--csv", csv_path,
            "--treatment", treatment,
            "--outcome", outcome,
        ]

        print("[INFO] Dispatch -> run_causalmodels_demo.py")
        print("[INFO] log:", log_path)
        sys.exit(run_cmd(cmd, log_path))

    # ------------------------------------------------------------
    # adjusted_curves / survival_descriptive
    # ------------------------------------------------------------
    if capability in ("adjusted_curves", "survival_descriptive"):
        time_col = (detected.get("time") or [None])[0]
        event_col = (detected.get("event") or [None])[0]
        group_col = (detected.get("treatment") or [None])[0]

        if not time_col or not event_col:
            print("[FAIL] survival analysis requires time and event.")
            print("detected_columns:", detected)
            sys.exit(2)

        cmd = [
            sys.executable,
            str(adj_demo),
            "--csv", csv_path,
            "--time", time_col,
            "--event", event_col,
        ]

        # Map treatment -> group (ONLY if present)
        if capability == "adjusted_curves":
            if not group_col:
                print("[FAIL] adjusted_curves requires treatment/group.")
                sys.exit(2)
            cmd += ["--group", group_col]

        print("[INFO] Dispatch -> run_adjustedcurves_demo.py")
        print("[INFO] log:", log_path)
        sys.exit(run_cmd(cmd, log_path))

    # ------------------------------------------------------------
    # Unsupported capability
    # ------------------------------------------------------------
    print(f"[WARN] Unsupported capability: {capability}")
    print("Plan file:", plan_path)
    sys.exit(3)


if __name__ == "__main__":
    main()
