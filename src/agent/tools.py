from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../src/agent -> repo root


def _run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    return p.returncode, p.stdout, p.stderr


def _resolve_csv_path(csv: str) -> Path:
    p = Path(csv)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def _ensure_out_dir(out_dir: str) -> Path:
    out_dir_path = (REPO_ROOT / out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    return out_dir_path


def _infer_covariates(df: pd.DataFrame, *, outcome: str, max_covariates: int) -> List[str]:
    cols = [c for c in df.columns if c != outcome]
    # Prefer numeric columns
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    chosen = num_cols[:max_covariates]
    if not chosen:
        chosen = cols[:max_covariates]
    return chosen


def run_causalmodels_tool(
    csv: str,
    treatment: str,
    outcome: str,
    covariates: Optional[List[str]],
    max_covariates: int,
    out_dir: str,
) -> Dict[str, Any]:
    out_dir_path = _ensure_out_dir(out_dir)
    out_json = out_dir_path / "causalmodels.summary.json"

    cmd = [
        "python",
        "src/run_causalmodels_demo.py",
        "--csv",
        csv,
        "--treatment",
        treatment,
        "--outcome",
        outcome,
        "--max_covariates",
        str(max_covariates),
        "--out_json",
        str(out_json),
    ]
    if covariates:
        cmd += ["--covariates", ",".join(covariates)]

    code, stdout, stderr = _run_cmd(cmd, cwd=REPO_ROOT)

    artifacts: Dict[str, Any] = {"summary_json": str(out_json)}
    summary_obj = None
    if out_json.exists():
        try:
            summary_obj = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception:
            summary_obj = None

    return {
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "artifacts": artifacts,
        "summary": summary_obj,
    }


def run_adjustedcurves_tool(
    csv: str,
    group: str,
    time: str,
    event: str,
    covariates: Optional[List[str]],
    out_dir: str,
) -> Dict[str, Any]:
    """
    Run IPTW-adjusted Kaplan–Meier curves via the demo script.

    Minimal artifact export:
      - Writes a compact JSON file containing run metadata + stdout/stderr.
      - Returns the path as artifacts["summary_json"] to match the ATE demo.
    """
    out_dir_path = _ensure_out_dir(out_dir)
    out_json = out_dir_path / "adjustedcurves.summary.json"

    cmd = [
        "python",
        "src/run_adjustedcurves_demo.py",
        "--csv",
        csv,
        "--group",
        group,
        "--time",
        time,
        "--event",
        event,
    ]
    if covariates:
        cmd += ["--covariates", ",".join(covariates)]

    code, stdout, stderr = _run_cmd(cmd, cwd=REPO_ROOT)

    artifacts: Dict[str, Any] = {"summary_json": str(out_json)}

    summary_obj: Optional[Dict[str, Any]] = None
    try:
        summary_obj = {
            "capability_id": "survival_adjusted_curves",
            "tool": "adjustedcurves",
            "method": "iptw_km",
            "inputs": {
                "csv": csv,
                "group": group,
                "time": time,
                "event": event,
                "covariates": covariates or [],
            },
            "exit_code": int(code),
            "stdout": stdout,
            "stderr": stderr,
        }
        out_json.write_text(json.dumps(summary_obj, indent=2), encoding="utf-8")
    except Exception:
        # Fail open: don't block the run if writing the artifact fails.
        summary_obj = None

    return {
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "artifacts": artifacts,
        "summary": summary_obj,
    }


def run_logistic_regression_assoc(
    csv: str,
    outcome: str,
    covariates: Optional[List[str]],
    max_covariates: int,
    out_dir: str,
) -> Dict[str, Any]:
    """
    Simple logistic regression (association / prediction baseline).
    Outputs a JSON summary artifact.
    """
    df = pd.read_csv(_resolve_csv_path(csv))
    if covariates is None:
        covariates = _infer_covariates(df, outcome=outcome, max_covariates=max_covariates)

    X = df[covariates].copy()
    y = df[outcome].copy()

    # Basic preprocessing: drop NA rows
    data = pd.concat([X, y], axis=1).dropna()
    X = data[covariates]
    y = data[outcome]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= 0.5).astype(int)

    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    acc = accuracy_score(y, yhat)

    out_dir_path = _ensure_out_dir(out_dir)
    out_json = out_dir_path / "logistic_assoc.summary.json"

    summary = {
        "capability_id": "logistic_assoc",
        "outcome": outcome,
        "covariates": covariates,
        "n": int(len(y)),
        "metrics": {"auc": float(auc), "accuracy": float(acc)},
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "exit_code": 0,
        "stdout": f"Logistic regression fit ok. AUC={auc:.4f}, ACC={acc:.4f}\n",
        "stderr": "",
        "artifacts": {"summary_json": str(out_json)},
        "summary": summary,
    }


def run_linear_regression_assoc(
    csv: str,
    outcome: str,
    covariates: Optional[List[str]],
    max_covariates: int,
    out_dir: str,
) -> Dict[str, Any]:
    """
    Simple linear regression (association / prediction baseline).
    Outputs a JSON summary artifact.
    """
    df = pd.read_csv(_resolve_csv_path(csv))
    if covariates is None:
        covariates = _infer_covariates(df, outcome=outcome, max_covariates=max_covariates)

    X = df[covariates].copy()
    y = df[outcome].copy()

    data = pd.concat([X, y], axis=1).dropna()
    X = data[covariates]
    y = data[outcome]

    model = LinearRegression()
    model.fit(X, y)
    yhat = model.predict(X)

    r2 = r2_score(y, yhat)

    out_dir_path = _ensure_out_dir(out_dir)
    out_json = out_dir_path / "linear_assoc.summary.json"

    summary = {
        "capability_id": "linear_assoc",
        "outcome": outcome,
        "covariates": covariates,
        "n": int(len(y)),
        "metrics": {"r2": float(r2)},
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "exit_code": 0,
        "stdout": f"Linear regression fit ok. R2={r2:.4f}\n",
        "stderr": "",
        "artifacts": {"summary_json": str(out_json)},
        "summary": summary,
    }
