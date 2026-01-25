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
    use = num_cols if num_cols else cols
    return use[:max_covariates]


def _prep_design_matrix(
    df: pd.DataFrame,
    *,
    outcome: str,
    covariates: Optional[List[str]],
    max_covariates: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' not found in CSV.")

    use_covs = covariates if covariates else _infer_covariates(df, outcome=outcome, max_covariates=max_covariates)
    use_covs = use_covs[:max_covariates]

    missing_covs = [c for c in use_covs if c not in df.columns]
    if missing_covs:
        raise ValueError(f"Covariate columns not found in CSV: {missing_covs}")

    X_df = df[use_covs].copy()
    y_ser = df[outcome].copy()

    # One-hot encode categoricals (minimal, robust)
    X_df = pd.get_dummies(X_df, drop_first=True)

    # Drop rows with missing values
    joined = pd.concat([y_ser, X_df], axis=1).dropna()
    y = joined[outcome].to_numpy()
    X = joined.drop(columns=[outcome]).to_numpy()

    used_feature_names = list(joined.drop(columns=[outcome]).columns)
    n = int(joined.shape[0])
    return X, y, used_feature_names, n


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
) -> Dict[str, Any]:
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

    return {
        "exit_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "artifacts": {},
        "summary": None,
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
    Outputs a JSON summary with coefficients + basic metrics.
    """
    out_dir_path = _ensure_out_dir(out_dir)
    out_json = out_dir_path / "logistic_regression.summary.json"

    try:
        df = pd.read_csv(_resolve_csv_path(csv))
        X, y, feature_names, n = _prep_design_matrix(
            df, outcome=outcome, covariates=covariates, max_covariates=max_covariates
        )

        # Ensure binary 0/1
        y_unique = sorted(pd.unique(y))
        if len(y_unique) != 2:
            raise ValueError(f"Logistic regression requires a binary outcome; got values={y_unique}")

        # Map to 0/1 if needed
        y01 = pd.Series(y).astype("category").cat.codes.to_numpy()

        model = LogisticRegression(max_iter=2000, solver="liblinear")
        model.fit(X, y01)

        p_hat = model.predict_proba(X)[:, 1]
        y_pred = (p_hat >= 0.5).astype(int)

        acc = float(accuracy_score(y01, y_pred))
        # AUC can fail if only one class after dropna (rare but possible)
        try:
            auc = float(roc_auc_score(y01, p_hat))
        except Exception:
            auc = None

        coef = model.coef_.reshape(-1)
        coef_table = [
            {"feature": feature_names[i], "coef": float(coef[i])} for i in range(len(feature_names))
        ]

        summary = {
            "model": "logistic_regression",
            "n": n,
            "outcome": outcome,
            "features": feature_names,
            "metrics": {"accuracy": acc, "auc": auc},
            "coefficients": coef_table,
            "intercept": float(model.intercept_[0]),
        }

        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "exit_code": 0,
            "stdout": f"[OK] Wrote {out_json}",
            "stderr": "",
            "artifacts": {"summary_json": str(out_json)},
            "summary": summary,
        }

    except Exception as e:
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": str(e),
            "artifacts": {"summary_json": str(out_json)},
            "summary": None,
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
    Outputs a JSON summary with coefficients + R^2.
    """
    out_dir_path = _ensure_out_dir(out_dir)
    out_json = out_dir_path / "linear_regression.summary.json"

    try:
        df = pd.read_csv(_resolve_csv_path(csv))
        X, y, feature_names, n = _prep_design_matrix(
            df, outcome=outcome, covariates=covariates, max_covariates=max_covariates
        )

        model = LinearRegression()
        model.fit(X, y)

        y_hat = model.predict(X)
        r2 = float(r2_score(y, y_hat))

        coef = model.coef_.reshape(-1)
        coef_table = [
            {"feature": feature_names[i], "coef": float(coef[i])} for i in range(len(feature_names))
        ]

        summary = {
            "model": "linear_regression",
            "n": n,
            "outcome": outcome,
            "features": feature_names,
            "metrics": {"r2": r2},
            "coefficients": coef_table,
            "intercept": float(model.intercept_),
        }

        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "exit_code": 0,
            "stdout": f"[OK] Wrote {out_json}",
            "stderr": "",
            "artifacts": {"summary_json": str(out_json)},
            "summary": summary,
        }

    except Exception as e:
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": str(e),
            "artifacts": {"summary_json": str(out_json)},
            "summary": None,
        }
