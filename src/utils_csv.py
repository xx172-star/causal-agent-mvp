"""
utils_csv.py

Robust CSV loading + schema profiling for agentic pipelines.

Design goals:
- Try very hard to read real-world CSV-like files
- Never crash on bad input
- Fail safely with structured diagnostics
- Explicitly log all fallbacks and risks

This version includes:
1) Multiple encodings × multiple delimiters for standard CSV parsing
2) A final fallback for whitespace-delimited data
3) A recovery step: if whitespace fallback yields a single column containing commas,
   parse that column as comma-separated data (common in "CSV-ish" files that break read_csv).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Sequence
from pathlib import Path
import warnings
import csv
import io

import pandas as pd


# ============================================================
# Report structures
# ============================================================

@dataclass
class ColumnProfile:
    name: str
    inferred_type: str
    pandas_dtype: str
    missing_rate: float
    n_unique: int
    is_likely_id: bool
    is_likely_categorical: bool
    example_values: List[str]


@dataclass
class CSVLoadReport:
    path: str
    success: bool
    n_rows: int
    n_cols: int

    used_encoding: Optional[str]
    used_sep: Optional[str]

    warnings: List[str]
    errors: List[str]

    parsed_datetime_cols: List[str]
    column_profiles: List[ColumnProfile]

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Public API
# ============================================================

DEFAULT_NA_VALUES = [
    "", "NA", "N/A", "na", "n/a", "NaN", "nan", "NULL", "null",
    ".", "?", "#N/A", "None", "none"
]


def load_csv_robust(
    path: str | Path,
    na_values: Optional[Sequence[str]] = None,
    keep_default_na: bool = True,
    parse_dates: bool = True,
    verbose_warnings: bool = False,
) -> Tuple[pd.DataFrame, CSVLoadReport]:
    """
    Robust CSV reader with explicit fallbacks and structured reporting.
    """
    p = Path(path)
    warns: List[str] = []
    errs: List[str] = []

    used_encoding: Optional[str] = None
    used_sep: Optional[str] = None

    na_vals = list(dict.fromkeys((list(na_values) if na_values else []) + DEFAULT_NA_VALUES))

    # ------------------------------------------------------------
    # Attempt 1: standard CSV parsing (multiple encodings × seps)
    # ------------------------------------------------------------
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    seps: List[Optional[str]] = [None, ",", "\t", ";", "|"]

    df: Optional[pd.DataFrame] = None
    read_attempts: List[str] = []

    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    p,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    na_values=na_vals,
                    keep_default_na=keep_default_na,
                    skipinitialspace=True,
                    low_memory=False,
                    on_bad_lines="warn",
                )
                used_encoding = enc
                used_sep = sep
                break
            except Exception as e:
                read_attempts.append(
                    f"FAIL encoding={enc}, sep={repr(sep)}: {type(e).__name__}"
                )
        if df is not None:
            break

    # ------------------------------------------------------------
    # Attempt 2 (FINAL FALLBACK): whitespace-delimited
    # ------------------------------------------------------------
    did_whitespace_fallback = False
    if df is None:
        try:
            df = pd.read_csv(
                p,
                sep=r"\s+",
                engine="python",
                header=None,
                na_values=na_vals,
                keep_default_na=keep_default_na,
            )
            used_encoding = "unknown"
            used_sep = "whitespace"
            did_whitespace_fallback = True
            warns.append(
                "All standard CSV parsing attempts failed. "
                "Falling back to whitespace-delimited parsing (no header assumed)."
            )
        except Exception as e:
            errs.append(
                f"All parsing attempts failed, including whitespace fallback: {type(e).__name__}: {e}"
            )

    # ------------------------------------------------------------
    # Recovery: if whitespace fallback produced 1 column with commas,
    # treat it as a comma-separated CSV stored in a single column.
    # ------------------------------------------------------------
    if df is not None and did_whitespace_fallback and df.shape[1] == 1:
        # Check first few non-empty rows for commas
        sample = df.iloc[:10, 0].dropna().astype(str).tolist()
        comma_hits = sum(1 for s in sample if "," in s)
        if comma_hits >= 3:  # strong signal it's CSV-in-one-column
            try:
                text = "\n".join(df[0].astype(str).tolist())
                recovered = pd.read_csv(
                    io.StringIO(text),
                    sep=",",
                    engine="python",
                    na_values=na_vals,
                    keep_default_na=keep_default_na,
                )
                df = recovered
                used_sep = "comma_recovered_from_single_column"
                warns.append(
                    "Whitespace fallback yielded a single column containing commas; "
                    "recovered by parsing that column as comma-separated data."
                )
            except Exception as e:
                errs.append(
                    f"Single-column comma recovery failed: {type(e).__name__}: {e}"
                )

    # ------------------------------------------------------------
    # If still failed → safe failure
    # ------------------------------------------------------------
    if df is None:
        warns.append(f"Read attempts: {read_attempts}")
        if verbose_warnings:
            for w in warns:
                warnings.warn(w)

        report = CSVLoadReport(
            path=str(p),
            success=False,
            n_rows=0,
            n_cols=0,
            used_encoding=None,
            used_sep=None,
            warnings=warns,
            errors=errs,
            parsed_datetime_cols=[],
            column_profiles=[],
        )
        return pd.DataFrame(), report

    # ------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------
    warns.append(f"Read attempts: {read_attempts}")

    # Strip whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]

    # Infer datetimes (remove deprecated infer_datetime_format)
    parsed_dt_cols: List[str] = []
    if parse_dates:
        for c in df.columns:
            if df[c].dtype == object:
                parsed = pd.to_datetime(df[c], errors="coerce")
                success_rate = float(parsed.notna().mean()) if len(df) else 0.0
                if success_rate >= 0.8:
                    df[c] = parsed
                    parsed_dt_cols.append(c)

    # ------------------------------------------------------------
    # Column profiling
    # ------------------------------------------------------------
    profiles: List[ColumnProfile] = []
    n = len(df)

    for c in df.columns:
        ser = df[c]
        inferred = infer_type(ser)
        n_unique = int(ser.nunique(dropna=True))
        missing_rate = float(ser.isna().mean()) if n > 0 else 0.0

        is_likely_id = n_unique > 0.9 * max(n, 1)
        is_likely_categorical = inferred in ("string", "boolean") and n_unique <= 50

        examples = (
            ser.dropna().astype(str).unique()[:5].tolist()
            if ser.notna().any()
            else []
        )

        profiles.append(
            ColumnProfile(
                name=str(c),
                inferred_type=inferred,
                pandas_dtype=str(ser.dtype),
                missing_rate=round(missing_rate, 4),
                n_unique=n_unique,
                is_likely_id=is_likely_id,
                is_likely_categorical=is_likely_categorical,
                example_values=examples,
            )
        )

    if verbose_warnings:
        for w in warns:
            warnings.warn(w)

    report = CSVLoadReport(
        path=str(p),
        success=True,
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        used_encoding=used_encoding,
        used_sep=used_sep,
        warnings=warns,
        errors=errs,
        parsed_datetime_cols=parsed_dt_cols,
        column_profiles=profiles,
    )
    return df, report


# ============================================================
# Helpers
# ============================================================

def infer_type(ser: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(ser):
        return "datetime"
    if pd.api.types.is_bool_dtype(ser):
        return "boolean"
    if pd.api.types.is_integer_dtype(ser):
        return "integer"
    if pd.api.types.is_float_dtype(ser):
        return "float"
    if pd.api.types.is_string_dtype(ser) or ser.dtype == object:
        return "string"
    return "other"
