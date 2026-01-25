"""
planner.py

Rule-based, explainable planner for an agentic causal / survival pipeline.

Features:
- Deterministic and auditable (no LLM required)
- Uses CSVLoadReport (schema from utils_csv)
- Produces a structured plan (LLM-ready schema)
- Supports explicit overrides for treatment/outcome/time/event:
  * If user specifies a column, planner trusts it (but validates existence)
  * Still records reasons + notes

Heuristics (conservative by default):
- Survival: requires time + event
- Causal: requires treatment + outcome
- Avoids guessing treatment/outcome when many candidates exist
- If survival structure exists, do NOT guess an "outcome" column
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from src.utils_csv import CSVLoadReport, ColumnProfile


# ============================================================
# Plan schema
# ============================================================

@dataclass
class Plan:
    chosen_capability: str
    confidence: float
    reasons: List[str]
    detected_columns: Dict[str, List[str]]
    notes: Optional[str] = None
    warnings: Optional[List[str]] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Public API
# ============================================================

def plan_from_csv_report(
    report: CSVLoadReport,
    user_request: Optional[str] = None,
    overrides: Optional[Dict[str, str]] = None,
) -> Plan:
    """
    Parameters
    ----------
    report : CSVLoadReport
        Output of load_csv_robust (structured).
    user_request : Optional[str]
        Natural language instruction (currently advisory; retained for LLM-ready interface).
    overrides : Optional[Dict[str, str]]
        Column overrides, e.g.:
          {"treatment": "A", "outcome": "Y", "time": "time", "event": "event"}

    Returns
    -------
    Plan
    """

    reasons: List[str] = []
    detected: Dict[str, List[str]] = {}
    plan_warnings: List[str] = []

    if not report.success or report.n_rows == 0 or report.n_cols == 0:
        return Plan(
            chosen_capability="abort",
            confidence=1.0,
            reasons=["CSV ingestion did not succeed or produced an empty table."],
            detected_columns={},
            notes="Downstream analysis skipped due to ingestion failure.",
            warnings=[],
        )

    cols = report.column_profiles
    colnames = [c.name for c in cols]
    colset = set(colnames)

    # ------------------------------------------------------------
    # Helper: apply and validate overrides
    # ------------------------------------------------------------
    ov = overrides or {}

    def take_override(role: str) -> Optional[str]:
        val = ov.get(role)
        if not val:
            return None
        if val not in colset:
            plan_warnings.append(f"Override for {role}='{val}' not found in columns; ignoring.")
            return None
        reasons.append(f"Using user override: {role}='{val}'.")
        return val

    # ------------------------------------------------------------
    # Detect survival structure first (time/event)
    # ------------------------------------------------------------
    time_cols = detect_time(cols)
    event_cols = detect_event(cols, require_time=bool(time_cols))

    # Apply time/event overrides (single column expected)
    time_override = take_override("time")
    event_override = take_override("event")
    if time_override:
        time_cols = [time_override]
    if event_override:
        event_cols = [event_override]

    # If survival structure exists, do NOT guess "outcome"
    survival_structure = bool(time_cols) and bool(event_cols)

    # ------------------------------------------------------------
    # Treatment / outcome detection (or overrides)
    # ------------------------------------------------------------
    treatment = take_override("treatment")
    outcome = take_override("outcome")

    if treatment is None:
        treatment = detect_treatment(cols)

    if outcome is None:
        if survival_structure:
            outcome = None
        else:
            outcome = detect_outcome(cols)

    # ------------------------------------------------------------
    # Populate detected columns
    # ------------------------------------------------------------
    if treatment:
        detected["treatment"] = [treatment]
    if outcome:
        detected["outcome"] = [outcome]
    if time_cols:
        detected["time"] = time_cols
    if event_cols:
        detected["event"] = event_cols

    # ------------------------------------------------------------
    # Decision rules
    # ------------------------------------------------------------
    if survival_structure:
        reasons.append("Detected time-to-event structure (time + event indicator).")
        if treatment:
            reasons.append("Treatment/exposure column is available; can adjust survival curves.")
            return Plan(
                chosen_capability="adjusted_curves",
                confidence=0.9 if (time_override or event_override or treatment in colset) else 0.85,
                reasons=reasons,
                detected_columns=detected,
                notes="Recommend survival analysis with treatment adjustment.",
                warnings=plan_warnings,
            )
        else:
            reasons.append("No explicit treatment/exposure column detected (or provided).")
            return Plan(
                chosen_capability="survival_descriptive",
                confidence=0.8 if (time_override or event_override) else 0.75,
                reasons=reasons,
                detected_columns=detected,
                notes="Recommend descriptive survival analysis (Kaplan–Meier / Cox without exposure).",
                warnings=plan_warnings,
            )

    # Non-survival: causal if treatment+outcome exist
    if treatment and outcome:
        reasons.append("Detected treatment and outcome suitable for causal analysis.")
        return Plan(
            chosen_capability="causal_models",
            confidence=0.85 if (treatment in colset and outcome in colset) else 0.8,
            reasons=reasons,
            detected_columns=detected,
            notes="Recommend causal effect estimation.",
            warnings=plan_warnings,
        )

    if outcome and not treatment:
        reasons.append("Detected outcome but no explicit treatment.")
        return Plan(
            chosen_capability="descriptive_analysis",
            confidence=0.65,
            reasons=reasons,
            detected_columns=detected,
            notes="Outcome detected without treatment; defaulting to descriptive analysis.",
            warnings=plan_warnings,
        )

    return Plan(
        chosen_capability="descriptive_only",
        confidence=0.5,
        reasons=["No clear causal or survival structure detected."],
        detected_columns=detected,
        notes="Planner could not confidently identify treatment/outcome/time/event roles.",
        warnings=plan_warnings,
    )


# ============================================================
# Strict role detection heuristics
# ============================================================

TREATMENT_NAMES = {
    "treatment", "treated", "treat", "trt", "tx", "exposure", "exposed", "a", "arm", "group"
}
OUTCOME_HINTS = ("y", "outcome", "response", "target", "label", "endpoint", "factual")
TIME_HINTS = ("time", "duration", "days", "follow", "fu", "surv")
EVENT_NAMES = {"event", "status", "censor", "censored", "death", "died", "failure"}


def detect_treatment(cols: List[ColumnProfile]) -> Optional[str]:
    # 1) explicit name match
    explicit = []
    for c in cols:
        if c.name.lower() in TREATMENT_NAMES:
            explicit.append(c.name)
    if explicit:
        for e in explicit:
            if e.lower() == "treatment":
                return e
        return explicit[0]

    # 2) fallback: ONE binary candidate ONLY if unique
    binary_candidates: List[str] = []
    for c in cols:
        if c.is_likely_id:
            continue
        if c.inferred_type == "boolean":
            binary_candidates.append(c.name)
        elif c.inferred_type == "integer" and 2 <= c.n_unique <= 3:
            binary_candidates.append(c.name)

    if len(binary_candidates) == 1:
        return binary_candidates[0]
    return None


def detect_outcome(cols: List[ColumnProfile]) -> Optional[str]:
    # Strong preference for y_factual if present
    for c in cols:
        if c.name.lower() == "y_factual":
            return c.name

    # explicit name match
    explicit = []
    for c in cols:
        lname = c.name.lower()
        if any(h in lname for h in OUTCOME_HINTS) and lname not in TREATMENT_NAMES:
            explicit.append(c.name)
    if explicit:
        return explicit[0]

    # fallback: ONE numeric candidate ONLY if unique (avoid guessing)
    numeric = []
    for c in cols:
        lname = c.name.lower()
        if c.is_likely_id:
            continue
        if lname in EVENT_NAMES:
            continue
        if any(h in lname for h in TIME_HINTS):
            continue
        if c.inferred_type in {"float", "integer"}:
            numeric.append(c.name)

    if len(numeric) == 1:
        return numeric[0]
    return None


def detect_time(cols: List[ColumnProfile]) -> List[str]:
    out = []
    for c in cols:
        lname = c.name.lower()
        if c.inferred_type == "datetime":
            out.append(c.name)
        elif any(h in lname for h in TIME_HINTS):
            out.append(c.name)
    return unique(out)


def detect_event(cols: List[ColumnProfile], require_time: bool = True) -> List[str]:
    if not require_time:
        return []
    out = []
    for c in cols:
        if c.name.lower() in EVENT_NAMES:
            out.append(c.name)
    return unique(out)


def unique(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
