# src/agent/graph.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Tuple

from src.agent.schemas_io import RunRequest, ToolResult

# Importing tools triggers registration via src/agent/tools/__init__.py
import src.agent.tools  # noqa: F401
from src.agent.tools.registry import get_tool


# -----------------------------
# Router helpers
# -----------------------------
def _router_fallback(req: RunRequest) -> Tuple[str, str, str]:
    """
    Rule-based fallback:
      - if time/event/group present -> survival_adjusted_curves
      - else -> causal_ate
    """
    if req.time and req.event and req.group:
        return "survival_adjusted_curves", "auto", "Auto: time/event/group detected."
    return "causal_ate", "auto", "Auto: defaulting to causal_ate."


def _try_llm_router(req: RunRequest) -> Optional[Tuple[str, str, str]]:
    """
    Use your repo's router_llm.llm_choose_capability if available.
    Returns (capability_id, selected_by, router_reason) or None.

    NOTE: This function must never raise; failures fall back to rule-based routing.
    """
    try:
        from src.agent import router_llm
    except Exception:
        return None

    llm_fn = getattr(router_llm, "llm_choose_capability", None)
    if not callable(llm_fn):
        return None

    # LLM needs a non-empty request string
    if not (req.request and isinstance(req.request, str) and req.request.strip()):
        return None

    try:
        obj = llm_fn(
            request=req.request,
            csv_columns=None,
            model=getattr(req, "llm_model", None),
        )
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    cap_id = (obj.get("capability_id") or obj.get("cap_id") or "").strip()
    reason = (obj.get("reason") or obj.get("router_reason") or "").strip()

    if not cap_id:
        return None

    return cap_id, "llm", (reason or "LLM selected capability.")


def _choose_capability(req: RunRequest) -> Tuple[str, str, str]:
    """
    Priority:
      1) forced by req.capability_id
      2) LLM router (if req.use_llm_router)
      3) rule-based fallback
    """
    if req.capability_id:
        return req.capability_id, "capability_id", "Forced by capability_id."

    if req.use_llm_router:
        llm_out = _try_llm_router(req)
        if llm_out is not None:
            cap_id, selected_by, router_reason = llm_out
            return cap_id, selected_by or "llm", router_reason or ""

    return _router_fallback(req)


# -----------------------------
# State / request normalization
# -----------------------------
def _coerce_req(obj: Any) -> RunRequest:
    """
    Accept:
      - RunRequest dataclass
      - dict with RunRequest fields (FastAPI payload or internal code)
      - other dataclass convertible to RunRequest
    """
    if isinstance(obj, RunRequest):
        return obj

    if is_dataclass(obj) and hasattr(obj, "__dict__"):
        return RunRequest(**asdict(obj))

    if isinstance(obj, dict):
        allowed = set(RunRequest.__dataclass_fields__.keys())
        clean = {k: v for k, v in obj.items() if k in allowed}
        return RunRequest(**clean)

    raise TypeError(f"Unsupported req type: {type(obj)}")


def _toolresult_to_dict(tr: Any) -> Dict[str, Any]:
    """
    Convert a ToolResult-like object to a JSON-friendly dict.

    Supports:
      - ToolResult dataclass
      - dict (already JSON-friendly)
      - any object with __dict__ (best-effort)
    """
    if tr is None:
        return {
            "status": "error",
            "selected_tool": "none",
            "stdout": "",
            "stderr": "Tool returned None",
            "exit_code": 2,
            "artifacts": {},
            "warnings": ["Tool returned None"],
        }

    if isinstance(tr, dict):
        return {
            "status": tr.get("status", "ok"),
            "selected_tool": tr.get("selected_tool") or tr.get("tool") or "none",
            "stdout": tr.get("stdout", ""),
            "stderr": tr.get("stderr", ""),
            "exit_code": tr.get("exit_code", 0 if tr.get("status", "ok") == "ok" else 2),
            "artifacts": tr.get("artifacts") or {},
            "warnings": tr.get("warnings") or [],
            "error": tr.get("error"),
        }

    if isinstance(tr, ToolResult):
        return {
            "status": tr.status,
            "selected_tool": tr.selected_tool,
            "stdout": tr.stdout,
            "stderr": tr.stderr,
            "exit_code": tr.exit_code,
            "artifacts": tr.artifacts or {},
            "warnings": tr.warnings or [],
        }

    d = getattr(tr, "__dict__", None)
    if isinstance(d, dict):
        return {
            "status": d.get("status", "ok"),
            "selected_tool": d.get("selected_tool") or d.get("tool") or "none",
            "stdout": d.get("stdout", ""),
            "stderr": d.get("stderr", ""),
            "exit_code": d.get("exit_code", 0 if d.get("status", "ok") == "ok" else 2),
            "artifacts": d.get("artifacts") or {},
            "warnings": d.get("warnings") or [],
            "error": d.get("error"),
        }

    return {
        "status": "ok",
        "selected_tool": "none",
        "stdout": str(tr),
        "stderr": "",
        "exit_code": 0,
        "artifacts": {},
        "warnings": [],
    }


def _normalize_tool_result(raw: Any, tool_name: str) -> Dict[str, Any]:
    """Normalize the raw tool output into the ToolResult dict shape we return."""
    trd = _toolresult_to_dict(raw)

    # Ensure selected_tool is set
    if not trd.get("selected_tool") or trd.get("selected_tool") == "none":
        trd["selected_tool"] = tool_name

    # Ensure artifacts is a dict
    if trd.get("artifacts") is None:
        trd["artifacts"] = {}
    if not isinstance(trd["artifacts"], dict):
        trd["artifacts"] = dict(trd["artifacts"])

    # Ensure warnings is a list
    if trd.get("warnings") is None:
        trd["warnings"] = []

    return trd


# -----------------------------
# Minimal graph object
# -----------------------------
class SimpleGraph:
    """
    Minimal graph compatible with existing usage:
      out = graph.invoke({"req": req})

    Guarantees:
      - Never assumes tool results are objects with attributes.
      - Always returns JSON-serializable dict output.
    """

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")

        if "req" not in state:
            raise KeyError("state must contain key 'req'")

        req = _coerce_req(state["req"])

        cap_id, selected_by, router_reason = _choose_capability(req)

        # Fetch tool from registry
        try:
            tool = get_tool(cap_id)
        except KeyError as e:
            artifacts = {
                "capability_id": cap_id,
                "selected_by": selected_by,
                "router_reason": router_reason,
            }
            tr = ToolResult(
                status="error",
                selected_tool="none",
                stdout="",
                stderr=str(e),
                exit_code=2,
                artifacts=artifacts,
                warnings=[str(e)],
            )
            tool_result = _toolresult_to_dict(tr)
            return {
                "status": tool_result["status"],
                "selected_tool": tool_result["selected_tool"],
                "stdout": tool_result["stdout"],
                "stderr": tool_result["stderr"],
                "artifacts": tool_result["artifacts"],
                "tool_result": tool_result,
            }

        ok, reason = tool.validate(req)
        if not ok:
            artifacts = {
                "capability_id": cap_id,
                "selected_by": selected_by,
                "router_reason": router_reason,
            }
            tr = ToolResult(
                status="error",
                selected_tool=tool.name,
                stdout="",
                stderr=reason,
                exit_code=2,
                artifacts=artifacts,
                warnings=[reason],
            )
            tool_result = _toolresult_to_dict(tr)
            return {
                "status": tool_result["status"],
                "selected_tool": tool_result["selected_tool"],
                "stdout": tool_result["stdout"],
                "stderr": tool_result["stderr"],
                "artifacts": tool_result["artifacts"],
                "tool_result": tool_result,
            }

        # Run tool
        raw_tr = tool.run(req)
        tool_result = _normalize_tool_result(raw_tr, tool.name)

        # Ensure router info is always present in artifacts
        artifacts = dict(tool_result.get("artifacts") or {})
        artifacts.update(
            {
                "capability_id": cap_id,
                "selected_by": selected_by,
                "router_reason": router_reason,
            }
        )
        tool_result["artifacts"] = artifacts

        return {
            "status": tool_result.get("status", "ok"),
            "selected_tool": tool_result.get("selected_tool", tool.name),
            "stdout": tool_result.get("stdout", ""),
            "stderr": tool_result.get("stderr", ""),
            "artifacts": artifacts,
            "tool_result": tool_result,
        }


# Singleton graph (importable by app.py)
graph = SimpleGraph()
