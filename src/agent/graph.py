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


def _toolresult_to_dict(tr: ToolResult) -> Dict[str, Any]:
    return {
        "status": tr.status,
        "selected_tool": tr.selected_tool,
        "stdout": tr.stdout,
        "stderr": tr.stderr,
        "exit_code": tr.exit_code,
        "artifacts": tr.artifacts or {},
        "warnings": tr.warnings or [],
    }


# -----------------------------
# Minimal graph object
# -----------------------------
class SimpleGraph:
    """
    Minimal graph compatible with existing usage:
      out = graph.invoke({"req": req})
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
            return {
                "status": "error",
                "selected_tool": "none",
                "stdout": tr.stdout,
                "stderr": tr.stderr,
                "artifacts": tr.artifacts,
                "tool_result": _toolresult_to_dict(tr),
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
            return {
                "status": "error",
                "selected_tool": tool.name,
                "stdout": tr.stdout,
                "stderr": tr.stderr,
                "artifacts": tr.artifacts,
                "tool_result": _toolresult_to_dict(tr),
            }

        # Run tool
        tr = tool.run(req)

        # Ensure router info is always present in artifacts
        artifacts = dict(tr.artifacts or {})
        artifacts.update(
            {
                "capability_id": cap_id,
                "selected_by": selected_by,
                "router_reason": router_reason,
            }
        )
        tr.artifacts = artifacts

        return {
            "status": tr.status,
            "selected_tool": tr.selected_tool,
            "stdout": tr.stdout,
            "stderr": tr.stderr,
            "artifacts": artifacts,
            "tool_result": _toolresult_to_dict(tr),
        }


# Singleton graph (importable by app.py)
graph = SimpleGraph()
