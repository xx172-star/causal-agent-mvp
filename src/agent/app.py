# src/agent/app.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .schemas import RunRequest, RunResult
from .graph import graph  # ✅ use singleton graph (SimpleGraph)

# import the *capability* router (returns {"capability_id", "reason"})
try:
    from .router_llm import llm_choose_capability
except Exception:
    llm_choose_capability = None  # type: ignore


app = FastAPI(title="Causal Agent MVP")


def _repo_root() -> Path:
    # src/agent/app.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_capabilities() -> Dict[str, Any]:
    p = _repo_root() / "pipeline_capabilities.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _capability_exists(cap_id: str) -> bool:
    try:
        caps = _load_capabilities().get("capabilities", [])
        return any(c.get("id") == cap_id for c in caps)
    except Exception:
        return False


def select_capability(req: RunRequest) -> Tuple[str, str, str]:
    """
    Minimal routing (LLM-first):

      Priority:
        1) req.capability_id (force)
        2) LLM router if use_llm_router + request
        3) req.task (explicit)
        4) rule-based auto

    Returns: (capability_id, selected_by, router_reason)
    """
    # 1) forced
    if req.capability_id:
        return req.capability_id, "capability_id", "Forced by capability_id."

    # 2) LLM router
    if req.use_llm_router and req.request and llm_choose_capability is not None:
        try:
            obj = llm_choose_capability(
                request=req.request,
                csv_columns=None,
                model=req.llm_model,
            )
            cap_id = (obj.get("capability_id") or "").strip()
            reason = (obj.get("reason") or "").strip() or "LLM selected capability."
            if cap_id:
                return cap_id, "llm", reason
        except Exception:
            pass

    # 3) explicit task
    if req.task == "ate":
        return "causal_ate", "task", "Selected by explicit task='ate'."
    if req.task == "survival":
        return "survival_adjusted_curves", "task", "Selected by explicit task='survival'."

    # 4) rule-based auto
    if req.time and req.event and req.group:
        return "survival_adjusted_curves", "auto", "Auto: time/event/group detected."
    return "causal_ate", "auto", "Auto: defaulting to causal_ate."


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run", response_model=RunResult)
def run(req: RunRequest):
    # --- capability selection ---
    cap_id, selected_by, router_reason = select_capability(req)

    if not _capability_exists(cap_id):
        return JSONResponse(
            status_code=400,
            content=RunResult(
                status="error",
                selected_tool=None,
                stdout="",
                stderr="",
                artifacts={
                    "capability_id": cap_id,
                    "selected_by": selected_by,
                    "router_reason": router_reason,
                },
                error=f"Unknown capability_id='{cap_id}'. Check pipeline_capabilities.json.",
            ).model_dump(),
        )

    # Only two capabilities are implemented right now
    if cap_id not in ("causal_ate", "survival_adjusted_curves"):
        return JSONResponse(
            status_code=400,
            content=RunResult(
                status="error",
                selected_tool=None,
                stdout="",
                stderr="",
                artifacts={
                    "capability_id": cap_id,
                    "selected_by": selected_by,
                    "router_reason": router_reason,
                },
                error=f"Capability '{cap_id}' is declared but not implemented in this API path yet.",
            ).model_dump(),
        )

    # ✅ IMPORTANT: graph.py expects either:
    #   - src.agent.schemas_io.RunRequest dataclass
    #   - OR a dict of RunRequest fields
    # Your FastAPI RunRequest is a Pydantic model -> pass dict via model_dump()
    payload = req.model_dump()

    # Force graph to use the already-selected capability and avoid re-routing
    payload["capability_id"] = cap_id
    payload["use_llm_router"] = False

    # --- new behavior (plugin framework graph) ---
    out = graph.invoke({"req": payload})

    tool_result = out.get("tool_result", {}) if isinstance(out, dict) else {}
    selected_tool = out.get("selected_tool") if isinstance(out, dict) else None

    base_artifacts = tool_result.get("artifacts", {}) if isinstance(tool_result, dict) else {}
    if not isinstance(base_artifacts, dict):
        base_artifacts = {}

    # Always attach router metadata (your app-level router)
    artifacts = {
        **base_artifacts,
        "capability_id": cap_id,
        "selected_by": selected_by,
        "router_reason": router_reason,
    }

    # Tool exit handling (graph may return ok/error)
    code = tool_result.get("exit_code", 1) if isinstance(tool_result, dict) else 1
    status = out.get("status") if isinstance(out, dict) else "error"

    if status != "ok" or code != 0:
        return JSONResponse(
            status_code=500,
            content=RunResult(
                status="error",
                selected_tool=selected_tool,
                stdout=str(tool_result.get("stdout", "")),
                stderr=str(tool_result.get("stderr", "")),
                artifacts=artifacts,
                error=f"Tool failed (status={status}, exit_code={code})",
            ).model_dump(),
        )

    return RunResult(
        status="ok",
        selected_tool=selected_tool,
        stdout=str(tool_result.get("stdout", "")),
        stderr=str(tool_result.get("stderr", "")),
        artifacts=artifacts,
        error=None,
    )

import os

