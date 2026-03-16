# src/agent/app.py
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .schemas import RunRequest, RunResult
from .graph import graph  

# import the *capability* router (returns {"capability_id", "reason"})
try:
    from .router_llm import llm_choose_capability  # type: ignore
except Exception:
    llm_choose_capability = None  # type: ignore

# import capability loader (cap_*.json)
try:
    from .router_llm import load_capabilities  # type: ignore
except Exception:
    load_capabilities = None  # type: ignore


app = FastAPI(title="Causal Agent MVP")


def _repo_root() -> Path:
    # src/agent/app.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _capabilities_dir() -> Path:
    # repo_root/src/agent/capabilities
    return _repo_root() / "src" / "agent" / "capabilities"


def _load_capabilities_fallback() -> List[Dict[str, Any]]:
    """
    Fallback loader if router_llm.load_capabilities() is unavailable.
    Reads src/agent/capabilities/cap_*.json
    """
    caps_dir = _capabilities_dir()
    if not caps_dir.exists():
        return []

    out: List[Dict[str, Any]] = []
    for p in sorted(caps_dir.glob("cap_*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            continue
    return out


def _load_capability_specs() -> List[Dict[str, Any]]:
    """
    Source of truth: src/agent/capabilities/cap_*.json
    Prefer router_llm.load_capabilities() if available; otherwise fallback.
    """
    if load_capabilities is not None:
        try:
            caps = load_capabilities()
            if isinstance(caps, list):
                return caps
        except Exception:
            pass
    return _load_capabilities_fallback()


def _allowed_capability_ids() -> List[str]:
    """
    Return allowed capability ids from cap specs.
    We accept either {"capability_id": ...} or {"id": ...}.
    """
    ids: List[str] = []
    for c in _load_capability_specs():
        if not isinstance(c, dict):
            continue
        cid = (c.get("capability_id") or c.get("id") or "").strip()
        if cid:
            ids.append(cid)
    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for x in ids:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


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


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _persist_api_outputs(
    *,
    payload: Dict[str, Any],
    status: str,
    selected_tool: Any,
    stdout: str,
    stderr: str,
    artifacts: Dict[str, Any],
    error: Any,
) -> Tuple[str, str]:
    """
    Persist request/result/stdout/stderr into:
      <repo_root>/out/api/<run_id>/
    Returns: (run_id, out_dir_str)
    """
    run_id = _new_run_id()
    out_base = _repo_root() / "out" / "api"
    out_dir = out_base / run_id

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "request.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        (out_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
        (out_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")

        (out_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "selected_tool": selected_tool,
                    "stdout": stdout,
                    "stderr": stderr,
                    "artifacts": artifacts,
                    "error": error,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        # Never break the API response because of file I/O.
        pass

    return run_id, str(out_dir)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run", response_model=RunResult)
def run(req: RunRequest):
    # --- capability selection ---
    cap_id, selected_by, router_reason = select_capability(req)

    allowed = _allowed_capability_ids()
    allowed_set = set(allowed)

    # Build payload early so even errors can be persisted.
    payload = req.model_dump()

    # Force graph to use the already-selected capability and avoid re-routing
    payload["capability_id"] = cap_id
    payload["use_llm_router"] = False

    # --- reject unknown capability_id ---
    if cap_id not in allowed_set:
        artifacts = {
            "capability_id": cap_id,
            "selected_by": selected_by,
            "router_reason": router_reason,
            "allowed_capabilities": allowed,
        }
        err_msg = (
            f"Unknown capability_id='{cap_id}'. "
            f"Check src/agent/capabilities/cap_*.json."
        )

        run_id, out_dir = _persist_api_outputs(
            payload=payload,
            status="error",
            selected_tool=None,
            stdout="",
            stderr="",
            artifacts={**artifacts, "run_id": "", "out_dir": ""},
            error=err_msg,
        )
        artifacts["run_id"] = run_id
        artifacts["out_dir"] = out_dir

        return JSONResponse(
            status_code=400,
            content=RunResult(
                status="error",
                selected_tool=None,
                stdout="",
                stderr="",
                artifacts=artifacts,
                error=err_msg,
            ).model_dump(),
        )

    # --- execute ---
    out = graph.invoke({"req": payload})

    tool_result = out.get("tool_result", {}) if isinstance(out, dict) else {}
    selected_tool = out.get("selected_tool") if isinstance(out, dict) else None

    base_artifacts = tool_result.get("artifacts", {}) if isinstance(tool_result, dict) else {}
    if not isinstance(base_artifacts, dict):
        base_artifacts = {}

    # Always attach router metadata
    artifacts: Dict[str, Any] = {
        **base_artifacts,
        "capability_id": cap_id,
        "selected_by": selected_by,
        "router_reason": router_reason,
    }

    stdout = str(tool_result.get("stdout", "")) if isinstance(tool_result, dict) else ""
    stderr = str(tool_result.get("stderr", "")) if isinstance(tool_result, dict) else ""

    # Tool exit handling (graph may return ok/error)
    code = tool_result.get("exit_code", 1) if isinstance(tool_result, dict) else 1
    status = out.get("status") if isinstance(out, dict) else "error"

    # Determine final API status/error
    if status != "ok" or code != 0:
        api_status = "error"
        api_error: Any = f"Tool failed (status={status}, exit_code={code})"
        http_code = 500
    else:
        api_status = "ok"
        api_error = None
        http_code = 200

    # --- persist to out/api/<run_id>/ ---
    run_id, out_dir = _persist_api_outputs(
        payload=payload,
        status=api_status,
        selected_tool=selected_tool,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
        error=api_error,
    )
    artifacts["run_id"] = run_id
    artifacts["out_dir"] = out_dir

    # --- return ---
    if api_status != "ok":
        return JSONResponse(
            status_code=http_code,
            content=RunResult(
                status="error",
                selected_tool=selected_tool,
                stdout=stdout,
                stderr=stderr,
                artifacts=artifacts,
                error=api_error,
            ).model_dump(),
        )

    return RunResult(
        status="ok",
        selected_tool=selected_tool,
        stdout=stdout,
        stderr=stderr,
        artifacts=artifacts,
        error=None,
    )
