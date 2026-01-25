from __future__ import annotations

from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph, END

from .schemas import RunRequest
from .tools import run_causalmodels_tool, run_adjustedcurves_tool

# LLM router is optional
try:
    from .router_llm import llm_choose_tool
except Exception:
    llm_choose_tool = None  # type: ignore


class AgentState(TypedDict, total=False):
    req: RunRequest
    selected_tool: str
    tool_result: Dict[str, Any]
    error: str


def choose_tool_rule_based(req: RunRequest) -> str:
    if req.task == "ate":
        return "causalmodels"
    if req.task == "survival":
        return "adjustedcurves"

    # auto
    if req.time and req.event and req.group:
        return "adjustedcurves"
    return "causalmodels"


def choose_tool(state: AgentState) -> AgentState:
    req = state["req"]

    # Default: rule-based (stable MVP)
    selected = choose_tool_rule_based(req)

    # Optional: LLM router (only if user asks AND available)
    if req.use_llm_router and llm_choose_tool is not None:
        try:
            obj = llm_choose_tool(req.model_dump(), model=req.llm_model)
            selected = obj["selected_tool"]
        except Exception as e:
            # fail open: fallback to rule-based
            state["error"] = f"LLM router failed (fallback to rule-based): {e}"

    state["selected_tool"] = selected
    return state


def run_tool(state: AgentState) -> AgentState:
    req = state["req"]
    tool = state["selected_tool"]

    try:
        if tool == "causalmodels":
            if not req.treatment or not req.outcome:
                raise ValueError("ATE requires 'treatment' and 'outcome'.")
            r = run_causalmodels_tool(
                csv=req.csv,
                treatment=req.treatment,
                outcome=req.outcome,
                covariates=req.covariates,
                max_covariates=req.max_covariates,
                out_dir=req.out_dir,
            )
        elif tool == "adjustedcurves":
            if not req.group or not req.time or not req.event:
                raise ValueError("Survival requires 'group', 'time', and 'event'.")
            r = run_adjustedcurves_tool(
                csv=req.csv,
                group=req.group,
                time=req.time,
                event=req.event,
                covariates=req.covariates,
            )
        else:
            raise ValueError(f"Unknown tool: {tool}")

        state["tool_result"] = r
        return state

    except Exception as e:
        state["error"] = str(e)
        return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("choose_tool", choose_tool)
    g.add_node("run_tool", run_tool)
    g.set_entry_point("choose_tool")
    g.add_edge("choose_tool", "run_tool")
    g.add_edge("run_tool", END)
    return g.compile()
