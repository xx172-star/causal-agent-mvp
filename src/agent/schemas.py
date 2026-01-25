from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

TaskName = Literal["auto", "ate", "survival"]
ToolName = Literal["causalmodels", "adjustedcurves"]


class RunRequest(BaseModel):
    csv: str = Field(..., description="Path to CSV file (relative to repo root or absolute)")
    task: TaskName = Field("auto", description="auto | ate | survival")

    # natural language requirement (optional)
    request: Optional[str] = Field(
        None,
        description="Natural language requirement, e.g. 'Estimate causal effect of treatment on outcome'."
    )

    # force a specific capability from pipeline_capabilities.json (optional)
    capability_id: Optional[str] = Field(
        None,
        description="Capability id from pipeline_capabilities.json, e.g. 'causal_ate'."
    )

    # ATE fields
    outcome: Optional[str] = None
    treatment: Optional[str] = None

    # Survival fields
    time: Optional[str] = None
    event: Optional[str] = None
    group: Optional[str] = None

    covariates: Optional[List[str]] = None
    max_covariates: int = 15

    # Output directory for artifacts
    out_dir: str = "out/api"

    # Planner controls (optional) 
    use_llm_router: bool = False
    llm_model: str = "gpt-4o-mini"  # placeholder model name; change if you want


class RunResult(BaseModel):
    status: Literal["ok", "error"]
    selected_tool: Optional[ToolName] = None
    stdout: str = ""
    stderr: str = ""
    artifacts: Dict[str, Any] = {}
    error: Optional[str] = None
