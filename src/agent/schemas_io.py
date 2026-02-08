# src/agent/schemas_io.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunRequest:
    csv: str
    request: str = ""
    capability_id: Optional[str] = None
    use_llm_router: bool = True

    # ATE
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    covariates: List[str] = field(default_factory=list)

    # Survival
    time: Optional[str] = None
    event: Optional[str] = None
    group: Optional[str] = None

    # Optional preprocessing config
    preprocess: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    status: str  # "ok" | "error"
    selected_tool: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    artifacts: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
