# src/agent/tools/tool_dummy.py
from __future__ import annotations
from typing import Tuple

from src.agent.schemas_io import RunRequest, ToolResult
from src.agent.tools.base import BaseTool


class DummyTool(BaseTool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def capability_id(self) -> str:
        return "dummy_capability"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        return True, "ok"

    def run(self, req: RunRequest) -> ToolResult:
        return ToolResult(
            status="ok",
            selected_tool=self.name,
            stdout=f"Dummy ran on csv={req.csv}",
            artifacts={"hello": "world"},
        )
