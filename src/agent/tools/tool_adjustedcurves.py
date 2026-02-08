# src/agent/tools/tool_adjustedcurves.py
from __future__ import annotations

import json
import subprocess
import sys
from typing import Tuple

from src.agent.schemas_io import RunRequest, ToolResult
from src.agent.tools.base import BaseTool


class AdjustedCurvesTool(BaseTool):
    @property
    def name(self) -> str:
        return "adjustedcurves"

    @property
    def capability_id(self) -> str:
        return "survival_adjusted_curves"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        if not req.csv:
            return False, "Survival requires csv."
        if not (req.time and req.event and req.group):
            return False, "Survival requires time, event, and group."
        return True, "ok"

    def run(self, req: RunRequest) -> ToolResult:
        # call existing script: src/run_adjustedcurves_demo.py
        cmd = [
            sys.executable,
            "src/run_adjustedcurves_demo.py",
            "--csv", req.csv,
            "--time", req.time or "",
            "--event", req.event or "",
            "--group", req.group or "",
            "--covariates", ",".join(req.covariates or []),
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        stdout, stderr = p.stdout, p.stderr

        artifacts = {}
        for line in reversed(stdout.splitlines()):
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    artifacts = json.loads(s)
                    break
                except Exception:
                    pass

        status = "ok" if p.returncode == 0 else "error"
        return ToolResult(
            status=status,
            selected_tool=self.name,
            stdout=stdout,
            stderr=stderr,
            exit_code=p.returncode,
            artifacts=artifacts,
        )
