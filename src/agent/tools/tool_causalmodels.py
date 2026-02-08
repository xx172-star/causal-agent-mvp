# src/agent/tools/tool_causalmodels.py
from __future__ import annotations

import json
import subprocess
import sys
from typing import Tuple

from src.agent.schemas_io import RunRequest, ToolResult
from src.agent.tools.base import BaseTool


class CausalModelsTool(BaseTool):
    @property
    def name(self) -> str:
        return "causalmodels"

    @property
    def capability_id(self) -> str:
        return "causal_ate"

    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        if not req.csv:
            return False, "ATE requires csv."
        if not req.treatment or not req.outcome:
            return False, "ATE requires treatment and outcome."
        return True, "ok"

    def run(self, req: RunRequest) -> ToolResult:
        # call existing script: src/run_causalmodels_demo.py
        cmd = [
            sys.executable,
            "src/run_causalmodels_demo.py",
            "--csv", req.csv,
            "--treatment", req.treatment or "",
            "--outcome", req.outcome or "",
            "--covariates", ",".join(req.covariates or []),
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        stdout, stderr = p.stdout, p.stderr

        artifacts = {}
        # If the demo prints a JSON line, parse it (optional)
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
