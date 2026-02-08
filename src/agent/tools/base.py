# src/agent/tools/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from src.agent.schemas_io import RunRequest, ToolResult


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capability_id(self) -> str:
        ...

    @abstractmethod
    def validate(self, req: RunRequest) -> Tuple[bool, str]:
        """Return (ok, reason)."""
        ...

    @abstractmethod
    def run(self, req: RunRequest) -> ToolResult:
        ...
