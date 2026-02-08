# src/agent/tools/registry.py
from __future__ import annotations

from typing import Dict

from src.agent.tools.base import BaseTool

_REGISTRY: Dict[str, BaseTool] = {}


def register(tool: BaseTool) -> None:
    _REGISTRY[tool.capability_id] = tool


def get_tool(capability_id: str) -> BaseTool:
    if capability_id not in _REGISTRY:
        raise KeyError(f"No tool registered for capability_id={capability_id}")
    return _REGISTRY[capability_id]


def list_tools() -> Dict[str, str]:
    return {k: v.name for k, v in _REGISTRY.items()}
