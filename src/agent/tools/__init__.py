from src.agent.tools.registry import register

from src.agent.tools.tool_dummy import DummyTool
from src.agent.tools.tool_causalmodels import CausalModelsTool
from src.agent.tools.tool_adjustedcurves import AdjustedCurvesTool

register(DummyTool())
register(CausalModelsTool())
register(AdjustedCurvesTool())
