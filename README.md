# causal-agent-mvp

A minimal agentic API for causal analysis, built with FastAPI and LangGraph.

## What it does
The API exposes a single `/run` endpoint that routes requests to different causal analysis tools:
- **CausalModels**: doubly robust ATE estimation (IHDP example)
- **adjustedCurves**: IPTW-adjusted survival curves (Dialysis example)

Routing is handled by a lightweight LangGraph workflow (rule-based by default, with optional OpenAI LLM routing).

## Architecture (high-level)
- FastAPI for the HTTP layer
- LangGraph for routing + tool orchestration
- Deterministic tool execution with unified JSON outputs

## Example usage
Start the API:
```bash
uvicorn src.agent.app:app --port 8000

ATE example (IHDP):

curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "csv":"data/ihdp_data.csv",
    "task":"ate",
    "treatment":"treatment",
    "outcome":"y_factual",
    "covariates":["x1","x2","x3","x4","x5"]
  }'

Survival example (Dialysis):

curl -X POST http://127.0.0.1:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "csv":"data/dialysis survival dataset.csv",
    "task":"survival",
    "group":"begin",
    "time":"time",
    "event":"event",
    "covariates":["age","center","disease_diabetes"]
  }'
