# causal-agent-mvp

A minimal **agentic causal analysis API** that routes natural-language requests to appropriate causal inference tools and executes them deterministically.

The system demonstrates how an LLM (optional) can be used for **capability selection**, while the statistical analysis itself remains transparent, reproducible, and auditable.

---

## What this project does

The API exposes a single `/run` endpoint that accepts:
- a CSV dataset
- a natural-language request
- optional column specifications (treatment, outcome, time, event, covariates)

A lightweight agentic workflow then:
1. **Interprets the request** (rule-based or LLM-based routing)
2. **Selects the appropriate causal capability**
3. **Executes the analysis deterministically**
4. **Returns structured outputs** (text + JSON artifacts)

---

## Supported capabilities

- **Causal ATE estimation**
  - Doubly robust estimators via `CausalModels`
  - Example: IHDP benchmark data

- **Adjusted survival analysis**
  - IPTW-adjusted Kaplan–Meier curves via `adjustedCurves`
  - Example: Dialysis survival data

---

## Demo 1 — Survival analysis with covariate adjustment

**Natural-language request**
> “Compare survival between groups with covariate adjustment”

```bash
curl -s -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "csv": "data/dialysis survival dataset.csv",
    "request": "Compare survival between groups with covariate adjustment",
    "use_llm_router": true,
    "time": "time",
    "event": "event",
    "group": "begin",
    "covariates": ["age"]
  }'
```
Result (excerpt)
```jason

{
  "status": "ok",
  "selected_tool": "adjustedcurves",
  "artifacts": {
    "capability_id": "survival_adjusted_curves",
    "selected_by": "llm"
  },
  "error": null
}
```

✔ The agent selects the survival adjusted curves capability and runs an IPTW-adjusted Kaplan–Meier analysis.

## Demo 2 — Causal effect estimation (ATE)

**Natural-language request**

> “Estimate the causal effect of treatment on outcome”
```bash
curl -s -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "csv": "data/ihdp_data.csv",
    "request": "Estimate the causal effect of treatment on outcome",
    "use_llm_router": true,
    "treatment": "treatment",
    "outcome": "y_factual",
    "covariates": ["x1", "x2", "x3", "x4", "x5"]
  }'
```
Result (excerpt)
```json
{
  "status": "ok",
  "selected_tool": "causalmodels",
  "artifacts": {
    "capability_id": "causal_ate",
    "summary_json": "out/api/causalmodels.summary.json"
  },
  "error": null
}
```

✔ The agent selects the causal ATE capability and runs a doubly robust estimator, producing an ATE with confidence intervals and a structured JSON summary.

Design principles

Agentic, not generative
LLMs are used only for decision-making, not statistical computation.

Deterministic execution
Once a capability is selected, analysis is fully reproducible.

Explainable routing
Each run records why a tool was chosen.

Structured outputs
Results are returned as both human-readable logs and machine-readable JSON.

Project status

This repository is an MVP / research prototype intended to demonstrate:

agent-based orchestration of causal analyses

clean separation between reasoning and estimation

extensibility to additional causal tasks

