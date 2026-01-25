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
