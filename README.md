# causal-agent-mvp

An agent-based causal inference pipeline that automatically selects and executes appropriate causal analysis tools (e.g., ATE estimation, survival adjusted curves) based on user requests and dataset structure.

The system combines rule-based checks, LLM-assisted routing, and deterministic statistical backends, and produces both human-readable outputs and structured JSON artifacts.

---

## Features

- Agentic workflow for causal inference
- Automatic capability selection via an LLM router
- Support for:
  - Average Treatment Effect (ATE) estimation
  - Survival analysis with confounder-adjusted curves
- End-to-end reproducible demos with real datasets
- Structured JSON outputs for downstream use

---

## Repository Structure

causal-agent-mvp/
├── data/ # Example datasets (PBC, GBSG2)
├── scripts/ # Demo and helper scripts
├── src/agent/ # Agent logic, router, schemas
├── out/ # Runtime outputs (gitignored)
├── README.md

## Requirements

- Python 3.9+
- R (required for survival adjusted curves)
- Required R packages:
  - adjustedCurves
  - WeightIt
  - survival

---

## Quickstart: End-to-End Demos

Below are two fully tested demo commands.  
Both have been run successfully end-to-end and generate structured JSON outputs under `out/api/`.

---

### Demo 1: Average Treatment Effect (ATE)

Estimate the causal effect of treatment on a binary 5-year outcome using doubly robust estimation.
```bash
curl -s -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "csv": "data/PBC_ate5y_cc.csv",
    "request": "Estimate the causal effect (ATE) of treatment on 5-year survival",
    "use_llm_router": true,
    "treatment": "trt01",
    "outcome": "Y5y",
    "covariates": ["age","bili","albumin","protime","edema","platelet","ast"]
  }'
```

Expected behavior:

Selected capability: causal_ate

Backend: doubly robust ATE estimation

Output includes ATE, standard error, and 95% confidence interval

JSON summary written to:

```pgsql

out/api/causalmodels.summary.json
```

Demo 2: Survival Adjusted Curves

Compare survival between treatment groups using inverse probability weighted Kaplan–Meier curves.

```bash
curl -s -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "csv": "data/GBSG2_agent01.csv",
    "request": "Compare survival between groups",
    "use_llm_router": true,
    "time": "time",
    "event": "event",
    "group": "horTh01",
    "covariates": []
  }'


```

Expected behavior:

Selected capability: survival_adjusted_curves

Method: IPTW-adjusted Kaplan–Meier

JSON summary written to:
```pgsql

out/api/adjustedcurves.summary.json
```



Output Format

Each run returns:

selected_tool: executed tool

stdout / stderr: human-readable logs

artifacts.summary_json: path to structured JSON output

artifacts.capability_id: selected causal capability

artifacts.router_reason: explanation of tool selection

Example:
```json
{
  "status": "ok",
  "selected_tool": "causalmodels",
  "artifacts": {
    "capability_id": "causal_ate",
    "summary_json": "out/api/causalmodels.summary.json"
  }
}
```


Input Data Assumptions

Treatment is binary (0/1 or two distinct levels)

Covariates are measured prior to treatment

No missing values after preprocessing

Positivity holds (non-zero probability of treatment assignment)

If these conditions are violated, results may be unstable or estimation may fail.

Notes

Informational messages from R packages (e.g., package loading) may appear in stderr and can be safely ignored.

Runtime outputs under out/ are not tracked by git.

