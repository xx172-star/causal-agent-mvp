# causal-agent-mvp

An agent-based causal inference pipeline that automatically selects and executes appropriate causal analysis tools (e.g., ATE estimation, survival adjusted curves) based on user requests and dataset structure.

The system combines LLM-assisted routing, rule-based safeguards, and
deterministic statistical backends, and produces both human-readable outputs
and structured JSON artifacts for downstream use.
---

## Key Features

- Agentic workflow for causal inference
- LLM-based capability routing with deterministic fallback
- Plugin-based, extensible tool architecture
- Support for:
  - Average Treatment Effect (ATE) estimation
  - Survival analysis with confounder-adjusted curves
- End-to-end reproducible demos on real-world datasets
- Standardized JSON artifacts for inspection and reuse

---

## Repository Structure

causal-agent-mvp/

├── data/ # Example datasets (PBC, GBSG2)

├── scripts/ # Demo and helper scripts

├── src/agent/ # Agent logic, router, schemas, and tools

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

Below are two fully tested demo commands illustrating the complete
LLM-routed causal analysis pipeline.

Both demos have been run successfully end-to-end and return structured
JSON outputs.

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

#### Expected behavior:

Selected capability: causal_ate

Backend: doubly robust ATE estimation

Output includes ATE, standard error, and 95% confidence interval


### Demo 2: Survival Adjusted Curves

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

#### Expected behavior:

Selected capability: survival_adjusted_curves

Method: IPTW-adjusted Kaplan–Meier

### Output Format

Each API call returns a structured response containing:

selected_tool: the executed backend tool

stdout / stderr: human-readable logs

artifacts:

capability_id: selected causal capability

selected_by: llm or auto

router_reason: explanation of the routing decision

optional paths to serialized JSON summaries

#### Example response:

```json
{
  "status": "ok",
  "selected_tool": "causalmodels",
  "artifacts": {
    "capability_id": "causal_ate",
    "selected_by": "llm",
    "router_reason": "The request explicitly asks for ATE estimation."
  }
}
```


## Input Data Requirements

The framework assumes that the input dataset satisfies the following conditions.
If these conditions are met, the pipeline can be executed end-to-end without
modification to the core codebase.

---

### General Format

Input data must be provided as a CSV file.

Each row corresponds to one observational unit.

Column names must be explicitly referenced in the API request.

---

### Required Variables

#### Treatment / Exposure

A treatment (or exposure) variable must be specified.

The treatment variable must be binary:

0 / 1, or

two distinct values encoding treated vs. control groups.

Treatment assignment is assumed to be observed at baseline.

---

#### Outcome

For ATE estimation:

The outcome may be binary or continuous.

For survival analysis:

A time-to-event variable is required.

An event indicator must be provided (1 = event, 0 = censored).

Outcome variables must not contain missing values after preprocessing.
---

#### Covariates (Confounders)

One or more covariates may be supplied for confounding adjustment.

Covariates must:

Be measured prior to treatment assignment

Be numeric or numerically encoded

Missing values must be handled before execution.
---

### Structural and Causal Assumptions

The framework relies on standard causal inference assumptions:

Consistency

Positivity

No unmeasured confounding, conditional on supplied covariates

These assumptions are not automatically verified and must be justified
by the user.

---

### Notes on Robustness

Informational messages from underlying statistical packages may appear
during execution and do not necessarily indicate failure.

Severe violations of positivity or covariate overlap may lead to unstable
estimates.

