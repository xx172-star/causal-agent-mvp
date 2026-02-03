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

Expected behavior:

Selected capability: survival_adjusted_curves

Method: IPTW-adjusted Kaplan–Meier

JSON summary written to:
```pgsql

out/api/adjustedcurves.summary.json
```



### Output Format

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


## Input Data Requirements

This framework assumes that the input dataset satisfies the following requirements.  
As long as these conditions are met, the pipeline can be executed end-to-end without modification to the core codebase.

---

### General Format

- Input data must be provided as a **CSV file**.
- Each row corresponds to one observational unit (e.g., patient or subject).
- Column names must be explicitly referenced in the API request.

---

### Required Variables

The required variables depend on the selected causal task.

#### 1. Treatment / Exposure

- A treatment (or exposure) variable must be specified.
- The treatment variable must be **binary**, represented as:
  - `0 / 1`, or  
  - two distinct values that clearly encode treated vs. control groups.
- Treatment assignment is assumed to be observed and fixed at baseline.

---

#### 2. Outcome

- For **ATE estimation**:
  - The outcome may be **binary or continuous**.
- For **survival analysis**:
  - A **time-to-event** variable must be provided.
  - An **event indicator** must be provided (`1` = event, `0` = censored).
- Outcome variables must not contain missing values after preprocessing.

---

#### 3. Covariates (Confounders)

- One or more covariates may be provided to adjust for confounding.
- Covariates must:
  - Be measured **prior to treatment assignment**
  - Be numeric or numerically encoded (e.g., one-hot encoding for categorical variables)
- Missing values must be handled prior to analysis.

---

### Structural and Causal Assumptions

The framework relies on standard causal inference assumptions:

- **Consistency**: observed outcomes correspond to the assigned treatment.
- **Positivity**: each covariate pattern has a non-zero probability of receiving each treatment level.
- **No unmeasured confounding**, conditional on the provided covariates.

These assumptions are not automatically verified by the framework and must be justified by the user.

---

### Preprocessing Expectations

To ensure stable execution:

- All required variables must be present and correctly specified in the request.
- Missing values should be removed or imputed prior to execution.
- Categorical variables should be encoded numerically.
- Extremely rare treatment groups or severe lack of covariate overlap may lead to unstable estimates.

---

### Notes on Robustness

- Informational messages or warnings from underlying statistical packages may appear during execution and do not necessarily indicate failure.
- If the input data violate the requirements above (e.g., severe non-positivity), estimation results may be unstable or unreliable.

In summary, this framework is designed to be **methodologically transparent rather than permissive**:  
it does not attempt to automatically coerce invalid inputs, but instead clearly documents the conditions under which valid causal analyses can be performed.


### Notes

Informational messages from R packages (e.g., package loading) may appear in stderr and can be safely ignored.

Runtime outputs under out/ are not tracked by git.

