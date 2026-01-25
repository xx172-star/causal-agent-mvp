#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo "[API DEMO 1] IHDP -> causalmodels (ATE)"
echo "======================================"
curl -s http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "csv":"data/ihdp_data.csv",
    "task":"auto",
    "treatment":"treatment",
    "outcome":"y_factual",
    "covariates":["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15"],
    "max_covariates": 15,
    "out_dir":"out/api",
    "use_llm_router": true
  }' | python -m json.tool

echo
echo "======================================"
echo "[API DEMO 2] Dialysis -> adjustedcurves"
echo "======================================"
curl -s http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{
    "csv":"data/dialysis survival dataset.csv",
    "task":"auto",
    "group":"begin",
    "time":"time",
    "event":"event",
    "covariates":["age","center","disease_diabetes","disease_hypert","disease_other","disease_renal"],
    "use_llm_router": true
  }' | python -m json.tool
