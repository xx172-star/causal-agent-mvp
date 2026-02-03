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
    "covariates":["age"],
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
    "csv":"data/PBC_agent01.csv",
    "task":"auto",
    "group":"trt01",
    "time":"time",
    "event":"event",
    "covariates":["age"],
    "use_llm_router": true
  }' | python -m json.tool
