#!/usr/bin/env bash
set -euo pipefail

mkdir -p out/demo

echo "======================================"
echo "[DEMO 1] IHDP -> CausalModels doubly_robust"
echo "======================================"
python src/run_causalmodels_demo.py \
  --csv data/ihdp_data.csv \
  --treatment treatment \
  --outcome y_factual \
  --out_json out/demo/ihdp_causalmodels.summary.json \
  --max_covariates 15

echo ""
echo "[OK] IHDP finished."
echo "Summary: out/demo/ihdp_causalmodels.summary.json"

echo ""
echo "======================================"
echo "[DEMO 2] Dialysis survival -> adjustedCurves"
echo "======================================"
python src/run_adjustedcurves_demo.py \
  --csv "data/PBC_agent01.csv" \
  --group trt01 \
  --time time \
  --event event \
  --covariates age

echo ""
echo "[OK] Dialysis survival demo finished."
