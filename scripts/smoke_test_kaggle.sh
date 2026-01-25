#!/usr/bin/env bash
set -euo pipefail

scripts/demo_kaggle.sh

echo ""
echo "[SMOKE] Check IHDP JSON output exists and status ok"

python - <<'PY'
import json, os, sys

p = "out/demo/ihdp_causalmodels.summary.json"
if not os.path.exists(p):
    print(f"[FAIL] Missing: {p}")
    sys.exit(1)

with open(p, "r", encoding="utf-8") as f:
    j = json.load(f)

status = j.get("status")
if status not in ("ok", "ok_with_na"):
    print(f"[FAIL] Bad status in {p}: {status}")
    sys.exit(1)

print(f"[OK] {p} status={status} ate={j.get('ate')}")
print("[SMOKE] PASS")
PY
