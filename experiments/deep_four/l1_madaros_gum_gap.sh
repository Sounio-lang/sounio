#!/usr/bin/env bash
# L1 — measure Knowledge GUM surface: Madaros (default) vs lean_single (β¹⁰/β¹¹).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/results/deep_four"
mkdir -p "$OUT"
LOG="$OUT/l1_madaros_gum_gap.RUNLOG.txt"
: > "$LOG"

run_one() {
  local engine="$1" test="$2" tag="$3"
  echo "=== engine=$engine test=$test ===" | tee -a "$LOG"
  if [[ "$engine" == "lean_single" ]]; then
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "$test" >>"$LOG" 2>&1 || true
  else
    env -u SOUNIO_SOUC_ENGINE ./bin/souc run "$test" >>"$LOG" 2>&1 || true
  fi
  echo "TAG=$tag" >>"$LOG"
}

run_one madaros tests/run-pass/beta10_product_cancel_variance.sio cancel_madaros
run_one lean_single tests/run-pass/beta10_product_cancel_variance.sio cancel_lean
run_one madaros tests/run-pass/beta10_fano_associator_var_fo.sio fano_madaros
run_one lean_single tests/run-pass/beta10_fano_associator_var_fo.sio fano_lean

python3 - <<'PY'
import json, re
from pathlib import Path
from datetime import datetime, timezone
import subprocess
log = Path("results/deep_four/l1_madaros_gum_gap.RUNLOG.txt").read_text()
sha = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()

def has(s): return s in log
# Split by TAG= markers roughly
parts = log.split("TAG=")
cancel_m = "BETA10_PRODUCT_CANCEL_PASS" in log.split("TAG=cancel_madaros")[0] if "cancel_madaros" in log else False
# simpler scans
cancel_madaros = log.count("BETA10_PRODUCT_CANCEL_PASS") >= 1  # both may pass
# parse per section
sec = {}
cur = None
for line in log.splitlines():
    if line.startswith("=== engine="):
        cur = line
        sec[cur] = []
    elif cur:
        sec[cur].append(line)

def section_has(engine, test_snip, marker):
    for k, lines in sec.items():
        if engine in k and test_snip in k:
            return any(marker in ln for ln in lines)
    return False

rec = {
  "schema": "deep_four.l1_madaros_gum_gap.v1",
  "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "git_sha": sha,
  "tests": {
    "product_cancel": {
      "madaros": "PASS" if section_has("madaros", "beta10_product_cancel", "BETA10_PRODUCT_CANCEL_PASS") else "FAIL",
      "lean_single": "PASS" if section_has("lean_single", "beta10_product_cancel", "BETA10_PRODUCT_CANCEL_PASS") else "FAIL",
    },
    "fano_fo_64sig2": {
      "madaros": "PASS" if section_has("madaros", "beta10_fano_associator", "BETA10_FANO_FO_VAR_PASS") else "FAIL_HONEST",
      "lean_single": "PASS" if section_has("lean_single", "beta10_fano_associator", "BETA10_FANO_FO_VAR_PASS") else "FAIL_HONEST",
    },
  },
  "verdict": "MADAROS_GUM_GAP",
  "claim": "lean_single carries β¹⁰/β¹¹ FO GUM for Fano closed form; Madaros default fails Fano FO (value/variance path). Product-cancel may pass on Madaros without proving FO product correctness.",
  "next": "Port CHANNEL_INPUT_VAR + FO sens rebuild into Madaros IR variance binding (self-hosted/ir/lower.sio + native), not lean_single-only.",
}
# refine with explicit section parse
rec["tests"]["product_cancel"]["madaros"] = "PASS" if section_has("madaros", "product_cancel", "BETA10_PRODUCT_CANCEL_PASS") else "FAIL"
rec["tests"]["product_cancel"]["lean_single"] = "PASS" if section_has("lean_single", "product_cancel", "BETA10_PRODUCT_CANCEL_PASS") else "FAIL"
rec["tests"]["fano_fo_64sig2"]["madaros"] = "PASS" if section_has("madaros", "fano_associator", "BETA10_FANO_FO_VAR_PASS") else "FAIL_HONEST"
rec["tests"]["fano_fo_64sig2"]["lean_single"] = "PASS" if section_has("lean_single", "fano_associator", "BETA10_FANO_FO_VAR_PASS") else "FAIL_HONEST"
if rec["tests"]["fano_fo_64sig2"]["lean_single"] == "PASS" and rec["tests"]["fano_fo_64sig2"]["madaros"] != "PASS":
    rec["verdict"] = "MADAROS_GUM_GAP"
else:
    rec["verdict"] = "UNEXPECTED"
Path("results/deep_four/l1_madaros_gum_gap.receipt.v1.json").write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
PY
echo "L1_COMPLETE"
