#!/usr/bin/env bash
# Aggregate gate for deep-four lanes L1–L4.
# L1: Madaros GUM gap receipt exists with explicit verdict
# L2: multi-component associator PASS
# L3: demog residual COMPLETE
# L4: fixed-point receipt exists (PASS or FAIL_HONEST both acceptable if complete)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
OUT=results/deep_four
mkdir -p "$OUT"

bash experiments/deep_four/l1_madaros_gum_gap.sh
./bin/souc run experiments/deep_four/l2_octonion_associator_multicomponent.sio | tee "$OUT/l2.RUNLOG.txt"
grep -q "L2_OCTONION_ASSOC_MULTI_PASS" "$OUT/l2.RUNLOG.txt"
./bin/souc run experiments/deep_four/l3_faers_demographics_residual.sio | tee "$OUT/l3.RUNLOG.txt"
grep -q "L3_FAERS_DEMOG_COMPLETE" "$OUT/l3.RUNLOG.txt"

# L4 is heavy — run only if DEEP_FOUR_FIXEDPOINT=1
if [[ "${DEEP_FOUR_FIXEDPOINT:-0}" == "1" ]]; then
  bash experiments/deep_four/l4_lean_single_fixedpoint.sh
  test -f "$OUT/l4_fixedpoint.receipt.v1.json"
else
  echo "L4 skipped (set DEEP_FOUR_FIXEDPOINT=1 to run self-compile fixed point)"
fi

test -f "$OUT/l1_madaros_gum_gap.receipt.v1.json"
echo "deep_four_lane_gate: PASS (L1–L3; L4 optional)"
