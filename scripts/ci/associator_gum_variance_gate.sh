#!/usr/bin/env bash
# Gate: associator GUM variance experiment must COMPLETE with PASS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
bash experiments/associator_gum_variance/run_and_receipt.sh
LOG=results/associator_gum_variance/RUNLOG.txt
RECEIPT=results/associator_gum_variance/receipt.v1.json
grep -q "ASSOC_GUM_EXPERIMENT_COMPLETE" "$LOG"
VERDICT=$(python3 -c "import json;print(json.load(open('$RECEIPT'))['verdict'])")
if [[ "$VERDICT" != "PASS" ]]; then
  echo "associator_gum_variance_gate: FAIL (verdict=$VERDICT)" >&2
  exit 1
fi
echo "associator_gum_variance_gate: PASS"
