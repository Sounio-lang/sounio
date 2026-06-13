#!/usr/bin/env bash
# slurm-jobs/erdos90/test_unified_array.sh — wiring gate for unified Slurm array
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SEEDS="424242" SMOKE=1 bash slurm-jobs/erdos90/run_unified_array.sh local | tee /tmp/erdos90_unified_array_smoke.log

grep -q "CLUSTER_DONE" /tmp/erdos90_unified_array_smoke.log
grep -q "seed=424242" /tmp/erdos90_unified_array_smoke.log
grep -q "bestHC=" /tmp/erdos90_unified_array_smoke.log

echo "[test_erdos90_unified_array] PASS"