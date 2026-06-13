#!/usr/bin/env bash
# slurm-jobs/erdos90/test_subset_array.sh — smoke gate for subset cluster launcher
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SRC="$ROOT/stdlib/research/erdos90_subset_cluster.sio"
sed -e 's/CLUSTER_SEED: i64 = 1234567/CLUSTER_SEED: i64 = 424242/' \
    -e 's/run_one(18, TARGET_NSQ, TARGET_N, 40, 800000)/run_one(18, TARGET_NSQ, TARGET_N, 2, 5000)/' \
    -e 's/run_one(20, TARGET_NSQ, TARGET_N, 60, 1200000)/run_one(20, TARGET_NSQ, TARGET_N, 2, 8000)/' \
    -e 's/run_one(22, TARGET_NSQ, TARGET_N, 60, 1200000)/run_one(22, TARGET_NSQ, TARGET_N, 1, 3000)/' \
    "$SRC" >"$WORK/smoke.sio"

"$ROOT/bin/souc" "$WORK/smoke.sio" "$WORK/smoke.elf" >/dev/null
chmod +x "$WORK/smoke.elf"
"$WORK/smoke.elf" | tee /tmp/erdos90_subset_smoke.log

grep -q "CLUSTER_DONE" /tmp/erdos90_subset_smoke.log
grep -q "seed=424242" /tmp/erdos90_subset_smoke.log
grep -q "BEST_N100" /tmp/erdos90_subset_smoke.log

echo "[test_erdos90_subset_array] PASS"