#!/usr/bin/env bash
# slurm-jobs/erdos90/run_unified_array.sh — parallel unified ℚ(√3) array
#
# Usage:
#   bash run_unified_array.sh local          # compile+run all seeds locally
#   bash run_unified_array.sh cluster        # ship each seed ELF via srun
#   SEEDS="1 2 3" bash run_unified_array.sh local
#   SMOKE=1 bash run_unified_array.sh local  # tiny iters for wiring check
#
# Each task patches CLUSTER_SEED in a temp copy of erdos90_unified_cluster.sio,
# compiles a static ELF, and runs the heavy hill-climb sweep.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-local}"
PART="${2:-cpu-ops}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="$ROOT/stdlib/research/erdos90_unified_cluster.sio"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"

if [ -z "${SEEDS:-}" ]; then
    SEEDS="1000003 2000003 3000003 4000007 5000009 6000011 7000013 8000019 9000023"
fi

patch_src() {
    local seed="$1"
    local patched="$WORK/cluster_${seed}.sio"
    sed "s/CLUSTER_SEED: i64 = [0-9]*/CLUSTER_SEED: i64 = ${seed}/" "$SRC" >"$patched"
    if [ "${SMOKE:-0}" = "1" ]; then
        sed -i \
            -e 's/run_one(12, 12, 60, 15, 400000)/run_one(12, 12, 60, 2, 4000)/' \
            -e 's/run_one(12, 12, 100, 25, 600000)/run_one(12, 12, 100, 2, 8000)/' \
            -e 's/run_one(14, 14, 150, 20, 500000)/run_one(14, 14, 150, 1, 3000)/' \
            "$patched"
    fi
    echo "$patched"
}

compile_seed() {
    local seed="$1"
    local patched
    patched="$(patch_src "$seed")"
    local elf="$WORK/cluster_${seed}.elf"
    "$SOUC" "$patched" "$elf" >/dev/null
    chmod +x "$elf"
    echo "$elf"
}

run_local() {
    for seed in $SEEDS; do
        echo "--- local seed=$seed ---"
        elf="$(compile_seed "$seed")"
        "$elf" 2>&1 | grep -E "seed=|bestHC=|BEAT-HARB|harb-ok|CLUSTER_DONE|cross=" || true
    done
}

run_cluster() {
    local time_limit="${TIME_LIMIT:-01:30:00}"
    for seed in $SEEDS; do
        echo "--- cluster seed=$seed ---"
        elf="$(compile_seed "$seed")"
        base64 -w0 "$elf" | srun --partition="$PART" --time="$time_limit" --chdir=/orangefs/training bash -c "
          base64 -d > /orangefs/training/e90_uq_${seed} && chmod +x /orangefs/training/e90_uq_${seed}
          echo '=== seed ${seed} on '\$(hostname)' ==='
          t0=\$(date +%s)
          /orangefs/training/e90_uq_${seed}
          echo wall=\$(( \$(date +%s) - t0 ))s
        " || echo "WARN: srun failed for seed=$seed (cluster offline?)"
    done
}

case "$MODE" in
    local) run_local ;;
    cluster) run_cluster ;;
    *)
        echo "usage: $0 [local|cluster] [partition]" >&2
        exit 1
        ;;
esac

echo "[erdos90-unified-array] finished mode=$MODE smoke=${SMOKE:-0}"