#!/usr/bin/env bash
# slurm-jobs/erdos90/run_eisenstein_array.sh — parallel Eisenstein subset array
#
# Usage:
#   bash run_eisenstein_array.sh local          # compile+run all seeds locally
#   bash run_eisenstein_array.sh cluster        # ship each seed ELF via srun
#   SEEDS="1 2 3" bash run_eisenstein_array.sh local
#
# Each task patches CLUSTER_SEED in a temp copy of erdos90_eisenstein_cluster.sio,
# compiles a static ELF, and runs the heavy hill-climb sweep.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-local}"
PART="${2:-cpu-ops}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="$ROOT/stdlib/research/erdos90_eisenstein_cluster.sio"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"

if [ -z "${SEEDS:-}" ]; then
    SEEDS="1000003 2000003 3000003 4000007 5000009 6000011 7000013 8000019 9000023"
fi

compile_seed() {
    local seed="$1"
    local patched="$WORK/cluster_${seed}.sio"
    local elf="$WORK/cluster_${seed}.elf"
    sed "s/CLUSTER_SEED: i64 = [0-9]*/CLUSTER_SEED: i64 = ${seed}/" "$SRC" >"$patched"
    "$SOUC" "$patched" "$elf" >/dev/null
    chmod +x "$elf"
    echo "$elf"
}

run_local() {
    for seed in $SEEDS; do
        echo "--- local seed=$seed ---"
        elf="$(compile_seed "$seed")"
        "$elf" 2>&1 | grep -E "seed=|bestHC=|BEAT-HARB|harb-ok|CLUSTER_DONE" || true
    done
}

run_cluster() {
    for seed in $SEEDS; do
        echo "--- cluster seed=$seed ---"
        elf="$(compile_seed "$seed")"
        base64 -w0 "$elf" | srun --partition="$PART" --time=00:45:00 --chdir=/orangefs/training bash -c "
          base64 -d > /orangefs/training/e90_ei_${seed} && chmod +x /orangefs/training/e90_ei_${seed}
          echo '=== seed ${seed} on '\$(hostname)' ==='
          /orangefs/training/e90_ei_${seed}
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

echo "[erdos90-eisenstein-array] finished mode=$MODE"