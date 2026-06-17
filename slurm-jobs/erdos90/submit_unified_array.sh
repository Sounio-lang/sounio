#!/usr/bin/env bash
# slurm-jobs/erdos90/submit_unified_array.sh — sbatch array for heavy unified ℚ(√3) sweep
#
# Usage:
#   bash slurm-jobs/erdos90/submit_unified_array.sh
#   ARRAY_MAX=8 bash slurm-jobs/erdos90/submit_unified_array.sh   # first 9 seeds only
#
# Output:
#   /orangefs/training/sounio/erdos90-uq-runs/<RUN_ID>/results/<seed>.log
#   /orangefs/training/sounio/erdos90-uq-runs/<RUN_ID>/logs/array-%A_%a.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="${SRC:-$ROOT/stdlib/research/erdos90_unified_cluster_heavy.sio}"
RUN_ID="${RUN_ID:-erdos90-uq-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="${STAGE_ROOT:-/orangefs/training/sounio/erdos90-uq-runs/${RUN_ID}}"
PARTITION="${PARTITION:-cpu-ops}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
ARRAY_MAX="${ARRAY_MAX:-17}"

SEEDS=(
    1000003 2000003 3000003 4000007 5000009
    6000011 7000013 8000019 9000023
    314159 271828 161803 141421 173205
    223607 3141592 577215 707106
)

if [ ! -f "$SRC" ] || [ ! -x "$SOUC" ]; then
    echo "missing source or souc: $SRC $SOUC" >&2
    exit 2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"

echo "run id:      ${RUN_ID}"
echo "stage root:  ${STAGE_ROOT}"
echo "partition:   ${PARTITION}"
echo "array:       0-${ARRAY_MAX}"
echo "time limit:  ${TIME_LIMIT}"

echo
echo "[1/4] compile heavy ELFs locally ..."
for i in $(seq 0 "$ARRAY_MAX"); do
    seed="${SEEDS[$i]}"
    patched="$WORK/cluster_${seed}.sio"
    elf="$WORK/cluster_${seed}.elf"
    sed "s/CLUSTER_SEED: i64 = [0-9]*/CLUSTER_SEED: i64 = ${seed}/" "$SRC" >"$patched"
    "$SOUC" "$patched" "$elf" >/dev/null
    chmod +x "$elf"
    echo "  compiled seed=$seed ($(stat -c%s "$elf") bytes)"
done

echo
echo "[2/4] stage ELFs to ${STAGE_ROOT}/bin via srun ..."
srun --partition="$PARTITION" --time=00:10:00 --chdir=/orangefs/training bash -c "
  mkdir -p '${STAGE_ROOT}/bin' '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs'
"

for i in $(seq 0 "$ARRAY_MAX"); do
    seed="${SEEDS[$i]}"
    elf="$WORK/cluster_${seed}.elf"
    base64 -w0 "$elf" | srun --partition="$PARTITION" --time=00:10:00 --chdir=/orangefs/training bash -c "
      base64 -d > '${STAGE_ROOT}/bin/e90_uq_${seed}'
      chmod +x '${STAGE_ROOT}/bin/e90_uq_${seed}'
    "
done

echo
echo "[3/4] submit sbatch array ..."
SEED_LIST=""
for i in $(seq 0 "$ARRAY_MAX"); do
    SEED_LIST+="${SEEDS[$i]} "
done

SBATCH_FILE="$WORK/job.sbatch"
cat >"$SBATCH_FILE" <<EOF
#!/bin/bash
#SBATCH -J ${RUN_ID}
#SBATCH -p ${PARTITION}
#SBATCH --array=0-${ARRAY_MAX}
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${STAGE_ROOT}/logs/array-%A_%a.log

set -euo pipefail
SEEDS=(${SEED_LIST})
SEED="\${SEEDS[\$SLURM_ARRAY_TASK_ID]}"
BIN="${STAGE_ROOT}/bin/e90_uq_\${SEED}"
OUT="${STAGE_ROOT}/results/\${SEED}.log"

echo "=== Erdos90 unified heavy seed=\${SEED} on \$(hostname) task=\${SLURM_ARRAY_TASK_ID} ==="
echo "start=\$(date -Is)"
t0=\$(date +%s)
"\$BIN" | tee "\$OUT"
echo "wall=\$(( \$(date +%s) - t0 ))s end=\$(date -Is)"
grep -E 'BEST_N100|BEAT-HARB|CLUSTER_DONE' "\$OUT" || true
EOF

JOB_ID="$(sbatch --parsable "$SBATCH_FILE")"
echo "  submitted: job ${JOB_ID}"

echo
echo "[4/4] summary"
cat <<SUM
  RUN_ID:     ${RUN_ID}
  JOB_ID:     ${JOB_ID}
  Stage:      ${STAGE_ROOT}
  Monitor:    squeue -j ${JOB_ID}
  Logs:       ${STAGE_ROOT}/logs/array-${JOB_ID}_<task>.log
  Results:    ${STAGE_ROOT}/results/<seed>.log
  Aggregate:  srun --partition=${PARTITION} --time=00:05:00 bash -lc "grep -h BEST_N100 ${STAGE_ROOT}/results/*.log 2>/dev/null | sort -t= -k3 -nr | head"
SUM