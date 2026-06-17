#!/usr/bin/env bash
# slurm-jobs/erdos90/submit_subset1600_array.sh — n=1600 subset array (N=25)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="${SRC:-$ROOT/stdlib/research/erdos90_subset1600_cluster.sio}"
RUN_ID="${RUN_ID:-erdos90-1600-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="${STAGE_ROOT:-/orangefs/training/sounio/erdos90-1600-runs/${RUN_ID}}"
PARTITION="${PARTITION:-cpu-ops}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
ARRAY_MAX="${ARRAY_MAX:-17}"

SEEDS=(
    1000003 2000003 3000003 4000007 5000009
    6000011 7000013 8000019 9000023
    314159 2711688 161803 141421 173205
    223607 3141592 577215 707106
)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"

echo "run id:      ${RUN_ID}"
echo "stage root:  ${STAGE_ROOT}"
echo "target:      n=1600 N=25 subset (beat grid40x40=8128)"

for i in $(seq 0 "$ARRAY_MAX"); do
    seed="${SEEDS[$i]}"
    patched="$WORK/cluster_${seed}.sio"
    elf="$WORK/cluster_${seed}.elf"
    sed "s/CLUSTER_SEED: i64 = [0-9]*/CLUSTER_SEED: i64 = ${seed}/" "$SRC" >"$patched"
    "$SOUC" "$patched" "$elf" >/dev/null
    chmod +x "$elf"
done

srun --partition="$PARTITION" --time=00:10:00 --chdir=/orangefs/training bash -c "
  mkdir -p '${STAGE_ROOT}/bin' '${STAGE_ROOT}/results' '${STAGE_ROOT}/logs'
"

for i in $(seq 0 "$ARRAY_MAX"); do
    seed="${SEEDS[$i]}"
    elf="$WORK/cluster_${seed}.elf"
    base64 -w0 "$elf" | srun --partition="$PARTITION" --time=00:10:00 --chdir=/orangefs/training bash -c "
      base64 -d > '${STAGE_ROOT}/bin/e90_1600_${seed}'
      chmod +x '${STAGE_ROOT}/bin/e90_1600_${seed}'
    "
done

SEED_LIST=""
for i in $(seq 0 "$ARRAY_MAX"); do SEED_LIST+="${SEEDS[$i]} "; done

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
BIN="${STAGE_ROOT}/bin/e90_1600_\${SEED}"
OUT="${STAGE_ROOT}/results/\${SEED}.log"
"\$BIN" | tee "\$OUT"
grep -E 'BEST_N1600|BEAT-GRID8128|CLUSTER1600_DONE' "\$OUT" || true
EOF

JOB_ID="$(sbatch --parsable "$SBATCH_FILE")"
echo "submitted: job ${JOB_ID}"
echo "monitor:   squeue -j ${JOB_ID}"
echo "leader:    grep -h BEST_N1600 ${STAGE_ROOT}/results/*.log | sort -t= -k3 -nr | head"