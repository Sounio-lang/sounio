#!/usr/bin/env bash
#SBATCH --job-name=erdos90-rr10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=erdos90_rr10000_%j.out
#
# examples/erdos/erdos90_optimize_rr10000.sbatch
#
# Run the extended Erdős #90 compact-disk sweep to rr=10000 on a cluster node.
# The source (stdlib/research/erdos90_optimize.sio) already uses run_sweep(50, 10000)
# and 32768-point arrays. Output is staged to a shared directory.
set -euo pipefail

ROOT="${SOUNIO_ROOT:-$PWD}"
SOUC="${SOUNIO_SOUC_BIN:-$ROOT/bin/souc-lean-single-x86_64}"
SRC="$ROOT/stdlib/research/erdos90_optimize.sio"

if [[ -d /orangefs/training ]]; then
  STAGE="/orangefs/training/erdos90_rr10000_${SLURM_JOB_ID:-local}"
else
  STAGE="${STAGE:-$PWD/artifacts/erdos90/rr10000}"
fi
mkdir -p "$STAGE"

ELF="$STAGE/erdos90_optimize_rr10000.elf"
LOG="$STAGE/optimize_rr10000.log"

[[ -x "$SOUC" ]] || { echo "error: SOUC not executable: $SOUC" >&2; exit 2; }

echo "[$(date -Iseconds)] build $SRC -> $ELF"
"$SOUC" "$SRC" "$ELF"
chmod +x "$ELF"

echo "[$(date -Iseconds)] run sweep rr=50..10000 -> $LOG"
"$ELF" > "$LOG" 2>&1
echo "[$(date -Iseconds)] done rc=$?"

# Extract summary.
MAX_COUNT="$(grep -oE 'count=[0-9]+' "$LOG" | cut -d= -f2 | sort -n | tail -1 || true)"
MAX_N="$(grep -B1 "count=$MAX_COUNT" "$LOG" | grep -oE 'n=[0-9]+' | tail -1 | cut -d= -f2 || true)"
MAX_RR="$(grep -oE 'rr=[0-9]+' "$LOG" | tail -1 | cut -d= -f2 || true)"
echo "SUMMARY n=$MAX_N count=$MAX_COUNT rr=$MAX_RR"
