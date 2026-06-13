#!/usr/bin/env bash
# slurm-jobs/erdos90/run_on_cluster.sh — run a Sounio Erdős-[90] kernel on Slurm
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
KERNEL="${1:-$ROOT/stdlib/research/erdos90_search.sio}"
PART="${2:-cpu-ops}"
SOUC="${SOUC:-$ROOT/bin/souc}"
ELF="$(mktemp /tmp/e90_XXXX.elf)"

echo "[1/3] compiling $KERNEL -> static ELF with $SOUC"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
"$SOUC" "$KERNEL" "$ELF"
file "$ELF" | grep -q "statically linked" || { echo "ERROR: ELF not static"; exit 1; }

echo "[2/3] shipping $(stat -c%s "$ELF") bytes to '$PART' via srun"
base64 -w0 "$ELF" | srun --partition="$PART" --time=00:30:00 --chdir=/orangefs/training bash -c '
  base64 -d > /orangefs/training/e90_run && chmod +x /orangefs/training/e90_run
  echo "=== node $(hostname) ($(nproc) cores) ==="
  t0=$(date +%s); /orangefs/training/e90_run; echo "wall=$(( $(date +%s) - t0 ))s"
'
echo "[3/3] done — configs re-certifiable in Lean (countUnit / native_decide)"