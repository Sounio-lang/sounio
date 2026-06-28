#!/usr/bin/env bash
# slurm-jobs/erdos90/run_on_cluster.sh — run a Sounio Erdős-[90] search kernel on the
# Slurm cluster, WITHOUT the BeagleCockpit MCP (direct srun fallback).
#
# Why this exists: the compute nodes do NOT mount /workspace. The only shared
# filesystem is OrangeFS at /orangefs/training (pvfs2), and the login container
# cannot see it either. But srun streams stdout back to the caller and the nodes
# run as the caller's uid, so we: (1) compile the kernel to a STATIC, self-contained
# ELF locally (no libc — the nodes have no Sounio runtime), (2) ship the ELF bytes to
# a node over srun stdin as base64, (3) decode + execute on the node's /orangefs/training,
# (4) collect results on stdout. This keeps the actual Sounio binary running on cluster
# hardware. The canonical path is BeagleCockpit MCP (cockpit_submit_job) with the
# beagle-sounio image — use it when those tools are loaded; this is the no-MCP fallback.
#
# Usage:  bash run_on_cluster.sh [search_kernel.sio] [partition]
# Default kernel: stdlib/research/erdos90_search.sio   Default partition: cpu-ops
#
# Requires a WORKING dev souc (the main-branch souc rejects the kernel on effect
# decls). This script uses /workspace/sounio/bin/souc if present, else ./bin/souc.

set -euo pipefail
KERNEL="${1:-stdlib/research/erdos90_search.sio}"
PART="${2:-cpu-ops}"
SOUC="/workspace/sounio/bin/souc"; [ -x "$SOUC" ] || SOUC="./bin/souc"
ELF="$(mktemp /tmp/e90_XXXX)"

echo "[1/3] compiling $KERNEL -> static ELF with $SOUC"
"$SOUC" compile "$KERNEL" -o "$ELF"
file "$ELF" | grep -q "statically linked" || { echo "ERROR: ELF not static"; exit 1; }

echo "[2/3] shipping $(stat -c%s "$ELF") bytes to a '$PART' node via srun + running"
base64 -w0 "$ELF" | srun --partition="$PART" --time=00:10:00 --chdir=/orangefs/training bash -c '
  base64 -d > /orangefs/training/e90_run && chmod +x /orangefs/training/e90_run
  echo "=== node $(hostname) ($(nproc) cores) running Sounio Erdős-[90] search ==="
  t0=$(date +%s); /orangefs/training/e90_run; echo "wall=$(( $(date +%s) - t0 ))s"
'
echo "[3/3] done. Any config beating the baseline is re-certifiable in Lean"
echo "      (formal/lean4/SounioErdos90PlanarLowerBound.lean, countUnit / native_decide)."
