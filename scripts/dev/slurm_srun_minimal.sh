#!/usr/bin/env bash
# Minimal WORKING Slurm launch for the Beagle pilot from the workspace login.
# Proven 2026-08-17: srun COMPLETED (job 10113) with receipt on /orangefs.
#
# Supported path today: srun (this script). sbatch is NOT repaired —
# user_env_retrieval_failed is an admin/controller issue for openvscode-server;
# lanes cannot fix it. See docs/ops/SLURM_LAUNCH_REPAIR_2026-08-17.md.
#
# Usage:
#   scripts/dev/slurm_srun_minimal.sh 'echo hello'
#   scripts/dev/slurm_srun_minimal.sh --time=00:30:00 --partition=all -- 'your command'
set -euo pipefail

export SLURM_CONF="${SLURM_CONF:-/tmp/slurm-direct.conf}"

PARTITION=cpu-ops
TIME=00:10:00
NODES=1
NTASKS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition=*) PARTITION="${1#*=}"; shift ;;
    --time=*) TIME="${1#*=}"; shift ;;
    --nodes=*) NODES="${1#*=}"; shift ;;
    --ntasks=*) NTASKS="${1#*=}"; shift ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "usage: $0 [--partition=cpu-ops] [--time=00:10:00] -- <command...>" >&2
  exit 2
fi

# Absolute interpreter; scrub workspace env; chdir off /workspace.
exec srun \
  --partition="$PARTITION" \
  --nodes="$NODES" \
  --ntasks="$NTASKS" \
  --time="$TIME" \
  --chdir=/tmp \
  --export=NONE,PATH=/usr/bin:/bin:/usr/local/bin,TMPDIR=/tmp,TMP=/tmp,TEMP=/tmp,HOME=/tmp \
  /bin/bash -lc "$*"
