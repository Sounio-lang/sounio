#!/usr/bin/env bash
#SBATCH --job-name=count-profile-nested-round
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=24:00:00
#SBATCH --export=NIL
# PREPARED ONLY. sbatch SCRIPT SOURCE OUTPUT COMPILER GENUINE_RECEIPT REMAINING_SECONDS
# REMAINING_SECONDS is the root-authorized remainder of the aggregate24h round,
# after pilots/prior invocations. This template does not reset or infer it.
set -euo pipefail
[[ $# == 5 ]] || exit 64
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
export MADAROS_RAW_BIN="$3" COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT="$4" COUNT_PROFILE_WALL_BUDGET_SECONDS="$5"
exec bash "$1/scripts/stdlib/count_profile_calibration.sh" "$1" "$2" qualification
