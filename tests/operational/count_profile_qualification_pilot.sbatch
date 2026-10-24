#!/usr/bin/env bash
#SBATCH --job-name=count-profile-nested-pilot
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
# PREPARED ONLY. sbatch SCRIPT SOURCE OUTPUT COMPILER GENUINE_RECEIPT REMAINING_SECONDS
# Root must supply output/chdir/log paths outside frozen source and retain the
# original allocation ledger. A receipt is never constructed by this template.
set -euo pipefail
[[ $# == 5 ]] || exit 64
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
export MADAROS_RAW_BIN="$3" COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT="$4" COUNT_PROFILE_WALL_BUDGET_SECONDS="$5"
exec bash "$1/scripts/stdlib/count_profile_calibration.sh" "$1" "$2" pilot
