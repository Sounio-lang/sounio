#!/bin/bash
#SBATCH --job-name=qpu-dyadic-p00-numeric-raw03
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/qpu-hoist55/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/qpu-hoist55/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated04
source "$R/tools-env-v3.sh"
JOB_ROOT=$B/services/qpu-hoist55
export NODE_BIN=$B/services-tools-recovered/tools/node
export SOUNIO_LOOP_CHECKPOINT_ELISION=0
RAW=$B/root-integrated03/results/madaros
mkdir "$JOB_ROOT/receipts"
finish(){ r=$?;trap - EXIT;sha256sum -c "$JOB_ROOT/receipts/TOOLCHAIN.sha256" >"$JOB_ROOT/receipts/TOOLCHAIN.final.verify" 2>&1||r=1;echo "$r" >"$JOB_ROOT/receipts/STATUS";exit "$r";};trap finish EXIT
(find "$R/private-tools-v3" "$R/pthread-installed-v3" "$R/private-runtime" "$G" -type f -print0;printf '%s\0' "$R/tools-env-v3.sh" "$NODE_BIN" "$RAW")|sort -zu|xargs -0 sha256sum >"$JOB_ROOT/receipts/TOOLCHAIN.sha256"
printf 'e0279ff62e97e861415cd8b105b854ae645aa61865b33b1e2b91184d2c28bf54  %s\n' "$RAW" >"$JOB_ROOT/receipts/RAW.sha256"
sha256sum -c "$JOB_ROOT/receipts/RAW.sha256" >"$JOB_ROOT/receipts/RAW.verify"
bash "$JOB_ROOT/input/scripts/stdlib/wave4_qpu_hoist_gate.sh" "$JOB_ROOT/results" "$RAW"
