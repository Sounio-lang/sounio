#!/bin/bash
#SBATCH --job-name=stdlib-worker-live-schema
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:04:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-raw03-live-schema08/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-raw03-live-schema08/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
ulimit -s 524288;ulimit -c 0
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/compiler-resumed/worker-raw03-live-schema08
G=$B/root-integrated03-loader/gate
RAW=$B/root-integrated03/results/madaros
trap 'r=$?;echo "$r" > "$R/STATUS";exit "$r"' EXIT
sha256sum "$RAW" "$R/probe.cpp" "$G/cases/direct.sio" > "$R/INPUTS.sha256"
sha256sum -c "$B/root-integrated03/RAW_READY" > "$R/RAW.verify"
sha256sum -c "$G/results/SOURCE.sha256" > "$R/SOURCE.before.verify"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -pedantic "$R/probe.cpp" -o "$R/probe"
timeout 120 "$R/probe" "$RAW" "$G/cases/direct.sio" "$G/source/stdlib" "$R" > "$R/probe.stdout" 2> "$R/probe.stderr"
sha256sum -c "$R/INPUTS.sha256" > "$R/INPUTS.final.verify"
sha256sum -c "$G/results/SOURCE.sha256" > "$R/SOURCE.final.verify"
echo LIVE_RAW03_DESCRIPTOR_AND_IR_CAPTURED_DIAGNOSTIC_ONLY
