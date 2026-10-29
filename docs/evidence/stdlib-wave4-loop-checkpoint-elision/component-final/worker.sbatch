#!/bin/bash
#SBATCH --job-name=loop-checkpoint-component
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/loop-checkpoint09/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/loop-checkpoint09/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R="$B/compiler-resumed/loop-checkpoint09"
S="$R/source"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
export PATH="$P/tools:$P/tooling/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$P/tooling/root/usr/lib/x86_64-linux-gnu"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
RAW="$B/root-integrated02/results/madaros"
ulimit -s524288;ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
cd "$S"
find self-hosted tests/fixtures/wave4_loop_checkpoint_elision -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/SOURCE.sha256"
printf '8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932  %s\n' "$RAW" > "$R/results/RAW.sha256"
sha256sum -c "$R/results/RAW.sha256" > "$R/results/RAW.before.verify"
SEED="$B/compiler-resumed/generic-module-native/source/bin/souc-lean-single-x86_64"
printf '9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a  %s\n' "$SEED" > "$R/results/SEED.sha256"
sha256sum -c "$R/results/SEED.sha256" > "$R/results/SEED.before.verify"
bash scripts/compiler/loop_checkpoint_elision_gate.sh "$S" "$RAW" "$SEED" "$R/results"
sha256sum -c "$R/results/SOURCE.sha256" > "$R/results/SOURCE.final.verify"
