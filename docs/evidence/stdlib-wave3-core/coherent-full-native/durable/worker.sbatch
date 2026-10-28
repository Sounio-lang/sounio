#!/bin/bash
#SBATCH --job-name=stdlib-coherent-full-durable
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-full-durable01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-full-durable01/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-full-durable01
C=/scratch/sounio-stdlib-wave3-20261027/core
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/coherent-text01
A="$D/full-native-source-receipts-19278-19285.tar.gz"
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
test "$(cat "$C/coherent-text-full02/results/WORKER_STATUS")" = 0
grep -Fx COHERENT_TEXT_FULL_NATIVE_ONLY_OK "$C/coherent-text-full02/results/VERDICT"
test "$(cat "$C/coherent-text-full02/results/PRISTINE_STRICT_STATUS")" = 1
test ! -e "$A"
mkdir -p "$D"
/usr/bin/time -v tar -C "$C" -czf "$A" coherent-text-full01 coherent-text-full02 > "$R/results/archive.stdout" 2> "$R/results/archive.stderr"
sha256sum "$A" > "$R/results/ARCHIVE.sha256"
cp "$R/results/ARCHIVE.sha256" "$D/FULL_NATIVE_ARCHIVES.sha256"
tar -tzf "$A" > "$R/results/CONTENTS.txt"
sha256sum -c "$R/results/ARCHIVE.sha256" > "$R/results/ARCHIVE.final.verify"
cp "$R/worker.sbatch" "$D/full-native-durable-worker.sbatch"
cp "$R/results/ARCHIVE.final.verify" "$D/full-native-archive.final.verify"
cp "$C/coherent-text-full02/results/SACCT.tsv" "$D/full-native-sacct-19285.tsv"
echo COHERENT_FULL_DURABLE_ARCHIVE_OK
