#!/bin/bash
#SBATCH --job-name=stdlib-public-adapter-durable
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/coherent-public-archive01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/coherent-public-archive01/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/core/coherent-public-archive01
B=/scratch/sounio-stdlib-wave3-20261027/core
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/coherent-text01
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
test "$(cat "$B/coherent-public-gate02/results/WORKER_STATUS")" = 0
sha256sum -c "$D/ARCHIVES.sha256" > "$R/results/EXISTING_ARCHIVES.verify"
sha256sum -c "$B/coherent-public-gate02/results/INPUTS.sha256" > "$R/results/INPUTS.final.verify"
sha256sum -c "$B/coherent-public-gate02/results/LEGACY_SOURCE.sha256" > "$R/results/LEGACY_SOURCE.final.verify"
test ! -e "$D/public-adapter-source-receipts-19289-19291.tar.gz"
tar -C "$B" -czf "$D/public-adapter-source-receipts-19289-19291.tar.gz" coherent-public-gate01 coherent-public-gate02 coherent-text-full01
sacct -j19278,19289,19291 -X --format=JobID,JobName,State,ExitCode,Elapsed -P > "$D/PUBLIC_JOBS.tsv"
cp "$R/worker.sbatch" "$D/public-archive-worker.sbatch"
sha256sum "$D/public-adapter-source-receipts-19289-19291.tar.gz" "$D/PUBLIC_JOBS.tsv" "$D/public-archive-worker.sbatch" > "$D/PUBLIC_ARCHIVES.sha256"
sha256sum -c "$D/PUBLIC_ARCHIVES.sha256" > "$D/PUBLIC_ARCHIVES.final.verify"
tar -xOzf "$D/public-adapter-source-receipts-19289-19291.tar.gz" coherent-public-gate02/input/scripts/stdlib/wave3_core_gate.sh | sha256sum > "$D/PUBLIC_ARCHIVED_GATE.sha256"
sha256sum "$B/coherent-public-gate02/input/scripts/stdlib/wave3_core_gate.sh" > "$D/PUBLIC_LIVE_GATE.sha256"
cut -d' ' -f1 "$D/PUBLIC_ARCHIVED_GATE.sha256" > "$R/results/archived-digest"
cut -d' ' -f1 "$D/PUBLIC_LIVE_GATE.sha256" > "$R/results/live-digest"
cmp "$R/results/archived-digest" "$R/results/live-digest"
echo COHERENT_PUBLIC_ADAPTER_DURABLE_OK
