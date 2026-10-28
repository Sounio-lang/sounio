#!/bin/bash
#SBATCH --job-name=stdlib-coherent-focused-durable
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-archive01/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-archive01/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/core/coherent-text-archive01
B=/scratch/sounio-stdlib-wave3-20261027/core
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/coherent-text01
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
for job in coherent-runner-args01 coherent-text-focused01;do
 test "$(cat "$B/$job/results/WORKER_STATUS")" = 0
 sha256sum -c "$B/$job/results/SOURCE.sha256" > "$R/results/$job-SOURCE.final.verify"
done
test ! -e "$D"
mkdir "$D"
sacct -j19269,19270 -X --format=JobID,JobName,State,ExitCode,Elapsed -P > "$D/JOBS.tsv"
tar -C "$B" -czf "$D/focused-source-elf-argv-receipts-19269-19270.tar.gz" coherent-runner-args01 coherent-text-focused01
cp "$R/worker.sbatch" "$D/worker.sbatch"
sha256sum "$D/focused-source-elf-argv-receipts-19269-19270.tar.gz" "$D/worker.sbatch" "$D/JOBS.tsv" > "$D/ARCHIVES.sha256"
sha256sum -c "$D/ARCHIVES.sha256" > "$D/ARCHIVES.final.verify"
tar -xOzf "$D/focused-source-elf-argv-receipts-19269-19270.tar.gz" coherent-text-focused01/coherent.elf | sha256sum > "$D/ARCHIVED_ELF.sha256"
sha256sum "$B/coherent-text-focused01/coherent.elf" > "$D/LIVE_ELF.sha256"
cut -d' ' -f1 "$D/ARCHIVED_ELF.sha256" > "$R/results/archived-digest"
cut -d' ' -f1 "$D/LIVE_ELF.sha256" > "$R/results/live-digest"
cmp "$R/results/archived-digest" "$R/results/live-digest"
echo COHERENT_FOCUSED_DURABLE_OK
