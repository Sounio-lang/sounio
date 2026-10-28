#!/bin/bash
#SBATCH --job-name=stdlib-worker-metadata-export
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-metadata-export01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-metadata-export01/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
OUT=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/worker-metadata-discard-export01
[[ ! -e "$OUT" && -d /orangefs/training ]]
mkdir -p "$OUT"
trap 'r=$?;echo "$r" > "$OUT/EXPORT_STATUS";exit "$r"' EXIT
SEED=$B/worker-portable-gate-v3/input/bin/souc-lean-single-x86_64
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated01/results/madaros
sha256sum "$SEED" "$RAW" > "$OUT/SEEDS.before.sha256"
for record in worker-stable-metadata07 env-discard-checker01 env-discard-controls01 env-borrow-state01;do
 cd "$B/$record"
 find . -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/$record.SOURCE.before.sha256"
 tar -czf "$OUT/$record.tar.gz" .
 tar -tzf "$OUT/$record.tar.gz" > "$OUT/$record.CONTENTS.txt"
 sha256sum -c "$OUT/$record.SOURCE.before.sha256" > "$OUT/$record.SOURCE.after.verify"
done
# Diagnostic failed layout: retain actual bytes and the false 64-byte assumption.
cd "$B/worker-stable-metadata06"
tar -czf "$OUT/layout-initial-diagnostic.tar.gz" inputs results
 tar -tzf "$OUT/layout-initial-diagnostic.tar.gz" > "$OUT/layout-initial-diagnostic.CONTENTS.txt"
tar -czf "$OUT/compiler-seeds.tar.gz" -C / "${SEED#/}" "${RAW#/}"
tar -tzf "$OUT/compiler-seeds.tar.gz" > "$OUT/compiler-seeds.CONTENTS.txt"
sha256sum -c "$OUT/SEEDS.before.sha256" > "$OUT/SEEDS.after.verify"
sacct -X -j 19473,19477,19482,19484 --format=JobID,JobName,State,ExitCode,Elapsed,WorkDir -P > "$OUT/QUALIFICATION_SLURM.txt"
scontrol show job "$SLURM_JOB_ID" > "$OUT/EXPORT_SLURM.txt"
cd "$OUT"
sha256sum ./*.tar.gz ./*.CONTENTS.txt ./*.SOURCE.before.sha256 SEEDS.before.sha256 > SHA256SUMS
sha256sum -c SHA256SUMS > ARCHIVES.final.verify
echo WORKER_METADATA_DISCARD_EXPORT_OK
