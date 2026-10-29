#!/bin/bash
#SBATCH --job-name=sig139-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-delivery01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-delivery01/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D="$B/compiler-resumed/sig139-delivery01"
R="$B/root-integrated04"
source "$R/tools-env-v2.sh"
trap 'r=$?;printf "%s\n" "$r" > "$D/WORKER_STATUS";date -u > "$D/ENDED_UTC";exit "$r"' EXIT
node "$D/collect.mjs" "$D/receipts"
sacct -X -j 19677,19681,19690,19695,19704,19716,19720 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,WorkDir -P > "$D/receipts/SACCT.txt"
cp "$R/tools-env-v2.sh" "$D/receipts/tools-env-v2.sh"
mkdir "$D/archive"
cp -a "$D/receipts" "$D/archive/"
for name in sig139-census01 sig139-census02 sig139-census03 sig139-census04 sig139-census05 sig139-packet01 sig139-packet02;do
 mkdir "$D/archive/$name"
 cp -a "$B/compiler-resumed/$name/input" "$B/compiler-resumed/$name/results" "$B/compiler-resumed/$name/worker.sbatch" "$D/archive/$name/"
done
cp "$B/root-integrated03/results/madaros" "$D/archive/raw03"
cp "$B/compiler-resumed/generic-module-native/source/bin/souc-lean-single-x86_64" "$D/archive/seed9cf"
cp -a "$B/compiler-resumed/sig139-census01/source" "$D/archive/original-sdk"
find "$D/archive" -type f -print0|sort -z|xargs -0 sha256sum > "$D/ARCHIVE_INPUTS.sha256"
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/sig139-census01
mkdir -p "$O"
A="$O/source-raw-seed-tracers-matrices-receipts.tar.gz"
test ! -e "$A"
tar -czf "$A" -C "$D" archive
sha256sum "$A" > "$D/ARCHIVE.sha256"
sha256sum -c "$D/ARCHIVE.sha256" > "$D/ARCHIVE.readback.verify"
tar -tzf "$A" > "$D/ARCHIVE.LIST"
sha256sum -c "$D/ARCHIVE_INPUTS.sha256" > "$D/ARCHIVE_INPUTS.final.verify"
cp "$D/ARCHIVE.sha256" "$D/ARCHIVE.readback.verify" "$D/ARCHIVE.LIST" "$O/"
printf 'SIG139_CUSTODY_READBACK_OK\n' > "$D/STATUS"
