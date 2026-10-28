#!/bin/bash
#SBATCH --job-name=quantum-fast-worker-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=512M
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/quantum-custody29/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/quantum-custody29/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/quantum-custody29
mkdir "$R/snapshot" "$R/archive" "$R/metadata"
trap 'r=$?;echo "$r" > "$R/metadata/STATUS";exit "$r"' EXIT
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$R/metadata/SNAPSHOT_TIME"
sacct -j 19508,19513,19515,19516,19518,19519,19521,19524,19525,19526,19527,19530,19533 --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS,NodeList -P > "$R/metadata/ACCOUNTING"
printf '%s\n' 'FAST RECEIPTS ONLY. 19524 and19533 slow native results are pending. Mutable slow stdout is excluded. No long-job completion or final guard inferred.' > "$R/metadata/SCOPE"
for group in quantum-pair-library25 quantum-pair-library28;do
 for n in INPUTS COMPILER SOURCES WITNESSES EXECUTABLES;do
  sha256sum -c "$B/services/$group/results/$n.sha256" > "$R/metadata/$group.$n.verify"
 done
done
for n in INPUTS COMPILER EXECUTABLES SDK LIBRARIES;do sha256sum -c "$B/services/quantum-observable27/results/$n.sha256" > "$R/metadata/observable.$n.verify";done
cd "$B"
tar --exclude='*/after.overtake_slow.native.stdout' -cf -  services/quantum-pair18 services/quantum-pair-trace19 services/quantum-pair20 services/quantum-pair21 services/quantum-pair22 services/quantum-pair23 services/quantum-pair24 services/quantum-pair-library25 services/quantum-pair26 services/quantum-observable27 services/quantum-pair-library28  root-integrated02/results/madaros root-integrated02/RAW_READY |tar -C "$R/snapshot" -xf -
cp -a "$R/input" "$R/snapshot/custody-input"
find "$R/snapshot" -type f -print0|sort -z|xargs -0 sha256sum > "$R/metadata/CONTENTS.sha256"
tar -czf "$R/archive/quantum-worker-fast.tar.gz" -C "$R" snapshot metadata
gzip -t "$R/archive/quantum-worker-fast.tar.gz"
sha256sum "$R/archive/quantum-worker-fast.tar.gz" > "$R/metadata/ARCHIVE.sha256"
stat -c '%s' "$R/archive/quantum-worker-fast.tar.gz" > "$R/metadata/ARCHIVE.bytes"
sha256sum -c "$R/metadata/CONTENTS.sha256" > "$R/metadata/CONTENTS.final.verify"
echo QUANTUM_FAST_AND_WORKER_COMPONENT_CUSTODY_OK
