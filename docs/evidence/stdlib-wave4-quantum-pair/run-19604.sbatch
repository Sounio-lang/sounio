#!/bin/bash
#SBATCH --job-name=quantum-original-mode0-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/quantum-mode0-custody41/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/quantum-mode0-custody41/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/quantum-mode0-custody41;D=$B/services/quantum-slow-mode0-35/results
mkdir "$R/archive" "$R/metadata"
trap 'r=$?;echo "$r" >"$R/metadata/STATUS";exit "$r"' EXIT
[[ $(cat "$D/STATUS") == 0 && $(cat "$D/NATIVE_STATUS") == 0 ]]
grep -Fxq ALL_PASS "$D/native.stdout"
for n in SOURCE COMPILER EXECUTABLE WITNESS;do sha256sum -c "$D/$n.sha256" >"$R/metadata/$n.verify";done
sacct -j19576 --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS,NodeList -P >"$R/metadata/ACCOUNTING"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$R/metadata/SNAPSHOT_TIME"
printf '%s\n' 'Original slow workload61B completed mode0 only, raw8abe, quantum328, no dyadic/hoist, original3000s deadline. Mode1 histories19524/19533 remain124. No fullstdlib or physicalQPU acceptance.' >"$R/metadata/SCOPE"
cd "$B"
tar -czf "$R/archive/quantum-original-mode0.tar.gz" services/quantum-slow-mode0-35 root-integrated02/results/madaros root-integrated02/RAW_READY
sha256sum "$R/archive/quantum-original-mode0.tar.gz" >"$R/metadata/ARCHIVE.sha256"
stat -c%s "$R/archive/quantum-original-mode0.tar.gz" >"$R/metadata/ARCHIVE.bytes"
tar -tzf "$R/archive/quantum-original-mode0.tar.gz" >"$R/metadata/CONTENTS.list"
for n in SOURCE COMPILER EXECUTABLE WITNESS;do sha256sum -c "$D/$n.sha256" >"$R/metadata/$n.final.verify";done
echo ORIGINAL_QUANTUM_MODE0_CUSTODY_OK
