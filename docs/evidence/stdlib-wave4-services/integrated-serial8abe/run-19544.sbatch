#!/bin/bash
#SBATCH --job-name=normal-integrated-durable-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/normal-replay-custody32/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/normal-replay-custody32/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/services/normal-replay-custody32
D=$B/services/normal-abort-integrated31/results
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/normal-integrated-export05
mkdir "$R/metadata"
trap 'r=$?;echo "$r" > "$R/metadata/STATUS";exit "$r"' EXIT
[[ $(cat "$D/STATUS") == 0 ]]
cd "$B/services/arrow-abort-committed02/src"
for n in SOURCES INPUTS EXECUTABLES ARTIFACTS INTEGRATED_COMPILER;do sha256sum -c "$D/$n.sha256" > "$R/metadata/$n.verify";done
for n in SOURCES TOOLS SDK_HEADERS EXECUTABLES DYNAMIC;do sha256sum -c "$B/services/arrow-abort02/results/$n.sha256" > "$R/metadata/DEPENDENCY_$n.verify";done
sha256sum -c "$B/services/arrow-abort-committed02/results/SOURCES.sha256" > "$R/metadata/COMMITTED_SOURCE.verify"
sacct -j 19539,19540 --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,ReqMem,NodeList -P > "$R/metadata/ACCOUNTING"
printf '%s\n' 'SERIAL EXTRACTED COMPONENT ONLY. No public WorkerFn or parallel pipeline acceptance. Both first39 harness failure and completed40 preserved.' > "$R/metadata/SCOPE"
cd "$B"
find services/normal-abort-integrated30 services/normal-abort-integrated31 services/normal-replay-custody32/input services/arrow-abort02 services/arrow-abort-committed02/src sdk/arrow26/root -type f -print0|sort -z|xargs -0 sha256sum > "$R/metadata/CONTENTS.sha256"
test ! -e "$O/normal-integrated.tar.gz"
tar -czf "$O/normal-integrated.tar.gz" services/normal-abort-integrated30 services/normal-abort-integrated31 services/normal-replay-custody32/input services/normal-replay-custody32/metadata services/arrow-abort02 services/arrow-abort-committed02/src sdk/arrow26/root root-integrated02/results/madaros root-integrated02/RAW_READY
gzip -t "$O/normal-integrated.tar.gz"
sha256sum "$O/normal-integrated.tar.gz" > "$R/metadata/ARCHIVE.sha256"
stat -c '%s' "$O/normal-integrated.tar.gz" > "$R/metadata/ARCHIVE.bytes"
sha256sum -c "$R/metadata/ARCHIVE.sha256" > "$R/metadata/ARCHIVE.readback.verify"
sha256sum -c "$R/metadata/CONTENTS.sha256" > "$R/metadata/CONTENTS.final.verify"
cp -a "$R/metadata" "$O/"
cp "$0" "$O/export.sbatch"
echo NORMAL_INTEGRATED_SERIAL_CUSTODY_OK
