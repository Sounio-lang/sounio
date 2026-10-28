#!/bin/bash
#SBATCH --job-name=array-cast-component-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=512M
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/array-cast-custody16/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/array-cast-custody16/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/array-cast-custody16
mkdir "$R/archive" "$R/metadata"
trap 'r=$?;echo "$r" > "$R/metadata/STATUS";exit "$r"' EXIT
cd "$B"
for job in 19498 19501;do sacct -j "$job" --format=JobID,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,ReqMem,NodeList -P > "$R/metadata/sacct-$job.txt";done
for name in INPUTS TOOLS EXECUTABLES;do sha256sum -c "$B/services/array-cast-checker15/results/$name.sha256" > "$R/metadata/$name.verify";done
find services/array-cast-checker14 services/array-cast-checker15 -type f -print0|sort -z|xargs -0 sha256sum > "$R/metadata/CONTENTS.sha256"
tar -czf "$R/archive/array-cast-checker.tar.gz" services/array-cast-checker14 services/array-cast-checker15 services/array-cast-custody16/metadata
gzip -t "$R/archive/array-cast-checker.tar.gz"
sha256sum "$R/archive/array-cast-checker.tar.gz" > "$R/metadata/ARCHIVE.sha256"
stat -c '%s' "$R/archive/array-cast-checker.tar.gz" > "$R/metadata/ARCHIVE.bytes"
sha256sum -c "$R/metadata/CONTENTS.sha256" > "$R/metadata/CONTENTS.final.verify"
echo FAILED_COMPONENT_CUSTODY_COMPLETE
