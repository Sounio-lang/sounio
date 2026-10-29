#!/bin/bash
#SBATCH --job-name=stdlib-argv-final-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/argv-custody45/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/argv-custody45/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/argv-custody45
mkdir "$R/archive" "$R/metadata"
trap 'r=$?;echo "$r" >"$R/metadata/STATUS";exit "$r"' EXIT
for group in argv-contract38 argv-contract39 argv-frozen40 argv-final42 argv-source-guards44;do
 D=$B/services/$group/results
 wanted=0;case "$group" in argv-contract*)wanted=1;;esac
 [[ $(cat "$D/STATUS") == "$wanted" ]]
 for n in INPUTS COMPILER EXECUTABLES;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" >"$R/metadata/$group.$n.verify";fi;done
done
sacct -j19597,19599,19603,19606,19608 --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS,NodeList -P >"$R/metadata/ACCOUNTING"
date -u +'%Y-%m-%dT%H:%M:%SZ' >"$R/metadata/SNAPSHOT_TIME"
printf '%s\n' 'Preserves19597 staging failure and19599 current-docs admission failure;19603 candidate and19606 final original/imported material BOTH modes pass;19608 final7suites and allsourceguards pass. Includes fullbefore/after sources/captures/raw8abe/compiler source/Node; not fullstdlib acceptance.' >"$R/metadata/SCOPE"
cd "$B"
tar -czf "$R/archive/stdlib-argv.tar.gz" services/argv-contract38 services/argv-contract39 services/argv-frozen40 services/argv-final42 services/argv-source-guards44 services/argv-custody45/patch03e2.txt root-integrated02/source root-integrated02/results root-integrated02/RAW_READY services-tools-recovered/tools/node
sha256sum "$R/archive/stdlib-argv.tar.gz" >"$R/metadata/ARCHIVE.sha256"
stat -c%s "$R/archive/stdlib-argv.tar.gz" >"$R/metadata/ARCHIVE.bytes"
tar -tzf "$R/archive/stdlib-argv.tar.gz" >"$R/metadata/CONTENTS.list"
for group in argv-contract38 argv-contract39 argv-frozen40 argv-final42 argv-source-guards44;do
 D=$B/services/$group/results
 for n in INPUTS COMPILER EXECUTABLES;do if [[ -s "$D/$n.sha256" ]];then sha256sum -c "$D/$n.sha256" >"$R/metadata/$group.$n.final.verify";fi;done
done
echo STDLIB_ARGV_CUSTODY_OK
