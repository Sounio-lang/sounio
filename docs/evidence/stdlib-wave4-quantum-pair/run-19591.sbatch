#!/bin/bash
#SBATCH --job-name=quantum-final-timeout-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/quantum-final-custody37/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/quantum-final-custody37/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027;R=$B/services/quantum-final-custody37
mkdir "$R/archive" "$R/metadata"
trap 'r=$?;echo "$r" > "$R/metadata/STATUS";exit "$r"' EXIT
for group in quantum-pair-library25 quantum-pair-library28;do
 D=$B/services/$group/results
 [[ $(cat "$D/COLLECTOR_STATUS") == 1 ]]
 grep -Fxq $'after.overtake_slow.native\t124' "$D/PHASES.tsv"
 for n in INPUTS COMPILER SOURCES WITNESSES EXECUTABLES;do sha256sum -c "$D/$n.sha256" > "$R/metadata/$group.$n.verify";done
done
for group in wide-counts33 wide-counts34 wide-projection36;do
 D=$B/services/$group/results;[[ $(cat "$D/STATUS") == 0 ]]
 for n in INPUTS COMPILER EXECUTABLES SDK IR;do sha256sum -c "$D/$n.sha256" > "$R/metadata/$group.$n.verify";done
done
sacct -j 19524,19533,19568,19571,19576,19580 --format=JobID,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocCPUS,NodeList -P > "$R/metadata/ACCOUNTING"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$R/metadata/SNAPSHOT_TIME"
printf '%s\n' 'FINAL mode1 results:19524/19533 native124/3000s, collectors1. Custody0 never converts timeout to pass. Mode0 original19576 remains a separate in-flight diagnostic; its mutable results are excluded.' > "$R/metadata/SCOPE"
cd "$B"
find services/quantum-pair-library25 services/quantum-pair-library28 services/wide-counts33 services/wide-counts34 services/wide-projection36 services/quantum-final-custody37/input -type f -print0|sort -z|xargs -0 sha256sum > "$R/metadata/CONTENTS.sha256"
tar -czf "$R/archive/quantum-final-timeouts.tar.gz" services/quantum-pair-library25 services/quantum-pair-library28 services/wide-counts33 services/wide-counts34 services/wide-projection36 services/quantum-final-custody37/input services/quantum-final-custody37/metadata services/quantum-slow-mode0-35/run01.sbatch root-integrated01/results/madaros root-integrated01/RAW_READY root-integrated02/results/madaros root-integrated02/RAW_READY
gzip -t "$R/archive/quantum-final-timeouts.tar.gz"
sha256sum "$R/archive/quantum-final-timeouts.tar.gz" > "$R/metadata/ARCHIVE.sha256"
stat -c '%s' "$R/archive/quantum-final-timeouts.tar.gz" > "$R/metadata/ARCHIVE.bytes"
sha256sum -c "$R/metadata/CONTENTS.sha256" > "$R/metadata/CONTENTS.final.verify"
echo QUANTUM_FINAL_TIMEOUT_CUSTODY_OK
