#!/bin/bash
#SBATCH --job-name=array-access-durable-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380p-dl380p-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=512M
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/array-access-custody13/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/array-access-custody13/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/services/array-access-custody13
mkdir "$R/archive" "$R/metadata"
trap 'r=$?;echo "$r" > "$R/metadata/STATUS";exit "$r"' EXIT
cd "$B"
for job in 19448 19460 19480;do sacct -j "$job" --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocCPUS,ReqMem,NodeList -P > "$R/metadata/sacct-$job.txt";done
sha256sum -c "$B/root-integrated01/RAW_READY" > "$R/metadata/RAW.verify"
for name in COMPILER INPUTS LIBRARIES WITNESS_TOOLS MATRIX_INPUTS REFERENCE;do
 sha256sum -c "$B/services/bigframe-local-arrays11/results/$name.sha256" > "$R/metadata/library-$name.verify"
done
for name in INPUTS COMPILER REFERENCE;do
 sha256sum -c "$B/services/array-semantics10/results/$name.sha256" > "$R/metadata/semantics-$name.verify"
done
find services/bigframe-local-arrays11 services/array-semantics10 services/array-access-custody13/input services/bigframe-abi07/input02 -type f -print0 | sort -z |xargs -0 sha256sum > "$R/metadata/CONTENTS.sha256"
tar -czf "$R/archive/array-access.tar.gz"  services/bigframe-local-arrays11 services/array-semantics10 services/bigframe-abi07/input02 services/array-access-custody13/input services/array-access-custody13/metadata/CONTENTS.sha256  root-integrated01/results/madaros root-integrated01/RAW_READY  root-integrated01/source/tests/stdlib/data/test_bigframe_ops_stdlib.sio root-integrated01/source/tests/stdlib/data/test_bigframe_stats.sio root-integrated01/source/tests/stdlib/data/test_bigframe_ml.sio
gzip -t "$R/archive/array-access.tar.gz"
sha256sum "$R/archive/array-access.tar.gz" > "$R/metadata/ARCHIVE.sha256"
stat -c '%s' "$R/archive/array-access.tar.gz" > "$R/metadata/ARCHIVE.bytes"
sha256sum -c "$R/metadata/CONTENTS.sha256" > "$R/metadata/CONTENTS.final.verify"
sha256sum -c "$R/metadata/ARCHIVE.sha256" > "$R/metadata/ARCHIVE.final.verify"
echo ARRAY_ACCESS_LOCAL_CUSTODY_OK
