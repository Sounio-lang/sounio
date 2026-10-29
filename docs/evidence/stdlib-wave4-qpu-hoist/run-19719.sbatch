#!/bin/bash
#SBATCH --job-name=qpu-hoist-durable-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/qpu-custody62/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/qpu-custody62/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/services/qpu-custody62
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/qpu-hoist-export11
mkdir "$D/metadata" "$O"
finish(){ r=$?;trap - EXIT;echo "$r" >"$D/metadata/STATUS";exit "$r";};trap finish EXIT
test "$(cat "$B/services/qpu-hoist55/results/STATUS")" = 0
test "$(cat "$B/services/qpu-hoist56/NUMERICAL_STATUS")" = 0
test "$(cat "$B/services/qpu-hoist56/CAPTURE_STATUS")" = 1
test "$(cat "$B/services/qpu-dyadic-compat58/STATUS")" = 1
test "$(cat "$B/services/qpu-dyadic-compat60/STATUS")" = 0
test "$(cat "$B/services/qpu-source-validator63/STATUS")" = 0
sacct -X -j 19693,19702,19710,19713,19718 --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem -P >"$D/metadata/ACCOUNTING"
cp "$0" "$D/input/custody.sbatch"
cd "$B"
items=(services/qpu-hoist55 services/qpu-hoist56 services/qpu-dyadic-compat58 services/qpu-dyadic-compat60 services/qpu-source-validator63 services/qpu-custody62/input root-integrated03/results/madaros root-integrated03/RAW_READY root-integrated04/results/madaros root-integrated04/RAW_READY root-integrated04/private-tools-v3 root-integrated04/private-runtime root-integrated04/pthread-installed-v3 root-integrated04/tools-env-v3.sh services/bigframe-abi07/gcc13-private/root)
find "${items[@]}" -type f -print0|sort -z|xargs -0 sha256sum >"$D/metadata/CONTENTS.sha256"
find "${items[@]}" -type l -printf '%p -> %l\n'|sort >"$D/metadata/SYMLINKS.before"
tar -czf "$D/qpu-hoist.tar.gz" "${items[@]}" services/qpu-custody62/metadata
gzip -t "$D/qpu-hoist.tar.gz"
sha256sum -c "$D/metadata/CONTENTS.sha256" >"$D/metadata/CONTENTS.final.verify"
find "${items[@]}" -type l -printf '%p -> %l\n'|sort >"$D/metadata/SYMLINKS.after"
cmp "$D/metadata/SYMLINKS.before" "$D/metadata/SYMLINKS.after"
sha256sum "$D/qpu-hoist.tar.gz" >"$D/metadata/ARCHIVE.sha256"
stat -c '%s' "$D/qpu-hoist.tar.gz" >"$D/metadata/ARCHIVE.bytes"
cp "$D/qpu-hoist.tar.gz" "$O/qpu-hoist.tar.gz"
read -r expected rest <"$D/metadata/ARCHIVE.sha256"
printf '%s  %s\n' "$expected" "$O/qpu-hoist.tar.gz" >"$O/ARCHIVE.sha256"
sha256sum -c "$O/ARCHIVE.sha256" >"$O/READBACK.verify"
echo 0 >"$O/COPY_STATUS"
cp -a "$D/metadata" "$O/metadata"
echo QPU_HOIST_CUSTODY_OK
