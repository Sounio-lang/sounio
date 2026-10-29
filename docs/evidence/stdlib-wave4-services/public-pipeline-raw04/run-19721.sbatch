#!/bin/bash
#SBATCH --job-name=stdlib-public-services-custody
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/public-custody64/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/public-custody64/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/services/public-custody64
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/public-pipeline-export12
mkdir "$D/metadata" "$O"
finish(){ r=$?;trap - EXIT;echo "$r" >"$D/metadata/STATUS";exit "$r";};trap finish EXIT
test "$(cat "$B/services/public-pipeline59/normal-results/STATUS")" = 0
test "$(cat "$B/services/public-pipeline61/pool-results/STATUS")" = 0
test "$(cat "$B/services/public-pipeline59/STATUS")" = 1
test "$(cat "$B/services/public-pipeline61/STATUS")" = 0
test "$(cat "$B/services/public-pipeline57/STATUS")" = 1
(cd "$B/services/public-pipeline59/input";sha256sum -c "$B/services/public-pipeline61/receipts/CANDIDATE.sha256") >"$D/metadata/CANDIDATE.verify"
sacct -X -j 19707,19711,19715 --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem -P >"$D/metadata/ACCOUNTING"
cp "$0" "$D/input/custody.sbatch"
cd "$B"
items=(services/public-pipeline57 services/public-pipeline59 services/public-pipeline61 services/toolchain-closure59 services/toolchain-closure61 services/public-custody64/input sdk/arrow26 root-integrated04/results/madaros root-integrated04/RAW_READY root-integrated04/private-tools-v3 root-integrated04/private-runtime root-integrated04/pthread-installed-v3 root-integrated04/tools-env-v3.sh services/bigframe-abi07/gcc13-private/root)
find "${items[@]}" -type f -print0|sort -z|xargs -0 sha256sum >"$D/metadata/CONTENTS.sha256"
find "${items[@]}" -type l -printf '%p -> %l\n'|sort >"$D/metadata/SYMLINKS.before"
tar -czf "$D/public-pipeline.tar.gz" "${items[@]}" services/public-custody64/metadata
gzip -t "$D/public-pipeline.tar.gz"
sha256sum -c "$D/metadata/CONTENTS.sha256" >"$D/metadata/CONTENTS.final.verify"
find "${items[@]}" -type l -printf '%p -> %l\n'|sort >"$D/metadata/SYMLINKS.after"
cmp "$D/metadata/SYMLINKS.before" "$D/metadata/SYMLINKS.after"
sha256sum "$D/public-pipeline.tar.gz" >"$D/metadata/ARCHIVE.sha256"
stat -c '%s' "$D/public-pipeline.tar.gz" >"$D/metadata/ARCHIVE.bytes"
cp "$D/public-pipeline.tar.gz" "$O/public-pipeline.tar.gz"
read -r expected rest <"$D/metadata/ARCHIVE.sha256"
printf '%s  %s\n' "$expected" "$O/public-pipeline.tar.gz" >"$O/ARCHIVE.sha256"
sha256sum -c "$O/ARCHIVE.sha256" >"$O/READBACK.verify"
echo 0 >"$O/COPY_STATUS"
cp -a "$D/metadata" "$O/metadata"
echo PUBLIC_SERVICES_CUSTODY_OK
