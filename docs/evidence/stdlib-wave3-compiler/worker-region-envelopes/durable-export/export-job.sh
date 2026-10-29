#!/bin/bash
#SBATCH --job-name=stdlib-worker-storage-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:08:00
#SBATCH --dependency=afterany:19575
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-region-storage-export02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-region-storage-export02/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
R=$B/worker-region-storage-export02
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/worker-region-storage-export02
trap 'r=$?;echo "$r" > "$R/STATUS";if [ -d "$O" ];then echo "$r" > "$O/EXPORT_STATUS";fi;exit "$r"' EXIT
test ! -e "$O"
mkdir -p "$O"
mountpoint /orangefs/training > "$O/MOUNT.txt"
findmnt -T "$O" > "$O/FINDMNT.txt"
findmnt -n -o FSTYPE -T "$O" | grep -Fx ceph > "$O/FSTYPE.txt"
printf '%s\n' worker-zero-token-dump01 worker-zero-runtime01 worker-zero-runtime03 worker-zero-runtime04 worker-zero-runtime05 worker-zero-runtime06 worker-region-envelope03 worker-frame-native01 worker-region-common01 worker-literal-preflight01 worker-literal-preflight02 worker-literal-preflight03 storage-type-foundation03 storage-type-foundation04 storage-type-foundation05 storage-type-foundation06 storage-type-foundation07 storage-linearity01 worker-portable-gate-v3/input/bin/souc-lean-single-x86_64 > "$O/MEMBERS.txt"
cd "$B"
while IFS= read -r member;do test -e "$member";done < "$O/MEMBERS.txt"
find $(cat "$O/MEMBERS.txt") -type f -print0|sort -z|xargs -0 sha256sum > "$O/SOURCE.before.sha256"
sha256sum /scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros > "$O/RAW.before.sha256"
tar -cf - -T "$O/MEMBERS.txt" | gzip -1 > "$O/worker-region-storage-sources-seeds-receipts-v1.tar.gz"
tar -tzf "$O/worker-region-storage-sources-seeds-receipts-v1.tar.gz" > "$O/TAR_CONTENTS.txt"
cp /scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros "$O/madaros-root-integrated02.raw"
sha256sum -c "$O/SOURCE.before.sha256" > "$O/SOURCE.final.verify"
sha256sum -c "$O/RAW.before.sha256" > "$O/RAW.final.verify"
sha256sum "$O/worker-region-storage-sources-seeds-receipts-v1.tar.gz" "$O/madaros-root-integrated02.raw" > "$O/SHA256SUMS"
SDK_ARCHIVE=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/root/integrated02-export01/integrated02.tar.gz
printf '%s  %s\n' d840afa3a675effab21316c8ee9a7a5ce22939dba7ef6c94b5024adc65ab817c "$SDK_ARCHIVE" > "$O/SDK_ARCHIVE.sha256"
sha256sum -c "$O/SDK_ARCHIVE.sha256" > "$O/SDK_ARCHIVE.verify"
mkdir -p "$O/sdk-audit"
tar -tzf "$SDK_ARCHIVE" | grep -E 'root-integrated02-loader/gate/(cases/|source/stdlib/|libstdlib_platform.so$)' | grep -v '/$' > "$O/SDK_MEMBERS.txt"
test -s "$O/SDK_MEMBERS.txt"
tar -xzf "$SDK_ARCHIVE" -C "$O/sdk-audit" --no-recursion -T "$O/SDK_MEMBERS.txt"
SDK=$(find "$O/sdk-audit" -type d -path '*/root-integrated02-loader/gate')
test -n "$SDK"
T=/scratch/sounio-stdlib-wave3-20261027/root-integrated02-loader/gate
P=$B/worker-literal-preflight03/results
sed "s|$T|$SDK|g" "$P/TARGET.before.sha256" > "$O/SDK_EXPECTED.sha256"
sed "s|$T|$SDK|g" "$P/RUNTIME.before.sha256" >> "$O/SDK_EXPECTED.sha256"
sha256sum -c "$O/SDK_EXPECTED.sha256" > "$O/SDK_MATCH.verify"
sha256sum -c "$P/TARGET.before.sha256" > "$O/TARGET.final.verify"
sha256sum -c "$P/RUNTIME.before.sha256" > "$O/RUNTIME.final.verify"
sacct -X -j19507,19520,19529,19538,19555,19559,19563,19569,19570,19572,19573,19574,19575 --format=JobID,JobName,State,ExitCode,Elapsed,WorkDir -P > "$O/SLURM_RECEIPTS.tsv"
cp "$R/job.sh" "$O/export-job.sh"
cd "$O"
sha256sum -c SHA256SUMS > ARCHIVE.final.verify
echo WORKER_STORAGE_CLOSED_COMPONENT_DURABLE_EXPORT_OK
