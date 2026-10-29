#!/bin/bash
#SBATCH --job-name=stdlib-raw03-abi0-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:08:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/raw03-abi0-export01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/raw03-abi0-export01/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/compiler-resumed/raw03-abi0-export01
O=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw03-abi0-export01
trap 'r=$?;echo "$r" > "$R/STATUS";if [[ -d "$O" ]];then echo "$r" > "$O/EXPORT_STATUS";fi;exit "$r"' EXIT
test ! -e "$O";mkdir -p "$O"
mountpoint /orangefs/training > "$O/MOUNT.txt"
findmnt -T "$O" > "$O/FINDMNT.txt"
findmnt -n -o FSTYPE -T "$O"|grep -Fx ceph > "$O/FSTYPE.txt"
printf '%s\n' root-integrated03 root-integrated03-loader compiler-resumed/worker-raw03-direct-schema01 compiler-resumed/worker-raw03-stub-arity01 compiler-resumed/worker-raw03-live-schema01 compiler-resumed/worker-raw03-live-schema02 compiler-resumed/worker-raw03-live-schema03 compiler-resumed/worker-raw03-live-schema04 compiler-resumed/worker-raw03-live-schema05 compiler-resumed/worker-raw03-live-schema06 compiler-resumed/worker-raw03-live-schema08 compiler-resumed/worker-canonical-stub01 compiler-resumed/worker-canonical-stub-native01 > "$O/MEMBERS.txt"
cd "$B"
(cd root-integrated03/source;sha256sum -c ../results/SOURCE.sha256) > "$O/ROOT_SOURCE.before.verify"
sha256sum -c root-integrated03/RAW_READY > "$O/RAW.before.verify"
while IFS= read -r path;do test -e "$path";done < "$O/MEMBERS.txt"
find $(cat "$O/MEMBERS.txt") -type f -print0|sort -z|xargs -0 sha256sum > "$O/SOURCE.before.sha256"
tar -cf - -T "$O/MEMBERS.txt"|gzip -1 > "$O/raw03-abi0-sources-seeds-elfs-receipts-v1.tar.gz"
tar -tzf "$O/raw03-abi0-sources-seeds-elfs-receipts-v1.tar.gz" > "$O/TAR_CONTENTS.txt"
sha256sum -c "$O/SOURCE.before.sha256" > "$O/SOURCE.final.verify"
sha256sum "$O/raw03-abi0-sources-seeds-elfs-receipts-v1.tar.gz" > "$O/SHA256SUMS"
(cd "$O";sha256sum -c SHA256SUMS) > "$O/ARCHIVE.final.verify"
sacct -X -j19587,19588,19589,19610,19620,19622,19624,19625,19626,19627,19628,19630,19632,19634 --format=JobID,JobName,User,State,ExitCode,Elapsed,WorkDir -P > "$O/SLURM_RECEIPTS.tsv"
cp "$R/job.sh" "$O/export-job.sh"
echo RAW03_AND_CANONICAL_ABI0_DURABLE_NOT_PUBLIC_CLI_QUALIFICATION
