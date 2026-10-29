#!/bin/bash
#SBATCH --job-name=stdlib-root05-failed-readback
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/root05-failed-readback-%j.stdout
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/root05-failed-readback-%j.stderr
set -euo pipefail
E=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw05-failed-export01
D=$E/readback-$SLURM_JOB_ID
mountpoint -q /orangefs/training
mkdir "$D"
sha256sum -c "$E/ARCHIVE.sha256" > "$D/ARCHIVE.verify"
tar -tzf "$E/raw05-failed-full.tar.gz" > "$D/ARCHIVE_CONTENTS.txt"
cmp "$E/ARCHIVE_CONTENTS.txt" "$D/ARCHIVE_CONTENTS.txt"
if grep -Fxq root-integrated05/RAW_READY "$D/ARCHIVE_CONTENTS.txt";then exit 1;fi
tar -xzf "$E/raw05-failed-full.tar.gz" -C "$D" root-integrated05/results/madaros root-integrated05/results/BUILD_STATUS root-integrated05/results/EXPLICIT_ERRORS root-integrated05/SOURCE_HEAD root-integrated05/preflight/STATUS
printf '55d49818a998fa6620f52cce237332baa9a52e9a2d1bc0ee575d65493cb8966c  %s\n' "$D/root-integrated05/results/madaros" > "$D/FAILED_ELF.sha256"
sha256sum -c "$D/FAILED_ELF.sha256" > "$D/FAILED_ELF.verify"
test "$(cat "$D/root-integrated05/results/BUILD_STATUS")" = 1
test "$(cat "$D/root-integrated05/preflight/STATUS")" = 0
test "$(cat "$D/root-integrated05/SOURCE_HEAD")" = 237ff2d674eb8e8f350d948c4f6f6393706cd67d
printf 'RAW05_FAILED_RECEIPT_READBACK_OK_NO_EXECUTION\n' > "$D/STATUS"
