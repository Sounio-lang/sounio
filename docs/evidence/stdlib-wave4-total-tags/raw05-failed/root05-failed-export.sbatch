#!/bin/bash
#SBATCH --job-name=stdlib-root05-failed-export
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/root05-failed-export-%j.stdout
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/root05-failed-export-%j.stderr
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R=$B/root-integrated05
E=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/compiler/raw05-failed-export01
mountpoint -q /orangefs/training
mkdir "$E" "$E/inputs"
test "$(cat "$R/results/BUILD_STATUS")" = 1
test ! -e "$R/RAW_READY"
printf '55d49818a998fa6620f52cce237332baa9a52e9a2d1bc0ee575d65493cb8966c  %s\n' "$R/results/madaros" > "$E/FAILED_ELF.sha256"
sha256sum -c "$E/FAILED_ELF.sha256" > "$E/FAILED_ELF.verify"
(cd "$R/source";sha256sum -c "$R/results/SOURCE.sha256") > "$E/ACTUAL_SOURCE.verify"
sha256sum -c "$R/preflight/SEED.sha256" > "$E/SEED.verify"
sha256sum -c "$R/preflight/INPUTS.sha256" > "$E/INPUTS.verify"
cp "$B/root-integrated04/tools-env-v3.sh" "$E/inputs/tools-env-v3.sh"
cp "$B/root-integrated04/private-tools-v3/g++" "$E/inputs/g++-v3-wrapper"
cp "$B/root-integrated04/results/madaros" "$E/inputs/accepted-raw04-control.elf"
cp "$B/root-integrated04/source/bin/souc-lean-single-x86_64" "$E/inputs/accepted-old-seed9cf.elf"
printf 'FAILED_BUILD_NO_RAW_READY\nmain19789 FAILED1; seed emitted ELF exit0 but nine explicit diagnostics rejected it. Failed ELF is preserved only; it was not executed or qualified. No modern raw05 validation ran.\n' > "$E/FAILED_RECEIPT.txt"
find "$R" -type f -print0 | sort -z | xargs -0 sha256sum > "$E/BEFORE.sha256"
cd "$B"
tar -czf "$E/raw05-failed-full.tar.gz" root-integrated05 -C "$E" inputs FAILED_RECEIPT.txt FAILED_ELF.sha256
sha256sum "$E/raw05-failed-full.tar.gz" > "$E/ARCHIVE.sha256"
tar -tzf "$E/raw05-failed-full.tar.gz" > "$E/ARCHIVE_CONTENTS.txt"
sha256sum -c "$E/ARCHIVE.sha256"
find "$R" -type f -print0 | sort -z | xargs -0 sha256sum > "$E/AFTER.sha256"
cmp "$E/BEFORE.sha256" "$E/AFTER.sha256"
sacct -X -j19788,19789 --format=JobID,State,ExitCode,Elapsed,MaxRSS -P > "$E/SLURM.txt"
printf 'RAW05_FAILED_EXPORT_OK_NOT_A_COMPILER_PASS\n' > "$E/STATUS"
