#!/bin/bash
#SBATCH --job-name=stdlib-argv-durable-readback
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/argv-copy46/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/argv-copy46/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
R=/scratch/sounio-stdlib-wave3-20261027/services/argv-custody45
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/argv-export08
trap 'r=$?;echo "$r" >"$D/COPY_STATUS";exit "$r"' EXIT
[[ $(cat "$R/metadata/STATUS") == 0 ]]
sha256sum -c "$R/metadata/ARCHIVE.sha256" >"$D/SOURCE.verify"
test ! -e "$D/stdlib-argv.tar.gz"
cp "$R/archive/stdlib-argv.tar.gz" "$D/stdlib-argv.tar.gz"
cp -a "$R/metadata/." "$D/metadata/"
cd "$D"
sha256sum stdlib-argv.tar.gz >ARCHIVE.sha256
test "$(cut -d' ' -f1 ARCHIVE.sha256)" = 3d7125422c24b7ed9709e7debf8078a6675ca247149c3c0ad91ec534d67366d3
test "$(stat -c%s stdlib-argv.tar.gz)" = 944737421
sha256sum -c ARCHIVE.sha256 >READBACK.verify
echo STDLIB_ARGV_DURABLE_READBACK_OK
