#!/bin/bash
#SBATCH --job-name=pireus-input-closure-readback
#SBATCH --chdir=/tmp
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/pireus-copy49/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/pireus-copy49/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
R=/scratch/sounio-stdlib-wave3-20261027/services/pireus-custody48
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/pireus-staging-export09
trap 'r=$?;echo "$r" >"$D/COPY_STATUS";exit "$r"' EXIT
[[ $(cat "$R/metadata/STATUS") == 0 ]]
sha256sum -c "$R/metadata/ARCHIVE.sha256" >"$D/SOURCE.verify"
test ! -e "$D/pireus-input-closure.tar.gz"
cp "$R/archive/pireus-input-closure.tar.gz" "$D/pireus-input-closure.tar.gz"
cp -a "$R/metadata/." "$D/metadata/"
cd "$D"
sha256sum pireus-input-closure.tar.gz >ARCHIVE.sha256
test "$(cut -d' ' -f1 ARCHIVE.sha256)" = "$(cut -d' ' -f1 "$R/metadata/ARCHIVE.sha256")"
test "$(stat -c%s pireus-input-closure.tar.gz)" = "$(cat "$R/metadata/ARCHIVE.bytes")"
sha256sum -c ARCHIVE.sha256 >READBACK.verify
echo PIREUS_INPUT_CLOSURE_DURABLE_READBACK_OK
