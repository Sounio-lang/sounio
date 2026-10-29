#!/bin/bash
#SBATCH --job-name=concrete-tuple-custody
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --chdir=/tmp
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/concrete-tuple-custody85/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/concrete-tuple-custody85/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
T="$B/services/concrete-tuple-custody85"
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/services/concrete-tuple-export14
mkdir "$D"
cd "$B"
FILES=(
 services/concrete-tuple-lifetime83
 services/concrete-tuple-windows84
 compiler-resumed/generic-module-native/source/bin/souc-lean-single-x86_64
 compiler-resumed/legacy-flow-bounds03/results/generation2-producer.elf
 compiler-resumed/legacy-flow-bounds03/results/candidate.sio
 root-integrated04/results/madaros
 root-integrated04/SOURCE_HEAD
 root-integrated04/tools-env-v3.sh
 root-integrated04/private-tools-v3/g++
 root-integrated04/source/self-hosted/ir/lower.sio
 root-integrated04/source/self-hosted/native/codegen_x86_linux.sio
 root-integrated05/source/stdlib
 root-integrated05/preflight/PRIVATE_SDK.sha256
 root-integrated05/preflight/PRIVATE_SDK_SYMLINKS.txt
 root-integrated05/seed-private/producer.sio
 compiler-resumed/root5-option-profiles01/tests/fixtures/wave4_producer_root5/option/input/native.sio
 compiler-resumed/root5-option-profiles01/tests/fixtures/wave4_producer_root5/option/reference.cpp
)
tar -cf - "${FILES[@]}" | gzip -1 > "$T/concrete-tuple.tar.gz"
sha256sum "$T/concrete-tuple.tar.gz" > "$T/LOCAL.sha256"
cp "$T/concrete-tuple.tar.gz" "$D/concrete-tuple.tar.gz"
sha256sum "$D/concrete-tuple.tar.gz" > "$T/REMOTE.sha256"
test "$(cut -d' ' -f1 "$T/LOCAL.sha256")" = "$(cut -d' ' -f1 "$T/REMOTE.sha256")"
tar -tzf "$D/concrete-tuple.tar.gz" > "$T/CENSUS.txt"
printf '%s\n' "$D/concrete-tuple.tar.gz" > "$T/REMOTE_PATH"
printf '0\n' > "$T/STATUS"
