#!/bin/bash
#SBATCH --job-name=tuple-byte-windows
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:03:00
#SBATCH --export=NIL
#SBATCH --chdir=/tmp
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/services/concrete-tuple-windows84/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/services/concrete-tuple-windows84/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
T="$B/services/concrete-tuple-windows84"
PARENT="$B/services/concrete-tuple-lifetime83/results"
R="$B/root-integrated04";source "$R/tools-env-v3.sh"
cd "$T"
sha256sum elf_window.cpp "$CXX" "$PARENT/seed9/flat.elf" "$PARENT/seed9/nested.elf" "$PARENT/seedcc/flat.elf" "$PARENT/seedcc/nested.elf" "$PARENT/raw0/nested.elf" > INPUTS.before.sha256
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror elf_window.cpp -o elf_window >build.stdout 2>build.stderr
cmp "$PARENT/seed9/flat.elf" "$PARENT/seedcc/flat.elf"
cmp "$PARENT/seed9/nested.elf" "$PARENT/seedcc/nested.elf"
printf 'seed9_ccfa flat AND nested ELF byte-identical\n' > PAIR_BYTE_IDENTITY
./elf_window "$PARENT/seedcc/nested.elf" 0x401000 245 > nested-function.hex
./elf_window "$PARENT/seedcc/nested.elf" 0x4010f5 340 > clobber-function.hex
./elf_window "$PARENT/seedcc/nested.elf" 0x4019dc 69 around > nested-callsite.hex
./elf_window "$PARENT/seedcc/nested.elf" 0x401b68 69 around > clobber-callsite.hex
./elf_window "$PARENT/raw0/nested.elf" 0x401000 96 > modern-nested-prefix.hex
set +e
./elf_window "$PARENT/seedcc/nested.elf" 0xffffffffffffffff 32 > invalid.stdout 2>invalid.stderr
bad=$?
set -e
test "$bad" = 1
sha256sum -c INPUTS.before.sha256 > INPUTS.final.verify
sha256sum elf_window > TOOL.sha256
printf '0\n' > STATUS
