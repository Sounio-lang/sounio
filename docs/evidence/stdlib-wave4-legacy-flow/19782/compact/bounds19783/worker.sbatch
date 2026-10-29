#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-bounds-component02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-bounds-component02/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/legacy-flow-bounds-component02
R=$B/root-integrated04
S=$B/compiler-resumed/generic-module-native/source
source "$R/tools-env-v2.sh"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
ulimit -s 524288
cd "$D"
find source -type f -print0 | sort -z | xargs -0 sha256sum > INPUTS.before.sha256
perl source/tests/fixtures/wave4_legacy_flow/bounds_component.pl "$B/compiler-resumed/legacy-flow-unified03/source/self-hosted/compiler/lean_single.sio" source/self-hosted/compiler/lean_single.sio source/tests/fixtures/wave4_legacy_flow/bounds_main.sio.in "$D" > generator.stdout
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror source/tests/fixtures/wave4_legacy_flow/bounds_reference.cpp -o reference
for f in before-*.sio after.sio; do
 arm=$(basename "$f" .sio)
 set +e
 timeout 60 "$S/bin/souc-lean-single-x86_64" "$f" "$arm.elf" >"$arm.compile.stdout" 2>"$arm.compile.stderr"
 cr=$?
 echo "$cr" > "$arm.compile.status"
 nr=999
 if [ "$cr" = 0 ]; then chmod +x "$arm.elf"; timeout 5 "./$arm.elf" >"$arm.native.stdout" 2>"$arm.native.stderr"; nr=$?; fi
 echo "$nr" > "$arm.native.status"
 set -e
done
set +e
./reference --individual "$D" >before.oracle.stdout 2>before.oracle.stderr
echo "$?" > before.ORACLE_STATUS
./reference after.native.stdout >after.oracle.stdout 2>after.oracle.stderr
echo "$?" > after.ORACLE_STATUS
set -e
sha256sum -c INPUTS.before.sha256 > INPUTS.final.verify
test "$(cat after.ORACLE_STATUS)" = 0
test "$(cat after.native.status)" = 0
test "$(cat before.ORACLE_STATUS)" = 4
