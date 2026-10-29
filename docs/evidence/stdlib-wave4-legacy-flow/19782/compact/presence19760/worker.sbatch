#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-presence01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/legacy-flow-presence01/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/legacy-flow-presence01
R=$B/root-integrated04
S=$B/compiler-resumed/generic-module-native/source
source "$R/tools-env-v2.sh"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
ulimit -s 524288
cd "$D"
find source -type f -print0 | sort -z | xargs -0 sha256sum > INPUTS.before.sha256
perl source/tests/fixtures/wave4_legacy_flow/presence_component.pl "$B/compiler-resumed/legacy-flow-final01/source/self-hosted/compiler/lean_single.sio" source/self-hosted/compiler/lean_single.sio source/tests/fixtures/wave4_legacy_flow/presence_main.sio.in "$D"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror source/tests/fixtures/wave4_legacy_flow/presence_reference.cpp -o reference
for arm in before after; do
 set +e
 timeout 60 "$S/bin/souc-lean-single-x86_64" "$arm.sio" "$arm.elf" >"$arm.compile.stdout" 2>"$arm.compile.stderr"
 cr=$?
 echo "$cr" > "$arm.compile.status"
 nr=999
 if [ "$cr" = 0 ]; then chmod +x "$arm.elf"; timeout 5 "./$arm.elf" >"$arm.native.stdout" 2>"$arm.native.stderr"; nr=$?; fi
 echo "$nr" > "$arm.native.status"
 ./reference "$arm.native.stdout" >"$arm.oracle.stdout" 2>"$arm.oracle.stderr"
 echo "$?" > "$arm.ORACLE_STATUS"
 set -e
done
sha256sum -c INPUTS.before.sha256 > INPUTS.final.verify
test "$(cat after.ORACLE_STATUS)" = 0
test "$(cat after.native.status)" = 0
test "$(cat before.ORACLE_STATUS)" = 4
