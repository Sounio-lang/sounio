#!/bin/bash
#SBATCH --job-name=legacy-flow-final
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-metadata02/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/root5-final-metadata02/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D=$B/compiler-resumed/root5-final-metadata02
R=$B/root-integrated04
S=$B/compiler-resumed/generic-module-native/source
source "$R/tools-env-v3.sh"
cd "$D"
mkdir presence bounds
C=$B/compiler-resumed/root5-final-transport04/source
producer=$B/compiler-resumed/root5-final-transport04/results/generation2-producer.elf
sha256sum "$producer" "$C/self-hosted/compiler/lean_single.sio" >INPUTS.before.sha256
find "$C/tests/fixtures/wave4_legacy_flow" -type f -print0|sort -z|xargs -0 sha256sum >>INPUTS.before.sha256
export SOUNIO_STDLIB_PATH="$S/self-hosted"
ulimit -s 524288
perl "$C/tests/fixtures/wave4_legacy_flow/presence_component.pl" "$B/compiler-resumed/legacy-flow-final01/source/self-hosted/compiler/lean_single.sio" "$C/self-hosted/compiler/lean_single.sio" "$C/tests/fixtures/wave4_legacy_flow/presence_main.sio.in" "$D/presence"
perl "$C/tests/fixtures/wave4_legacy_flow/bounds_component.pl" "$B/compiler-resumed/legacy-flow-unified03/source/self-hosted/compiler/lean_single.sio" "$C/self-hosted/compiler/lean_single.sio" "$C/tests/fixtures/wave4_legacy_flow/bounds_main.sio.in" "$D/bounds" >bounds/generator.stdout
for kind in presence bounds;do
 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$C/tests/fixtures/wave4_legacy_flow/${kind}_reference.cpp" -o "$kind/reference"
 timeout --kill-after=10 120 "$producer" "$kind/after.sio" "$kind/after.elf" >"$kind/compile.stdout" 2>"$kind/compile.stderr"
 chmod 755 "$kind/after.elf"
 timeout --kill-after=5 15 "$kind/after.elf" >"$kind/native.stdout" 2>"$kind/native.stderr"
 "$kind/reference" "$kind/native.stdout" >"$kind/oracle.stdout" 2>"$kind/oracle.stderr"
 echo PASS >"$kind/STATUS"
done
sha256sum -c INPUTS.before.sha256 >INPUTS.final.verify
cat presence/oracle.stdout bounds/oracle.stdout
echo PASS >STATUS

