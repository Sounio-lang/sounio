#!/bin/bash
#SBATCH --job-name=stdlib-alias-actual-worktree
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical14/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical14/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical14
A=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical05
M=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical08
B=/scratch/sounio-stdlib-wave3-20261027
S="$R/source"
RAW="$B/root-integrated01/results/madaros"
SEED="$B/compiler-resumed/generic-module-native/source/bin/souc-lean-single-x86_64"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
T="$P/tooling"
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$T/root/usr/lib/x86_64-linux-gnu"
export MAGIC="$T/root/usr/lib/file/magic.mgc"
ulimit -s524288
ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
date -u > "$R/results/STARTED_UTC"
cp -a "$A/source" "$S"
for path in check/specializer.sio compiler/module_frontend.sio ir/ir.sio main.sio compiler/main.sio ir/opt_cleanup.sio ir/serialize.sio native/codegen_x86_linux.sio native/frame.sio;do cp "$R/input/self-hosted/$path" "$S/self-hosted/$path";done
export SOUNIO_STDLIB_PATH="$S/self-hosted"
cd "$S"
sha256sum -c "$A/results/COMPILERS.sha256" > "$R/results/COMPILERS.before.verify"
sha256sum -c "$R/input/WORKTREE_SELF_HOSTED.sha256" > "$R/results/ACTUAL_WORKTREE.before.verify"
find self-hosted -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/COMPILE_CLOSURE.sha256"
cmp "$R/input/WORKTREE_SELF_HOSTED.sha256" "$R/results/COMPILE_CLOSURE.sha256"
echo EXACT_ACTUAL_WORKTREE_ALL_SELF_HOSTED_FILES > "$R/results/SOURCE_COMPARISON"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/INPUTS.sha256"
(
 status=0
 /usr/bin/time -v timeout 1200 "$RAW" check "$S/self-hosted/compiler/module_frontend.sio" > "$R/results/check.stdout" 2> "$R/results/check.stderr" || status=$?
 echo "$status" > "$R/results/CHECK_STATUS"
 exit "$status"
) &
check_pid=$!
(
 status=0
 /usr/bin/time -v timeout 600 "$SEED" "$R/input/tests/fixtures/wave4_alias_canonicalization/plain_driver.sio.in" "$R/results/component.elf" > "$R/results/seed.stdout" 2> "$R/results/seed.stderr" || status=$?
 echo "$status" > "$R/results/SEED_STATUS"
 test "$status" = 0
 chmod +x "$R/results/component.elf"
 sha256sum "$R/results/component.elf" > "$R/results/COMPONENT.sha256"
 bash "$M/input/scripts/compiler/alias_canonical_component_gate.sh" "$R/results/component.elf" "$M/input/tests/fixtures/wave4_alias_canonicalization" "$R/matrix" > "$R/results/gate.stdout" 2> "$R/results/gate.stderr"
) &
native_pid=$!
check_status=0;wait "$check_pid" || check_status=$?
native_status=0;wait "$native_pid" || native_status=$?
echo "$check_status" > "$R/results/CHECK_WAIT_STATUS"
echo "$native_status" > "$R/results/NATIVE_WAIT_STATUS"
sha256sum -c "$A/results/COMPILERS.sha256" > "$R/results/COMPILERS.final.verify"
sha256sum -c "$R/input/WORKTREE_SELF_HOSTED.sha256" > "$R/results/ACTUAL_WORKTREE.final.verify"
find self-hosted -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/ACTUAL_WORKTREE.final.sha256"
cmp "$R/results/COMPILE_CLOSURE.sha256" "$R/results/ACTUAL_WORKTREE.final.sha256"
sha256sum -c "$R/results/INPUTS.sha256" > "$R/results/INPUTS.final.verify"
test "$check_status" = 0
test "$native_status" = 0
cat "$R/results/gate.stdout"
echo ALIAS_ACTUAL_WORKTREE_COMPONENT_ONLY_OK
