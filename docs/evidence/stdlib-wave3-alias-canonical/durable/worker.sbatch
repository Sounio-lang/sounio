#!/bin/bash
#SBATCH --job-name=stdlib-alias-component-durable
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical18/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical18/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical18
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
D=/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/alias-canonical01
S="$R/stage"
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
mkdir -p "$S/actual-worktree" "$S/compilers" "$S/prior-attempts" "$D"
test "$(cat "$B/alias-canonical11/results/WORKER_STATUS")" = 0
test "$(cat "$B/alias-canonical12/results/WORKER_STATUS")" = 0
grep -Fx ALIAS_EXPLICIT_NOMINAL_COMPONENT_ONLY_OK "$B/alias-canonical14/matrix/STATUS"
(cd "$B/alias-canonical14/source";sha256sum -c "$B/alias-canonical14/input/WORKTREE_SELF_HOSTED.sha256") > "$R/results/ACTUAL_SOURCE.before.verify"
cp -a "$B/alias-canonical14/source" "$S/actual-worktree/source"
cp -a "$B/alias-canonical14/input" "$S/actual-worktree/input"
cp -a "$B/alias-canonical14/matrix" "$S/actual-worktree/matrix"
cp "$B/alias-canonical14/worker.sbatch" "$S/actual-worktree/worker.sbatch"
mkdir -p "$S/actual-worktree/native-results"
for file in component.elf COMPONENT.sha256 SEED_STATUS SOURCE_COMPARISON COMPILE_CLOSURE.sha256 ACTUAL_WORKTREE.before.verify STARTED_UTC;do cp "$B/alias-canonical14/results/$file" "$S/actual-worktree/native-results/$file";done
date -u > "$S/actual-worktree/native-results/CAPTURED_UTC"
cp "$B/alias-canonical14/results/check.stdout" "$S/actual-worktree/native-results/whole-check-observed-at-capture.stdout"
printf '%s\n' 'NATIVE COMPONENT PHASE ONLY; whole frontend check emitted baseline E038 in unchanged functions and is not claimed PASS here.' > "$S/actual-worktree/native-results/SCOPE"
for name in 06 08 11 12;do
 mkdir -p "$S/alias-canonical$name"
 cp -a "$B/alias-canonical$name/results" "$S/alias-canonical$name/results"
 cp -a "$B/alias-canonical$name/input" "$S/alias-canonical$name/input"
 cp "$B/alias-canonical$name/worker.sbatch" "$S/alias-canonical$name/worker.sbatch"
 if test -d "$B/alias-canonical$name/matrix";then cp -a "$B/alias-canonical$name/matrix" "$S/alias-canonical$name/matrix";fi
done
cp "$B/alias-canonical11/source-before/self-hosted/check/specializer.sio" "$S/alias-canonical11/BEFORE-specializer.sio"
cp "$B/alias-canonical11/source-after/self-hosted/check/specializer.sio" "$S/alias-canonical11/AFTER-specializer.sio"
mkdir -p "$S/alias-canonical12/source-profile"
for file in ir/ir.sio compiler/main.sio ir/opt_cleanup.sio ir/serialize.sio native/codegen_x86_linux.sio native/frame.sio;do mkdir -p "$S/alias-canonical12/source-profile/$(dirname "$file")";cp "$B/alias-canonical12/source/self-hosted/$file" "$S/alias-canonical12/source-profile/$file";done
for name in 01 02 03 04 05 07 09 10 13;do
 path="$B/alias-canonical$name"
 if test -d "$path/results";then
  mkdir -p "$S/prior-attempts/alias-canonical$name"
  cp "$path/worker.sbatch" "$S/prior-attempts/alias-canonical$name/worker.sbatch"
  if test -d "$path/input";then cp -a "$path/input" "$S/prior-attempts/alias-canonical$name/input";fi
  tar -C "$path" --exclude='*.elf' -cf - results | tar -C "$S/prior-attempts/alias-canonical$name" -xf -
 fi
done
cp -a "$R/input" "$S/cleanup-input"
cp /scratch/sounio-stdlib-wave3-20261027/root-integrated01/results/madaros "$S/compilers/madaros-11332"
cp "$B/generic-module-native/source/bin/souc-lean-single-x86_64" "$S/compilers/seed-9cf"
printf '11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4  %s\n9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a  %s\n' "$S/compilers/madaros-11332" "$S/compilers/seed-9cf" > "$R/results/COMPILERS.sha256"
sha256sum -c "$R/results/COMPILERS.sha256" > "$R/results/COMPILERS.verify"
(cd "$S/actual-worktree/source";sha256sum -c "$B/alias-canonical14/input/WORKTREE_SELF_HOSTED.sha256") > "$R/results/ARCHIVED_SOURCE.verify"
(cd "$B/alias-canonical14/source";sha256sum -c "$B/alias-canonical14/input/WORKTREE_SELF_HOSTED.sha256") > "$R/results/ACTUAL_SOURCE.after-native.verify"
A="$D/alias-component-source-receipts-v1.tar.gz"
test ! -e "$A"
/usr/bin/time -v tar -C "$S" -czf "$A" . > "$R/results/archive.stdout" 2> "$R/results/archive.stderr"
sha256sum "$A" > "$R/results/ARCHIVE.sha256"
tar -tzf "$A" > "$R/results/CONTENTS.txt"
sha256sum -c "$R/results/ARCHIVE.sha256" > "$R/results/ARCHIVE.final.verify"
cp "$R/results/ARCHIVE.sha256" "$D/ARCHIVES.sha256"
cp "$R/worker.sbatch" "$D/durable-worker.sbatch"
echo ALIAS_COMPONENT_DURABLE_SOURCE_OK
