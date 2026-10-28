#!/bin/bash
#SBATCH --job-name=stdlib-frontend-borrow-source
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/frontend-source-borrow03/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/frontend-source-borrow03/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/frontend-source-borrow03
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated01/results/madaros
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
T="$P/tooling"
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$T/root/usr/lib/x86_64-linux-gnu"
export MAGIC="$T/root/usr/lib/file/magic.mgc"
export SOUNIO_STDLIB_PATH=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical14/source/stdlib
ulimit -s524288
ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
printf '11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4  %s\n' "$RAW" > "$R/results/RAW.sha256"
sha256sum -c "$R/results/RAW.sha256" > "$R/results/RAW.before.verify"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/INPUTS.sha256"
for kind in rhs index;do
 source="$R/input/tests/fixtures/wave4_alias_canonicalization/frontend_borrow_${kind}_before.sio"
 status=0;timeout 30 "$RAW" check "$source" > "$R/results/$kind-check.stdout" 2> "$R/results/$kind-check.stderr" || status=$?
 echo "$status" > "$R/results/$kind-CHECK_STATUS"
 test "$status" = 1
 grep -F -e 'error[E037]' -e 'error[E038]' -e 'error[E091]' "$R/results/$kind-check.stdout"
 status=0;timeout 30 "$RAW" compile "$source" -o "$R/results/$kind.elf" > "$R/results/$kind-compile.stdout" 2> "$R/results/$kind-compile.stderr" || status=$?
 echo "$status" > "$R/results/$kind-COMPILE_STATUS"
 test "$status" = 1
 test ! -e "$R/results/$kind.elf"
done
source="$R/input/tests/fixtures/wave4_alias_canonicalization/frontend_borrow_after.sio"
timeout 30 "$RAW" check "$source" > "$R/results/after-check.stdout" 2> "$R/results/after-check.stderr"
timeout 30 "$RAW" compile "$source" -o "$R/results/after.elf" > "$R/results/after-compile.stdout" 2> "$R/results/after-compile.stderr"
chmod +x "$R/results/after.elf"
timeout 15 "$R/results/after.elf" > "$R/results/after-native.stdout" 2> "$R/results/after-native.stderr"
sha256sum "$R/results/after.elf" > "$R/results/AFTER_ELF.sha256"
sha256sum -c "$R/results/INPUTS.sha256" > "$R/results/INPUTS.final.verify"
sha256sum -c "$R/results/RAW.sha256" > "$R/results/RAW.final.verify"
echo FRONTEND_EXPLICIT_BORROW_ARGUMENT_ORDER_COMPONENT_ONLY_OK
