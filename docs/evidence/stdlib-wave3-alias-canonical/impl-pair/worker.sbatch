#!/bin/bash
#SBATCH --job-name=stdlib-alias-impl-shadow
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical11/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical11/results/slurm-%j.err
set -euo pipefail
R=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical11
A=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/alias-canonical05
B=/scratch/sounio-stdlib-wave3-20261027
SEED="$B/compiler-resumed/generic-module-native/source/bin/souc-lean-single-x86_64"
RAW="$B/root-integrated01/results/madaros"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
T="$P/tooling"
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$T/root/usr/lib/x86_64-linux-gnu"
export MAGIC="$T/root/usr/lib/file/magic.mgc"
ulimit -s524288
ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
sha256sum -c "$A/results/COMPILERS.sha256" > "$R/results/COMPILERS.before.verify"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/INPUTS.sha256"
for mode in before after;do
 S="$R/source-$mode"
 cp -a "$A/source" "$S"
 if test "$mode" = before;then cp "$A/input/self-hosted/check/specializer.sio" "$S/self-hosted/check/specializer.sio";else cp "$R/input/self-hosted/check/specializer.sio" "$S/self-hosted/check/specializer.sio";fi
 cp "$R/input/tests/fixtures/wave4_alias_canonicalization/impl_shadow_driver.sio.in" "$S/impl_shadow_driver.sio"
 export SOUNIO_STDLIB_PATH="$S/self-hosted"
 cd "$S"
 sha256sum "$S/self-hosted/check/specializer.sio" "$S/impl_shadow_driver.sio" > "$R/results/$mode-SOURCE.sha256"
 /usr/bin/time -v timeout 300 "$SEED" "$S/impl_shadow_driver.sio" "$R/results/$mode.elf" > "$R/results/$mode-seed.stdout" 2> "$R/results/$mode-seed.stderr"
 chmod +x "$R/results/$mode.elf"
 sha256sum "$R/results/$mode.elf" > "$R/results/$mode-ELF.sha256"
 status=0
 /usr/bin/time -v timeout 30 "$R/results/$mode.elf" "$R/input/tests/fixtures/wave4_alias_canonicalization/generic_shadow.sio" > "$R/results/$mode.stdout" 2> "$R/results/$mode.stderr" || status=$?
 echo "$status" > "$R/results/$mode-STATUS"
 if test "$mode" = before;then test "$status" = 1;grep -Fx ALIAS_IMPL_PARAMETER_CAPTURE "$R/results/$mode.stdout";else test "$status" = 0;grep -Fx ALIAS_IMPL_PARAMETER_PRESERVED "$R/results/$mode.stdout";fi
 sha256sum -c "$R/results/$mode-SOURCE.sha256" > "$R/results/$mode-SOURCE.final.verify"
 sha256sum -c "$R/results/$mode-ELF.sha256" > "$R/results/$mode-ELF.final.verify"
done
status=0
/usr/bin/time -v timeout 600 "$RAW" check "$R/source-after/self-hosted/check/specializer.sio" > "$R/results/specializer-check.stdout" 2> "$R/results/specializer-check.stderr" || status=$?
echo "$status" > "$R/results/SPECIALIZER_CHECK_STATUS"
sha256sum -c "$R/results/INPUTS.sha256" > "$R/results/INPUTS.final.verify"
sha256sum -c "$A/results/COMPILERS.sha256" > "$R/results/COMPILERS.final.verify"
test "$status" = 0
echo ALIAS_IMPL_GENERIC_SHADOW_PAIRED_ONLY_OK
