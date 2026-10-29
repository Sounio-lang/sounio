#!/bin/bash
#SBATCH --job-name=loop-checkpoint-component
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/loop-full-trace02/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/loop-full-trace02/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R="$B/compiler-resumed/loop-full-trace02"
S="$R/source"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
export PATH="$P/tools:$P/tooling/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$P/tooling/root/usr/lib/x86_64-linux-gnu"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
RAW="$B/root-integrated02/results/madaros"
ulimit -s524288;ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
cd "$S"
find self-hosted tests/fixtures/wave4_loop_checkpoint_elision -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/SOURCE.sha256"
printf '8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932  %s\n' "$RAW" > "$R/results/RAW.sha256"
sha256sum -c "$R/results/RAW.sha256" > "$R/results/RAW.before.verify"
SEED="$B/compiler-resumed/generic-module-native/source/bin/souc-lean-single-x86_64"
printf '9cfb6a970d73718b09091331dc514d015e3d4e6f402ee756cdc7f7b0cc55110a  %s\n' "$SEED" > "$R/results/SEED.sha256"
sha256sum -c "$R/results/SEED.sha256" > "$R/results/SEED.before.verify"
status=0
/usr/bin/time -v timeout 300 "$RAW" check tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_probe.sio > "$R/results/check.stdout" 2> "$R/results/check.stderr" || status=$?
echo "$status" > "$R/results/CHECK_STATUS"
test "$status" = 0
status=0
/usr/bin/time -v timeout 300 "$SEED" tests/fixtures/wave4_loop_checkpoint_elision/nv2_trace_full_probe.sio "$R/results/component.elf" > "$R/results/compile.stdout" 2> "$R/results/compile.stderr" || status=$?
echo "$status" > "$R/results/COMPILE_STATUS"
if test "$status" = 0;then
 chmod +x "$R/results/component.elf";sha256sum "$R/results/component.elf" > "$R/results/ELF.sha256"
 for mode in valid badargs badname badregion badparam budget;do
  status=0
  env SOUNIO_NV2_IR_TRACE_FULL=1 timeout 30 "$R/results/component.elf" "$mode" > "$R/results/$mode.stdout" 2> "$R/results/$mode.stderr" || status=$?
  echo "$status" > "$R/results/$mode-STATUS"
 done
 env -u SOUNIO_NV2_IR_TRACE_FULL timeout 30 "$R/results/component.elf" valid > "$R/results/legacy.stdout" 2> "$R/results/legacy.stderr"
 printf 'LEGACY_CONTROL_BYTES\n' > "$R/results/legacy.expected"
 cmp "$R/results/legacy.expected" "$R/results/legacy.stdout" > "$R/results/legacy.verify"
fi
sha256sum -c "$R/results/SOURCE.sha256" > "$R/results/SOURCE.final.verify"
sha256sum -c "$R/results/RAW.sha256" > "$R/results/RAW.final.verify"
sha256sum -c "$R/results/SEED.sha256" > "$R/results/SEED.final.verify"

test "$(cat "$R/results/COMPILE_STATUS")" = 0
g++ -std=c++23 -O2 -Wall -Wextra -Werror "$R/input/reference.cpp" -o "$R/results/reference" > "$R/results/reference-build.stdout" 2> "$R/results/reference-build.stderr"
"$R/results/reference" --probe "$R/results/valid.stdout" > "$R/results/reference-valid.stdout" 2> "$R/results/reference-valid.stderr"
for control in truncate duplicate total args bits name float-word unknown end;do
 "$R/results/reference" --mutate "$control" "$R/results/valid.stdout" "$R/results/control-$control.stdout"
 status=0
 "$R/results/reference" --probe "$R/results/control-$control.stdout" > "$R/results/reject-$control.stdout" 2> "$R/results/reject-$control.stderr" || status=$?
 echo "$status" > "$R/results/reject-$control-STATUS"
 test "$status" = 1
done
for pair in badargs:8 badname:7 badregion:6 badparam:4 budget:5;do
 mode="${pair%:*}";reason="${pair#*:}"
 "$R/results/reference" --refusal "$reason" "$R/results/$mode.stdout" > "$R/results/refusal-$mode.stdout" 2> "$R/results/refusal-$mode.stderr"
 status=0
 "$R/results/reference" --check "$R/results/$mode.stdout" > "$R/results/refusal-$mode-check.stdout" 2> "$R/results/refusal-$mode-check.stderr" || status=$?
 test "$status" = 1
done
sha256sum "$R/results/reference" > "$R/results/REFERENCE.sha256"
echo NV2_FULL_TRACE_FORMATTER_COMPONENT_ONLY_OK > "$R/results/STATUS"
