#!/bin/bash
#SBATCH --job-name=stdlib-worker-canonical-native
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:10:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-canonical-stub-native01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-canonical-stub-native01/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
ulimit -s 524288;ulimit -c 0
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
R=$B/worker-canonical-stub-native01
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated03/results/madaros
trap 'r=$?;echo "$r" > "$R/STATUS";exit "$r"' EXIT
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$R/INPUTS.sha256"
for variant in stack frame;do
 S=$R/$variant/source;D=$R/$variant/results;mkdir -p "$R/$variant" "$D"
 cp -a "$B/worker-stack-native-v4/source" "$S"
 C=$S/tests/fixtures/wave4_worker_capability/components
 cat "$R/input/worker_proof.sio" "$R/input/${variant}_body.sio" > "$S/self-hosted/native/worker_proof.sio"
 cp "$R/input/${variant}_driver.sio" "$S/tests/fixtures/wave4_worker_capability/native_stack_driver.sio"
 cd "$S";export SOUNIO_STDLIB_PATH=$S/self-hosted
 find self-hosted stdlib tests -type f -print0|sort -z|xargs -0 sha256sum > "$D/SOURCE.sha256"
 sha256sum "$RAW" bin/souc-lean-single-x86_64 "$C/native_stack_reference.cpp" "$C/runtime_thunk.inc" > "$D/TOOLS.sha256"
 timeout 180 "$RAW" check tests/fixtures/wave4_worker_capability/native_stack_check.sio > "$D/source.check.stdout" 2>&1 &
 a=$!
 timeout 400 bin/souc-lean-single-x86_64 tests/fixtures/wave4_worker_capability/native_stack_driver.sio "$D/producer.elf" > "$D/producer.compile.stdout" 2>&1 &
 b=$!
 wait "$a";wait "$b"
 ! grep -Eq '^error:|^error\[|would have dropped|cannot compile an assignment' "$D/producer.compile.stdout"
 chmod +x "$D/producer.elf"
 timeout 60 "$D/producer.elf" "$D/worker-output.elf" good > "$D/offsets.stdout" 2>&1
 for negative in observed_region_result noncanonical_descriptor2;do
  rc=0;timeout 60 "$D/producer.elf" "$D/$negative.elf" "$negative" > "$D/$negative.stdout" 2>&1 ||rc=$?
  [[ "$rc" == 1 && ! -e "$D/$negative.elf" ]]
  grep -q 'WorkerFn final IR capability or closure proof refused' "$D/$negative.stdout"
 done
 g++ -std=c++23 -O2 -fno-omit-frame-pointer -Wall -Wextra -Werror -pedantic -pthread "$C/native_stack_reference.cpp" -o "$D/native-reference"
 timeout 90 "$D/native-reference" "$D/worker-output.elf" "$D/offsets.stdout" > "$D/native-reference.stdout" 2>&1
 grep -Fxq 'WORKER_STACK_MIXED_OK serial_calls=50000 cloned_calls=150000 threads=4 context_unchanged=true' "$D/native-reference.stdout"
 grep -Fxq 'WORKER_NATIVE_GUARD_OK positive=11 invalid_words=14 wrong_controls=3 writable_segments=PROT_NONE native_thunk=canonical' "$D/native-reference.stdout"
 timeout 60 "$D/producer.elf" "$D/unused-result.elf" unused_region_result > "$D/unused-result-offsets.stdout" 2>&1
 timeout 90 "$D/native-reference" "$D/unused-result.elf" "$D/unused-result-offsets.stdout" > "$D/unused-result-reference.stdout" 2>&1
 grep -Fxq 'WORKER_STACK_MIXED_OK serial_calls=50000 cloned_calls=150000 threads=4 context_unchanged=true' "$D/unused-result-reference.stdout"
 sha256sum -c "$D/SOURCE.sha256" > "$D/SOURCE.final.verify"
 sha256sum -c "$D/TOOLS.sha256" > "$D/TOOLS.final.verify"
 sha256sum "$D/producer.elf" "$D/worker-output.elf" "$D/unused-result.elf" "$D/native-reference" > "$D/ELFS.sha256"
 echo CANONICAL_NATIVE_VARIANT_OK "$variant"
done
sha256sum -c "$R/INPUTS.sha256" > "$R/INPUTS.final.verify"
echo CANONICAL_PLACEHOLDER_NATIVE_STRICT_OK_NO_FRONTEND_ADMISSION
