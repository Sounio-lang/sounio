#!/bin/bash
# Submit/invoke ONLY under Slurm UID1000 after the owner releases a corrected ELF.
# Source is the preserved unannotated failing program, not a diagnostic rewrite.
set -euo pipefail
export PATH=/usr/bin:/bin
[[ ${SLURM_JOB_ID:-} && $(id -u) == 1000 ]] || exit 64
[[ $# == 2 ]] || exit 64
compiler=$1
out=$2
src=/scratch/sounio-stdlib-advanced-20261023/trace_focused02/source
[[ -x $compiler && $out == /scratch/* && ! -e $out ]] || exit 64
mkdir -p "$out"
cd "$src"
ulimit -s 524288
sha256sum "$compiler" stdlib/stats/regression/{count_fit_trace.sio,dynamic_count_glm.sio,dynamic_dispersion.sio,dynamic_count_profile_bootstrap.sio,nb2_likelihood.sio,nb2_log_increment.sio,count_profile_embedding.sio} tests/run-pass/stdlib_scale_count_{fit_trace,profile_traced,profile_nested_bootstrap}.sio tests/reference/{deep_stats_count_profile_bootstrap_reference,deep_count_sampling_reference}.cpp > "$out/INPUTS.sha256"
printf '%s\n' "$src" > "$out/ORIGINAL_SOURCE.txt"
# Fail closed. A partial stdout is never fed into an acceptance claim.
rc=0
timeout 180 g++ -std=c++23 -O2 tests/reference/deep_stats_count_profile_bootstrap_reference.cpp -lquadmath -o "$out/reference" > "$out/cpp.compile.stdout" 2> "$out/cpp.compile.stderr" || rc=$?
printf '%s\n' "$rc" > "$out/cpp.compile.status"
[[ $rc == 0 ]] || exit "$rc"
for name in stdlib_scale_count_fit_trace stdlib_scale_count_profile_traced stdlib_scale_count_profile_nested_bootstrap; do
 rc=0
 env -i PATH=/usr/bin:/bin SOUNIO_STDLIB_PATH="$src/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 timeout 180 "$compiler" check "tests/run-pass/$name.sio" > "$out/$name.check.stdout" 2> "$out/$name.check.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$name.check.status"
 [[ $rc == 0 ]] || exit "$rc"
 rc=0
 env -i PATH=/usr/bin:/bin SOUNIO_STDLIB_PATH="$src/stdlib" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 timeout 240 "$compiler" build "tests/run-pass/$name.sio" --target native -o "$out/$name.elf" > "$out/$name.compile.stdout" 2> "$out/$name.compile.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$name.compile.status"
 [[ $rc == 0 ]] || exit "$rc"
 chmod +x "$out/$name.elf"
 rc=0
 env -i PATH=/usr/bin:/bin /usr/bin/time -v -o "$out/$name.resource" timeout 300 "$out/$name.elf" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$name.native.status"
 [[ $rc == 0 ]] || exit "$rc"
 markers=0
 while IFS= read -r marker; do
  [[ -n $marker ]] || continue
  grep -Fq -- "$marker" "$out/$name.stdout"
  markers=$((markers+1))
 done < <(sed -nE 's|^//@ expect-stdout(-contains)?: ||p' "tests/run-pass/$name.sio")
 [[ $markers -gt 0 ]]
 printf '0\n' > "$out/$name.marker.status"
 if [[ $name != stdlib_scale_count_fit_trace ]]; then
  grep -Fq 'DEEP COUNT PROFILE NESTED BOOTSTRAP PASS' "$out/$name.stdout"
 rc=0
  timeout 180 "$out/reference" --nested-single "$out/$name.stdout" > "$out/$name.cpp.stdout" 2> "$out/$name.cpp.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$name.cpp.status"
 [[ $rc == 0 ]] || exit "$rc"
 fi
done
sha256sum "$out/"*.elf "$out/reference" > "$out/ELFS.sha256"
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.after.verify"
printf 'ORIGINAL_TRACED_CORRECTED_COMPILER_PASS\n' > "$out/STATUS"
