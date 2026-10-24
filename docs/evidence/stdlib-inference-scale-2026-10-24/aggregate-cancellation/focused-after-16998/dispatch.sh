#!/bin/bash
# Submit/invoke ONLY under Slurm UID1000 after the owner releases a corrected ELF.
# Source is the preserved unannotated failing program, not a diagnostic rewrite.
set -euo pipefail
export PATH=/usr/bin:/bin
[[ ${SLURM_JOB_ID:-} && $(id -u) == 1000 ]] || exit 64
[[ $# == 2 ]] || exit 64
compiler=$1
out=$2
src=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_guard_focused01/source
[[ -x $compiler && $out == /scratch/* && ! -e $out ]] || exit 64
mkdir -p "$out"
cd "$src"
ulimit -s 524288
sha256sum "$compiler" stdlib/stats/regression/{count_fit_trace.sio,dynamic_count_glm.sio,dynamic_dispersion.sio,dynamic_count_profile_bootstrap.sio,nb2_likelihood.sio,nb2_log_increment.sio,count_profile_embedding.sio} tests/run-pass/stdlib_scale_count_{fit_trace,profile_traced,profile_nested_bootstrap,rank_difference,rank_status,likelihood_difference,normalizer_cache,cached_likelihood,profile_cache,cached_rank_status,cached_rank_difference}.sio tests/run-pass/stdlib_deep_stats_nb2_likelihood.sio tests/run-pass/stdlib_deep_poisson_zero_likelihood.sio tests/reference/{deep_stats_count_profile_bootstrap_reference,deep_count_sampling_reference,count_likelihood_difference_reference,deep_stats_nb2_likelihood_reference,deep_poisson_zero_likelihood_reference}.cpp /scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02/case-{3,11,13}.stdout > "$out/INPUTS.sha256"
printf '%s\n' "$src" > "$out/ORIGINAL_SOURCE.txt"
# Fail closed. A partial stdout is never fed into an acceptance claim.
rc=0
timeout 180 g++ -std=c++23 -O2 tests/reference/deep_stats_count_profile_bootstrap_reference.cpp -lquadmath -o "$out/reference" > "$out/cpp.compile.stdout" 2> "$out/cpp.compile.stderr" || rc=$?
printf '%s\n' "$rc" > "$out/cpp.compile.status"
[[ $rc == 0 ]] || exit "$rc"
for name in stdlib_scale_count_aggregate_cancellation stdlib_scale_count_cached_rank_far stdlib_scale_count_cached_rank_status stdlib_scale_count_normalizer_cache stdlib_scale_count_fit_trace stdlib_scale_count_profile_cache stdlib_scale_count_profile_traced stdlib_scale_count_profile_nested_bootstrap stdlib_scale_count_rank_status stdlib_scale_count_rank_difference stdlib_scale_count_likelihood_difference stdlib_scale_count_cached_rank_difference stdlib_scale_count_cached_likelihood stdlib_deep_stats_nb2_likelihood stdlib_deep_poisson_zero_likelihood; do
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
 if [[ $name == stdlib_scale_count_profile_traced || $name == stdlib_scale_count_profile_nested_bootstrap || $name == stdlib_scale_count_profile_cache ]]; then
  grep -Fq 'DEEP COUNT PROFILE NESTED BOOTSTRAP PASS' "$out/$name.stdout"
 rc=0
  timeout 180 "$out/reference" --nested-single "$out/$name.stdout" > "$out/$name.cpp.stdout" 2> "$out/$name.cpp.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$name.cpp.status"
 [[ $rc == 0 ]] || exit "$rc"
 fi
done
timeout 240 g++ -std=c++23 -O2 tests/reference/count_likelihood_difference_reference.cpp -o "$out/paired_reference" > "$out/paired.compile.stdout" 2> "$out/paired.compile.stderr"
for name in stdlib_scale_count_rank_difference stdlib_scale_count_likelihood_difference stdlib_scale_count_cached_rank_difference; do
 rc=0
 timeout 180 "$out/paired_reference" --native "$out/$name.stdout" /scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02/case-3.stdout /scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02/case-11.stdout /scratch/sounio-stdlib-advanced-20261023/lr_diagnostic02/case-13.stdout > "$out/$name.cpp.stdout" 2> "$out/$name.cpp.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$name.cpp.status"
 [[ $rc == 0 ]] || exit "$rc"
done
timeout 240 g++ -std=c++23 -O2 tests/reference/deep_stats_nb2_likelihood_reference.cpp -lquadmath -o "$out/likelihood_reference" > "$out/likelihood_reference.compile.stdout" 2> "$out/likelihood_reference.compile.stderr"
for name in stdlib_scale_count_cached_likelihood stdlib_deep_stats_nb2_likelihood; do
 rc=0
 timeout 300 "$out/likelihood_reference" "$out/$name.stdout" > "$out/$name.cpp.stdout" 2> "$out/$name.cpp.stderr" || rc=$?
 printf '%s\n' "$rc" > "$out/$name.cpp.status"
 [[ $rc == 0 ]] || exit "$rc"
done
timeout 180 g++ -std=c++23 -O2 tests/reference/deep_poisson_zero_likelihood_reference.cpp -o "$out/zero_reference" > "$out/zero_reference.compile.stdout" 2> "$out/zero_reference.compile.stderr"
timeout 180 "$out/zero_reference" "$out/stdlib_deep_poisson_zero_likelihood.stdout" > "$out/zero.cpp.stdout" 2> "$out/zero.cpp.stderr"
printf '0\n' > "$out/zero.cpp.status"
# Arithmetic preservation of the refactored legacy core against the frozen prior candidate.
cmp "$out/stdlib_scale_count_rank_difference.stdout" /scratch/sounio-stdlib-advanced-20261023/normalizer_cache01/artifacts/stdlib_scale_count_rank_difference.stdout
cmp "$out/stdlib_scale_count_likelihood_difference.stdout" /scratch/sounio-stdlib-advanced-20261023/normalizer_cache01/artifacts/stdlib_scale_count_likelihood_difference.stdout
printf '0\n' > "$out/legacy_bitwise.status"
sha256sum "$out/"*.elf "$out/reference" > "$out/ELFS.sha256"
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.after.verify"
timeout 180 g++ -std=c++23 -O2 tests/reference/count_aggregate_cancellation_reference.cpp -o "$out/aggregate_reference" >"$out/aggregate.compile.stdout" 2>"$out/aggregate.compile.stderr"
"$out/aggregate_reference" "$out/stdlib_scale_count_aggregate_cancellation.stdout" >"$out/aggregate.cpp.stdout" 2>"$out/aggregate.cpp.stderr"
for mode in ownerfix effectbridge;do "$out/aggregate_reference" --before "/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation01/artifacts/$mode.native.stdout" >"$out/$mode.before.cpp.stdout" 2>"$out/$mode.before.cpp.stderr";done
timeout 240 g++ -std=c++23 -O2 tests/reference/count_cached_rank_reference.cpp -o "$out/far_reference" >"$out/far.compile.stdout" 2>"$out/far.compile.stderr"
"$out/far_reference" "$out/stdlib_scale_count_cached_rank_far.stdout" >"$out/far.cpp.stdout" 2>"$out/far.cpp.stderr"
# Same final regression source against frozen pre-guard1132 library; compile then
# native assertion failure is expected. Keep it separate from accepted output.
before=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_guard_focused01/before
(cd "$before"; SOUNIO_STDLIB_PATH="$before/stdlib" "$compiler" build witness.sio --target native -o "$out/before.elf") >"$out/before.compile.stdout" 2>"$out/before.compile.stderr"
chmod +x "$out/before.elf"
rc=0
"$out/before.elf" >"$out/before.native.stdout" 2>"$out/before.native.stderr" || rc=$?
printf '%s\n' "$rc" >"$out/before.native.status"
[[ $rc == 1 ]]
sha256sum "$compiler" tests/run-pass/stdlib_scale_count_aggregate_cancellation.sio tests/reference/count_aggregate_cancellation_reference.cpp stdlib/stats/regression/nb2_likelihood.sio >"$out/NEW_INPUTS.sha256"
sha256sum -c "$out/INPUTS.sha256" >"$out/INPUTS.final.verify"
printf 'AGGREGATE_GUARD_FOCUSED_PASS\n' >"$out/STATUS"
