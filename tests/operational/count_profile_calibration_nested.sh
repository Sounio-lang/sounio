#!/usr/bin/env bash
# Operational qualification controls only. No campaign, submission or green
# functional receipt is produced. All C++/Sounio execution requires Slurm UID1000.
# Args: SOURCE_ROOT BASELINE_CPP FROZEN_MATH_ROOT COMPILER OLD_RESULTS OUTPUT
set -euo pipefail
[[ $# == 6 && ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ && $(id -u) == 1000 ]] || exit 65
src=$(realpath "$1");baseline=$(realpath "$2");math=$(realpath "$3");compiler=$(realpath "$4");old_results=$(realpath "$5");out=$6
[[ ! -e "$out" && -x "$compiler" ]] || exit 65
mkdir -p "$out"/{tools,generated,controls,tmp,cache}
out=$(realpath "$out")
case "$out/" in "$src/"*|"$math/"*) exit 65;; esac
finish(){ local rc=$?;printf '%s\n' "$rc" > "$out/STATUS"; }
trap finish EXIT
ulimit -s 524288
export TMPDIR="$out/tmp" SOUNIO_STDLIB_PATH="$math/stdlib" MADAROS_RAW_BIN="$compiler" SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_MADAROS_CACHE="$out/cache"
printf 'kind\tsource\texit_status\n' > "$out/RESULTS.tsv"
expect(){ local label=$1 wanted=$2 rc=0;shift 2;"$@" > "$out/$label.stdout" 2> "$out/$label.stderr" || rc=$?;printf '%s\n' "$rc" > "$out/$label.status";[[ $rc == "$wanted" ]];printf 'control\t%s:expected=%s:actual=%s\t0\n' "$label" "$wanted" "$rc" >> "$out/RESULTS.tsv"; }
sha256sum "$baseline" "$compiler" "$old_results" "$src/tests/reference/count_profile_calibration.cpp" "$src/tests/reference/deep_count_sampling_reference.cpp" "$src/tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio" "$src/scripts/stdlib/count_profile_calibration.sh" > "$out/INPUTS.sha256"
find "$math/stdlib" -type f -print0 | sort -z | xargs -0 sha256sum >> "$out/INPUTS.sha256"
expect bash-syntax 0 bash -n "$src/scripts/stdlib/count_profile_calibration.sh"
expect cpp-baseline 0 g++ -std=c++23 -O2 -I "$src/tests/reference" "$baseline" -lquadmath -o "$out/tools/baseline"
expect cpp-current 0 g++ -std=c++23 -O2 "$src/tests/reference/count_profile_calibration.cpp" -lquadmath -o "$out/tools/current"
# Independent C++ sampler is unchanged. Compare complete old-mode driver and
# metadata bytes across all9cells; then compare nested raw data/options/seeds.
cells=('16 0' '16 .25' '16 1' '64 0' '64 .25' '64 1' '256 0' '256 .25' '256 1')
cell=0
for pair in "${cells[@]}";do
 read -r n alpha <<< "$pair"
 read -r generation bootstrap < <("$out/tools/current" --seeds "$cell")
 for variant in baseline current nested traced;do
  tool="$out/tools/current";mode=--generate
  [[ $variant != baseline ]] || tool="$out/tools/baseline"
  [[ $variant != nested ]] || mode=--generate-nested
  [[ $variant != traced ]] || mode=--generate-nested-traced
  expect "generate-$variant-$cell" 0 "$tool" "$mode" "$src/tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio" "$out/generated/$variant-$cell" "$n" "$alpha" 31 16 0 "$generation" "$bootstrap"
 done
 cmp "$out/generated/baseline-$cell/driver.sio" "$out/generated/current-$cell/driver.sio"
 cmp "$out/generated/baseline-$cell/datasets.tsv" "$out/generated/current-$cell/datasets.tsv"
 cmp "$out/generated/current-$cell/datasets.tsv" "$out/generated/nested-$cell/datasets.tsv"
 cmp "$out/generated/current-$cell/datasets.tsv" "$out/generated/traced-$cell/datasets.tsv"
 for token in 'let data:\[i64;4096\]=' 'let options=CountProfileBootstrapOptions{' 'let initial_rng=count_rng(';do
  grep "$token" "$out/generated/current-$cell/driver.sio" > "$out/controls/old-line"
  grep "$token" "$out/generated/nested-$cell/driver.sio" > "$out/controls/new-line"
  cmp "$out/controls/old-line" "$out/controls/new-line"
  grep "$token" "$out/generated/traced-$cell/driver.sio" > "$out/controls/traced-line"
  cmp "$out/controls/old-line" "$out/controls/traced-line"
 done
 printf 'byte-identity\tcell-%s-old-mode-and-nested-data-options-seeds\t0\n' "$cell" >> "$out/RESULTS.tsv"
 cell=$((cell+1))
done
expect seeds-invalid 2 "$out/tools/current" --seeds 9
expect old-B2-rejected 2 "$out/tools/current" --generate "$src/tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio" "$out/generated/invalid-old-B2" 16 .25 2 1 0 24000000 34000000
expect nested-first-overflow-rejected 2 "$out/tools/current" --generate-nested "$src/tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio" "$out/generated/invalid-first" 16 .25 31 16 2147483647 24000000 34000000
expect nested-trailing-token-rejected 2 "$out/tools/current" --generate-nested "$src/tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio" "$out/generated/invalid-token" 16junk .25 31 16 0 24000000 34000000
expect nested-B2-generation 0 "$out/tools/current" --generate-nested "$src/tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio" "$out/generated/smoke" 16 .25 2 1 0 24000000 34000000
expect nested-B2-check 0 "$math/bin/souc" check "$out/generated/smoke/driver.sio"
expect nested-B2-compile 0 "$math/bin/souc" compile "$out/generated/smoke/driver.sio" -o "$out/generated/smoke/driver.elf"
chmod +x "$out/generated/smoke/driver.elf"
sha256sum "$out/generated/smoke/driver.elf" > "$out/generated/smoke/ELF.sha256"
expect native-invalid-env64 64 env COUNT_PROFILE_CALIBRATION_CASE=64 "$out/generated/smoke/driver.elf"
expect native-invalid-env-batch1 64 env COUNT_PROFILE_CALIBRATION_CASE=1 "$out/generated/smoke/driver.elf"
expect native-invalid-env-leadingzero 64 env COUNT_PROFILE_CALIBRATION_CASE=00 "$out/generated/smoke/driver.elf"
expect nested-B2-native 0 env COUNT_PROFILE_CALIBRATION_CASE=0 "$out/generated/smoke/driver.elf"
cp "$out/nested-B2-native.stdout" "$out/generated/smoke/native.stdout"
printf '%s\n' 'COUNT PROFILE CALIBRATION BATCH COMPLETE' >> "$out/generated/smoke/native.stdout"
expect nested-B2-summary 0 "$out/tools/current" --summarize-nested 2 1 0 "$out/generated/smoke/native.stdout"
grep -Eq '^CAL_RESULT 0 0 [0-2] 2 -1 0 ' "$out/nested-B2-native.stdout"
grep -Eq '^CAL_WORK 0 [0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+$' "$out/nested-B2-native.stdout"
grep -Fv 'CAL_WORK ' "$out/generated/smoke/native.stdout" > "$out/controls/missing-work.stdout"
expect missing-work-rejected 2 "$out/tools/current" --summarize-nested 2 1 0 "$out/controls/missing-work.stdout"
grep -Fv 'COUNT PROFILE CALIBRATION BATCH COMPLETE' "$out/generated/smoke/native.stdout" > "$out/controls/incomplete.stdout"
expect incomplete-summary-rejected 2 "$out/tools/current" --summarize-nested 2 1 0 "$out/controls/incomplete.stdout"
# Optional early native traced coverage requires an explicitly frozen math tree
# containing that additive entry. This is still a tiny B2 test, never a campaign.
if [[ ${COUNT_PROFILE_TEST_TRACED:-0} == 1 ]];then
 expect traced-B2-generation 0 "$out/tools/current" --generate-nested-traced "$src/tests/run-pass/stdlib_deep_stats_count_profile_bootstrap.sio" "$out/generated/traced-smoke" 16 .25 2 1 0 24000000 34000000
 expect traced-B2-check 0 "$math/bin/souc" check "$out/generated/traced-smoke/driver.sio"
 expect traced-B2-compile 0 "$math/bin/souc" compile "$out/generated/traced-smoke/driver.sio" -o "$out/generated/traced-smoke/driver.elf"
 chmod +x "$out/generated/traced-smoke/driver.elf"
 expect traced-B2-native 0 env COUNT_PROFILE_CALIBRATION_CASE=0 "$out/generated/traced-smoke/driver.elf"
 cp "$out/traced-B2-native.stdout" "$out/generated/traced-smoke/native.stdout"
 printf '%s\n' 'COUNT PROFILE CALIBRATION BATCH COMPLETE' >> "$out/generated/traced-smoke/native.stdout"
 expect traced-B2-summary 0 "$out/tools/current" --summarize-nested-traced 2 1 0 "$out/generated/traced-smoke/native.stdout"
 grep -E '^CAL_(RESULT|P) ' "$out/nested-B2-native.stdout" > "$out/controls/off-result"
 grep -E '^CAL_(RESULT|P) ' "$out/traced-B2-native.stdout" > "$out/controls/on-result"
 cmp "$out/controls/off-result" "$out/controls/on-result"
 grep -Fv 'CAL_TRACE ' "$out/generated/traced-smoke/native.stdout" > "$out/controls/missing-trace.stdout"
 expect missing-trace-rejected 2 "$out/tools/current" --summarize-nested-traced 2 1 0 "$out/controls/missing-trace.stdout"
 # Explicit zero trace budget: preserve the actual unavailable result/status;
 # never fill or infer work counters to turn this into an available observation.
 sed -e 's/{count_fit_trace_slot,/{count_fit_trace_slot,count_fit_trace_bounded,/' -e 's/var fit_trace=count_fit_trace_slot()/var fit_trace=count_fit_trace_bounded(0)/' "$out/generated/traced-smoke/driver.sio" > "$out/generated/traced-smoke/zero-limit.sio"
 expect trace-zero-limit-check 0 "$math/bin/souc" check "$out/generated/traced-smoke/zero-limit.sio"
 expect trace-zero-limit-compile 0 "$math/bin/souc" compile "$out/generated/traced-smoke/zero-limit.sio" -o "$out/generated/traced-smoke/zero-limit.elf"
 chmod +x "$out/generated/traced-smoke/zero-limit.elf"
 expect trace-zero-limit-native 0 env COUNT_PROFILE_CALIBRATION_CASE=0 "$out/generated/traced-smoke/zero-limit.elf"
 grep -Eq '^CAL_RESULT 0 4 0 0 -1 10 ' "$out/trace-zero-limit-native.stdout"
 cp "$out/trace-zero-limit-native.stdout" "$out/generated/traced-smoke/zero-limit.stdout"
 printf '%s\n' 'COUNT PROFILE CALIBRATION BATCH COMPLETE' >> "$out/generated/traced-smoke/zero-limit.stdout"
 expect trace-zero-limit-summary 0 "$out/tools/current" --summarize-nested-traced 2 1 0 "$out/generated/traced-smoke/zero-limit.stdout"
 grep -Fxq $'available\t0' "$out/trace-zero-limit-summary.stdout"
 grep -Fxq $'unavailable\t1' "$out/trace-zero-limit-summary.stdout"
fi

# Deliberately forged NEGATIVE input: bind actual OLD passing rows to the NEW
# source manifest and compiler. It must fail65 before compiling any generator.
(cd "$src" && find -L bin scripts self-hosted stdlib tests examples -type f -print0 | sort -z | xargs -0 sha256sum) > "$out/controls/current-inputs.sha256"
csha=$(sha256sum "$compiler");msha=$(sha256sum "$out/controls/current-inputs.sha256");rsha=$(sha256sum "$old_results")
cat > "$out/controls/FORGED_OLD_ROWS.receipt" <<RECEIPT
format=stdlib-advanced-functional-v1
status=0
compiler_sha256=${csha%% *}
canonical_inputs_sha256=${msha%% *}
canonical_inputs_path=$out/controls/current-inputs.sha256
results_sha256=${rsha%% *}
results_path=$old_results
RECEIPT
expect old-gate-current-manifest-rejected 65 env COUNT_PROFILE_FUNCTIONAL_GATE_RECEIPT="$out/controls/FORGED_OLD_ROWS.receipt" MADAROS_RAW_BIN="$compiler" bash "$src/scripts/stdlib/count_profile_calibration.sh" "$src" "$out/rejected-campaign" pilot
grep -Fq 'missing/duplicate gate result: check tests/run-pass/' "$out/old-gate-current-manifest-rejected.stderr"
[[ ! -e "$out/rejected-campaign/pilot_nested_B31/tools/compile.resource" ]]
[[ ! -e "$out/rejected-campaign/pilot_B31" && ! -e "$out/rejected-campaign/qualification_B199" ]]
printf 'pre-compile-rejection\told-full02-even-with-current-manifest\t0\n' >> "$out/RESULTS.tsv"
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.after.verify"
sha256sum -c "$out/generated/smoke/ELF.sha256" > "$out/generated/smoke/ELF.after.verify"
find "$out" -type f ! -name OUTPUTS.sha256 ! -name STATUS -print0 | sort -z | xargs -0 sha256sum > "$out/OUTPUTS.sha256"
printf '%s\n' 'NESTED QUALIFICATION OPERATIONAL CONTROLS PASS; no campaign run or valid green receipt created.'
