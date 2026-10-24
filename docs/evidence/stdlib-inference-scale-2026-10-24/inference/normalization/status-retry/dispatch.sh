#!/bin/bash
set -uo pipefail
run=$(cd "$(dirname "$0")" && pwd)
cd "$run/source" || exit 65
export SOUNIO_STDLIB_PATH="$run/source/stdlib" SOUNIO_SOUC_ENGINE=madaros
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-advanced-20261023/compiler/artifacts/madaros.elf
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_MADAROS_CACHE="$run/cache" TMPDIR="$run/tmp"
ulimit -s 524288
find stdlib tests scripts examples -type f -print0 | sort -z | xargs -0 sha256sum > "$run/artifacts/INPUTS.sha256"
sha256sum "$MADAROS_RAW_BIN" > "$run/artifacts/COMPILER.sha256"
printf 'kind\tsource\texit_status\n' > "$run/artifacts/RESULTS.tsv"
fail=0
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$run/artifacts/RESULTS.tsv"; [[ "$3" == 0 ]] || fail=1; }
while IFS= read -r src; do
 [[ -n "$src" ]] || continue
 name=$(basename "$src" .sio)
 rc=0
 timeout 180 bin/souc check "$src" > "$run/artifacts/$name.check.log" 2>&1 || rc=$?
 record check "$src" "$rc"
 [[ "$rc" == 0 ]] || continue
 [[ "$src" == tests/run-pass/* || "$src" == examples/* ]] || continue
 timeout 240 bin/souc compile "$src" -o "$run/artifacts/$name.elf" > "$run/artifacts/$name.compile.log" 2>&1 || rc=$?
 record compile "$src" "$rc"
 [[ "$rc" == 0 ]] || continue
 sha256sum "$run/artifacts/$name.elf" > "$run/artifacts/$name.elf.sha256"
 /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$run/artifacts/$name.resource" timeout 300 "$run/artifacts/$name.elf" > "$run/artifacts/$name.stdout" 2> "$run/artifacts/$name.stderr" || rc=$?
 record native "$src" "$rc"
 if [[ "$rc" == 0 ]]; then
  rc=0
  expected=$(sed -nE 's|^//@ expect-stdout(-contains)?: ||p' "$src")
  if [[ -z "$expected" ]]; then
   case "$name" in display_headless) expected=DISPLAY_HEADLESS_PASS;; viz_replay_deterministic) expected=VIZ_REPLAY_DETERMINISTIC_PASS;; viz_app_frame) expected=VIZ_APP_FRAME_PASS;; esac
  fi
  [[ -n "$expected" ]] || rc=1
  while IFS= read -r marker; do [[ -z "$marker" ]] || grep -Fq -- "$marker" "$run/artifacts/$name.stdout" || rc=1; done <<< "$expected"
  record marker "$src" "$rc"
 fi
done < "$run/fixtures.txt"

cpp_compare() {
 local stem=$1 fixture=$2 mode=${3:-} rc=0
 [[ -f "$run/source/tests/reference/$stem.cpp" && -f "$run/artifacts/$fixture.stdout" ]] || return 0
 if [[ ! -x "$run/artifacts/$stem" ]]; then
  timeout 180 g++ -std=c++23 -O2 -Wall -Wextra "tests/reference/$stem.cpp" -lquadmath -o "$run/artifacts/$stem" > "$run/artifacts/$stem.cpp.log" 2>&1 || rc=$?
  record cpp-compile "$stem" "$rc"
  [[ "$rc" == 0 ]] || return 0
 fi
 if [[ -n "$mode" ]]; then timeout 180 "$run/artifacts/$stem" "$mode" "$run/artifacts/$fixture.stdout" > "$run/artifacts/$stem.$fixture.concordance.stdout" 2> "$run/artifacts/$stem.$fixture.concordance.stderr" || rc=$?
 else timeout 180 "$run/artifacts/$stem" "$run/artifacts/$fixture.stdout" > "$run/artifacts/$stem.$fixture.concordance.stdout" 2> "$run/artifacts/$stem.$fixture.concordance.stderr" || rc=$?; fi
 record concordance "$stem:$fixture" "$rc"
}
cpp_compare deep_stats_count_profile_bootstrap_reference stdlib_deep_stats_count_profile_bootstrap
cpp_compare deep_stats_dynamic_robust_reference stdlib_deep_stats_dynamic_robust
cpp_compare deep_stats_dynamic_robust_contrast_reference stdlib_deep_stats_dynamic_robust_contrast
cpp_compare deep_count_dataset_reference stdlib_deep_count_dataset_reference
cpp_compare deep_count_dataset_reference viz_dynamic_count_workflow --workflow
cpp_compare deep_viz_scientific_document stdlib_deep_viz_scientific_document
cpp_compare deep_viz_scientific_document stdlib_deep_viz_count_diagnostics --count
cpp_compare deep_ode_diagnostic_observer_reference stdlib_deep_ode_diagnostic_observer
cpp_compare deep_ode_diagnostic_observer_reference viz_dynamic_ode_diagnostics --figure


cpp_compare deep_checked_log_slots_reference stdlib_deep_checked_log_slots
cpp_compare deep_poisson_zero_likelihood_reference stdlib_deep_poisson_zero_likelihood
if [[ -x "$run/artifacts/deep_checked_log_slots_reference" ]]; then
 rc=0
 "$run/artifacts/deep_checked_log_slots_reference" --conversion-selftest > "$run/artifacts/checked-log-conversion-selftest.stdout" 2> "$run/artifacts/checked-log-conversion-selftest.stderr" || rc=$?
 record cpp-controls checked-log-conversion-selftest "$rc"
fi
cpp_compare deep_stats_count_glm_reference stdlib_deep_stats_count_glm
cpp_compare deep_stats_count_increment_reference stdlib_deep_stats_count_glm
if [[ -f "$run/artifacts/display_font_orientation.stdout" ]]; then
 rc=0
 timeout 180 g++ -std=c++23 -O2 -Wall -Wextra tests/reference/display_font_orientation_reference.cpp -o "$run/artifacts/font_oracle" > "$run/artifacts/font_oracle.cpp.log" 2>&1 || rc=$?
 record cpp-compile display_font_orientation_reference "$rc"
 if [[ "$rc" == 0 ]]; then
  "$run/artifacts/font_oracle" --canonical tests/reference/data/font8x8_basic.h "$run/artifacts/display_font_orientation.stdout" > "$run/artifacts/font_canonical.stdout" 2> "$run/artifacts/font_canonical.stderr" || rc=$?
  record concordance font-canonical "$rc"
 fi
fi

if [[ -x "$run/artifacts/deep_stats_count_profile_bootstrap_reference" ]]; then
 rc=0
 "$run/artifacts/deep_stats_count_profile_bootstrap_reference" --mutant-fixed-alpha "$run/artifacts/stdlib_deep_stats_count_profile_bootstrap.stdout" > "$run/artifacts/profile-fixed-mutant.stdout" 2> "$run/artifacts/profile-fixed-mutant.stderr" || rc=$?
 [[ "$rc" == 2 ]] && record mutant-rejection fixed-alpha 0 || record mutant-rejection fixed-alpha 1
fi

cpp_compare normalization_into_reference stdlib_scale_normalization_into
sha256sum -c "$run/artifacts/INPUTS.sha256" > "$run/artifacts/INPUTS.after.verify" 2> "$run/artifacts/INPUTS.after.stderr" || fail=1
sha256sum "$MADAROS_RAW_BIN" > "$run/artifacts/COMPILER.after.sha256"
cmp "$run/artifacts/COMPILER.sha256" "$run/artifacts/COMPILER.after.sha256" || fail=1
find "$run/artifacts" -maxdepth 1 -type f \( -name "*.elf" -o -name "*.stdout" -o -name "*.stderr" \) -print0 | sort -z | xargs -0 sha256sum > "$run/artifacts/OUTPUTS.sha256"
printf '%s\n' "$fail" > "$run/artifacts/STATUS"
exit "$fail"
