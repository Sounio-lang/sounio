#!/usr/bin/env bash
# Native Linux worker only. MADAROS_RAW_BIN must identify a source-built compiler.
set -uo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${STDLIB_DEEP_OUT:?set STDLIB_DEEP_OUT to a dedicated artifact directory}
[[ "$out" == /* && -f "${MADAROS_RAW_BIN:?set MADAROS_RAW_BIN}" ]] || exit 64
mkdir -p "$out" || exit 65
[[ -z "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "refusing nonempty output directory" >&2; exit 65; }
cd "$repo" || exit 65
export SOUNIO_STDLIB_PATH="$repo/stdlib" SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288
printf 'kind\tsource\texit_status\n' > "$out/RESULTS.tsv"
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256" || exit 65
find stdlib tests -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256" || exit 65
g++ --version > "$out/CXX_TOOLCHAIN.txt" || exit 65
fail=0
record() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$out/RESULTS.tsv"
    [[ "$3" == 0 ]] || fail=1
}
native() {
    local src=$1 name=$2 rc=0
    timeout 180 bin/souc check "$src" > "$out/$name.check.txt" 2>&1 || rc=$?
    record check "$src" "$rc"
    [[ $rc == 0 ]] || return
    timeout 240 bin/souc compile "$src" -o "$out/$name.elf" > "$out/$name.compile.txt" 2>&1 || rc=$?
    record compile "$src" "$rc"
    [[ $rc == 0 ]] || return
    sha256sum "$src" "$out/$name.elf" > "$out/$name.identities"
    /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 300 "$out/$name.elf" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?
    record native "$src" "$rc"
    if [[ $rc == 0 ]]; then
        local expected marker_rc=0
        expected=$(sed -nE 's|^//@ expect-stdout(-contains)?: ||p' "$src")
        [[ -n "$expected" ]] || marker_rc=1
        while IFS= read -r marker; do
            [[ -n "$marker" ]] || continue
            grep -Fq -- "$marker" "$out/$name.stdout" || marker_rc=1
        done <<< "$expected"
        record stdout-marker "$src" "$marker_rc"
    fi
}
for src in tests/compile-fail/stdlib_deep_*.sio; do
    name=$(basename "$src" .sio); rc=0
    timeout 120 bin/souc check "$src" > "$out/$name.check.txt" 2>&1 || rc=$?
    pattern=linear
    [[ "$name" != *forge* ]] || pattern=private
    [[ "$name" != *callback_private* ]] || pattern=E175
    if [[ $rc != 1 ]] || ! grep -qi "$pattern" "$out/$name.check.txt"; then record negative "$src" 1
    else record negative "$src" 0; fi
done
for src in tests/run-pass/stdlib_deep_*.sio; do native "$src" "$(basename "$src" .sio)"; done
for name in stdlib_scale_normalization_into stdlib_scale_normalization_status stdlib_scale_count_likelihood_difference stdlib_scale_count_profile_nested_bootstrap; do native "tests/run-pass/$name.sio" "$name"; done
# Legacy render paths are replayed in the canonical engine as well as the
# Stage2 full suite: these names exposed nested-label and first-tick defects.
for name in stats_inference_forest viz_bootstrap_forest viz_count_matrix_qc \
    viz_entropy_covariance viz_information_channels viz_scatter_viewport \
    viz_scientific_clip viz_statistical_interval_forest viz_bca_mean \
    viz_permutation_mc viz_axis_first_tick_guard; do
    native "tests/run-pass/$name.sio" "$name"
done
compare() {
    local stem=$1 fixture=$2 rc=0
    if [[ ! -f "$out/$fixture.stdout" ]]; then record concordance "$stem:missing-$fixture" 1; return; fi
    timeout 180 "$out/$stem" "$out/$fixture.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
    record concordance "$stem" "$rc"
}
compare_mode() {
    local stem=$1 mode=$2 fixture=$3 rc=0
    if [[ ! -f "$out/$fixture.stdout" ]]; then record concordance "$stem:$mode:missing-$fixture" 1; return; fi
    timeout 180 "$out/$stem" "$mode" "$out/$fixture.stdout" > "$out/$stem.$fixture.concordance.stdout" 2> "$out/$stem.$fixture.concordance.stderr" || rc=$?
    record concordance "$stem:$mode" "$rc"
}
emit() {
    local stem=$1 suffix=$2 mode=$3 rc=0
    if [[ -n "$mode" ]]; then timeout 180 "$out/$stem" "$mode" > "$out/$stem$suffix.sio" 2> "$out/$stem$suffix.emitter.stderr" || rc=$?
    else timeout 180 "$out/$stem" > "$out/$stem$suffix.sio" 2> "$out/$stem$suffix.emitter.stderr" || rc=$?; fi
    record emit "$stem$suffix" "$rc"
    [[ $rc != 0 ]] || native "$out/$stem$suffix.sio" "$stem$suffix.generated"
}
for src in tests/reference/deep_*.cpp; do
    stem=$(basename "$src" .cpp); rc=0
    timeout 180 g++ -std=c++23 -O2 -Wall -Wextra "$src" -lquadmath -o "$out/$stem" > "$out/$stem.cpp.txt" 2>&1 || rc=$?
    record cpp-compile "$src" "$rc"
    [[ $rc == 0 ]] || continue
    case "$stem" in
      deep_ode_reference|deep_special_reference|deep_ode_dynamic_reference|deep_factor_reference|deep_svd_reference|deep_ros4_reference|deep_kde_reference|deep_dynamic_count_tsv_reference|deep_numeric_select_reference)
        emit "$stem" "" "" ;;
      deep_dynamic_qr|deep_normalization|deep_stats_reference)
        timeout 180 "$out/$stem" > "$out/$stem.stdout" 2> "$out/$stem.stderr" || rc=$?
        record cpp-native "$stem" "$rc" ;;
      deep_stats_dispersion_reference) compare "$stem" stdlib_deep_genomics_shrinkage ;;
      deep_stats_multiple_testing_reference) compare "$stem" stdlib_deep_stats_multiple_testing ;;
      deep_stats_count_glm_reference) compare "$stem" stdlib_deep_stats_count_glm ;;
      deep_stats_nb2_likelihood_reference) compare "$stem" stdlib_deep_stats_nb2_likelihood ;;
      deep_stats_count_fisher_reference) compare "$stem" stdlib_deep_stats_count_fisher ;;
      deep_stats_count_bootstrap_reference) compare "$stem" stdlib_deep_stats_count_bootstrap ;;
      deep_stats_count_increment_reference) compare "$stem" stdlib_deep_stats_count_glm ;;
      deep_nb2_slots_reference) compare "$stem" stdlib_deep_nb2_slots ;;
      deep_normalization_precision_reference)
        rc=0
        timeout 180 "$out/$stem" > "$out/$stem.fixture.sio" 2> "$out/$stem.emitter.stderr" || rc=$?
        record emit "$stem" "$rc"
        rc=0
        cmp tests/run-pass/stdlib_deep_normalization_precision.sio "$out/$stem.fixture.sio" > "$out/$stem.fixture-byte-cmp.txt" 2>&1 || rc=$?
        record source-custody "$stem" "$rc"
        rc=0
        timeout 180 "$out/$stem" --verify "$out/stdlib_deep_normalization_precision.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
        record concordance "$stem" "$rc" ;;
      deep_ros4_adaptive_reference)
        for mode in pure-time pure-time-corrected corrected augmented; do
          rc=0; expected=0
          [[ "$mode" != pure-time ]] || expected=1
          timeout 180 "$out/$stem" "--$mode" > "$out/$stem.$mode.stdout" 2> "$out/$stem.$mode.stderr" || rc=$?
          verified=0
          [[ "$rc" == "$expected" ]] || verified=1
          record oracle-control "$stem:$mode:expected=$expected:actual=$rc" "$verified"
        done ;;
      deep_checked_log_slots_reference)
        compare "$stem" stdlib_deep_checked_log_slots
        rc=0
        timeout 180 "$out/$stem" --conversion-selftest > "$out/$stem.conversion-selftest.stdout" 2> "$out/$stem.conversion-selftest.stderr" || rc=$?
        record cpp-controls "$stem:conversion-selftest" "$rc" ;;
      deep_poisson_zero_likelihood_reference) compare "$stem" stdlib_deep_poisson_zero_likelihood ;;
      deep_stats_count_profile_bootstrap_reference)
        compare "$stem" stdlib_deep_stats_count_profile_bootstrap
        rc=0
        timeout 180 "$out/$stem" --mutant-fixed-alpha "$out/stdlib_deep_stats_count_profile_bootstrap.stdout" > "$out/$stem.fixed-mutant.stdout" 2> "$out/$stem.fixed-mutant.stderr" || rc=$?
        rejected=1
        [[ "$rc" != 2 ]] || rejected=0
        record mutant-rejection "$stem:fixed-alpha:expected=2:actual=$rc" "$rejected"
        rc=0
        timeout 180 "$out/$stem" --nested-single "$out/stdlib_scale_count_profile_nested_bootstrap.stdout" > "$out/$stem.nested.stdout" 2> "$out/$stem.nested.stderr" || rc=$?
        record concordance "$stem:nested" "$rc"
        rc=0
        timeout 180 "$out/$stem" --nested-mutant-fixed-alpha "$out/stdlib_scale_count_profile_nested_bootstrap.stdout" > "$out/$stem.nested-mutant.stdout" 2> "$out/$stem.nested-mutant.stderr" || rc=$?
        rejected=1; [[ "$rc" != 2 ]] || rejected=0
        record mutant-rejection "$stem:nested-fixed-alpha:expected=2:actual=$rc" "$rejected"
        grep -v '^DEEP COUNT PROFILE NESTED BOOTSTRAP PASS$' "$out/stdlib_scale_count_profile_nested_bootstrap.stdout" > "$out/$stem.nested-prefix.txt"
        rc=0
        timeout 180 "$out/$stem" --nested-single "$out/$stem.nested-prefix.txt" > "$out/$stem.nested-prefix.stdout" 2> "$out/$stem.nested-prefix.stderr" || rc=$?
        rejected=1; [[ "$rc" != 2 ]] || rejected=0
        record mutant-rejection "$stem:nested-prefix:expected=2:actual=$rc" "$rejected" ;;
      deep_stats_dynamic_robust_reference) compare "$stem" stdlib_deep_stats_dynamic_robust ;;
      deep_stats_dynamic_robust_contrast_reference) compare "$stem" stdlib_deep_stats_dynamic_robust_contrast ;;
      deep_viz_scientific_document)
        compare "$stem" stdlib_deep_viz_scientific_document
        compare_mode "$stem" --count stdlib_deep_viz_count_diagnostics ;;
      deep_stats_dispersion_profile_reference) compare "$stem" stdlib_deep_stats_dispersion_profile ;;
      deep_summaries_reference) compare "$stem" stdlib_deep_summaries ;;
      deep_qr_factor_reference) compare "$stem" stdlib_deep_qr_factor ;;
      deep_viz_forest) compare "$stem" stdlib_deep_viz_forest ;;
      deep_viz_axis_layout) compare "$stem" stdlib_deep_viz_axis_layout ;;
      deep_viz_fitted_axis) compare "$stem" stdlib_deep_viz_fitted_axis ;;
      deep_viz_line_reduce) compare "$stem" stdlib_deep_viz_line_reduce_million ;;
      deep_viz_dynamic_plot) compare "$stem" stdlib_deep_viz_dynamic_plot ;;
      deep_viz_dynamic_fields) compare "$stem" stdlib_deep_viz_dynamic_fields ;;
      deep_viz_dynamic_scaled) compare "$stem" stdlib_deep_viz_dynamic_scaled ;;
      deep_viz_contour_predicate) compare "$stem" stdlib_deep_viz_contour_predicate ;;
      deep_viz_dynamic_distribution) compare "$stem" stdlib_deep_viz_dynamic_distribution ;;
      deep_viz_dynamic_contour) compare "$stem" stdlib_deep_viz_dynamic_contour ;;
      deep_count_sampling_reference)
        timeout 180 "$out/$stem" --status-checks > "$out/$stem.status.stdout" 2> "$out/$stem.status.stderr" || rc=$?
        record cpp-controls "$stem" "$rc"
        emit "$stem" "" ""
        emit "$stem" ".moments" "--emit-moments"
        rc=0
        timeout 180 "$out/$stem" --check-moments "$out/$stem.moments.generated.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
        record concordance "$stem" "$rc"
        for pair in "pcg.into emit-pcg-into check-pcg-into" "count.into emit-count-into-million check-count-into" "ptrs.into emit-ptrs-into-million check-ptrs-million"; do
          read -r suffix generator checker <<< "$pair"
          emit "$stem" ".$suffix" "--$generator"
          rc=0
          timeout 180 "$out/$stem" "--$checker" "$out/$stem.$suffix.generated.stdout" > "$out/$stem.$suffix.concordance.stdout" 2> "$out/$stem.$suffix.concordance.stderr" || rc=$?
          record concordance "$stem.$suffix" "$rc"
        done ;;
      *) record unregistered-oracle "$stem" 1 ;;
    esac
done
# Canonical bitmap data and actual native raster orientation, including clipping.
native tests/run-pass/display_font_orientation.sio display_font_orientation
rc=0
timeout 180 g++ -std=c++23 -O2 -Wall -Wextra tests/reference/display_font_orientation_reference.cpp -o "$out/display_font_orientation_reference" > "$out/display_font_orientation_reference.cpp.txt" 2>&1 || rc=$?
record cpp-compile display_font_orientation_reference "$rc"
if [[ $rc == 0 ]]; then
 rc=0
 timeout 180 "$out/display_font_orientation_reference" --canonical tests/reference/data/font8x8_basic.h "$out/display_font_orientation.stdout" > "$out/display_font_orientation_reference.concordance.stdout" 2> "$out/display_font_orientation_reference.concordance.stderr" || rc=$?
 record concordance font-canonical "$rc"
fi

lr_data=tests/reference/data/count_profile_lr_20261024
for stem in count_dispersion_increment_reference count_profile_lr_fixed_reference count_profile_lr_v10_reference count_likelihood_difference_reference; do
 rc=0
 timeout 180 g++ -std=c++23 -O2 -Wall -Wextra "tests/reference/$stem.cpp" -lquadmath -o "$out/$stem" > "$out/$stem.cpp.txt" 2>&1 || rc=$?
 record cpp-compile "$stem" "$rc"
 [[ $rc == 0 ]] || continue
 rc=0
 case "$stem" in
  count_dispersion_increment_reference) timeout 600 "$out/$stem" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$? ;;
  count_profile_lr_fixed_reference) timeout 180 "$out/$stem" "$lr_data/case-3.stdout" "$lr_data/case-11.stdout" "$lr_data/case-13.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$? ;;
  count_profile_lr_v10_reference) timeout 180 "$out/$stem" "$lr_data/v10.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$? ;;
  count_likelihood_difference_reference) timeout 180 "$out/$stem" --native "$out/stdlib_scale_count_likelihood_difference.stdout" "$lr_data/case-3.stdout" "$lr_data/case-11.stdout" "$lr_data/case-13.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$? ;;
 esac
 record concordance "$stem" "$rc"
 if [[ "$stem" == count_dispersion_increment_reference || "$stem" == count_profile_lr_fixed_reference ]]; then
  rc=0
  if [[ "$stem" == count_profile_lr_fixed_reference ]]; then timeout 180 "$out/$stem" --mutant-sign "$lr_data/case-3.stdout" "$lr_data/case-11.stdout" "$lr_data/case-13.stdout" > "$out/$stem.mutant.stdout" 2> "$out/$stem.mutant.stderr" || rc=$?
  else timeout 180 "$out/$stem" --mutant-sign > "$out/$stem.mutant.stdout" 2> "$out/$stem.mutant.stderr" || rc=$?; fi
  rejected=1; [[ "$rc" != 2 ]] || rejected=0
  record mutant-rejection "$stem:orientation:expected=2:actual=$rc" "$rejected"
 fi
done
rc=0
timeout 180 g++ -std=c++23 -O2 -Wall -Wextra tests/reference/normalization_into_reference.cpp -o "$out/normalization_into_reference" > "$out/normalization_into_reference.cpp.txt" 2>&1 || rc=$?
record cpp-compile normalization_into_reference "$rc"
if [[ $rc == 0 ]]; then compare normalization_into_reference stdlib_scale_normalization_into; fi
rc=0
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.verify.txt" 2>&1 || rc=$?
record input-hashes unchanged "$rc"
rc=0
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.verify.txt" 2>&1 || rc=$?
record compiler-hash unchanged "$rc"
printf '%s\n' "$fail" > "$out/STATUS"
exit "$fail"
