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
compare() {
    local stem=$1 fixture=$2 rc=0
    if [[ ! -f "$out/$fixture.stdout" ]]; then record concordance "$stem:missing-$fixture" 1; return; fi
    timeout 180 "$out/$stem" "$out/$fixture.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
    record concordance "$stem" "$rc"
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
      deep_ode_reference|deep_special_reference|deep_ode_dynamic_reference|deep_factor_reference|deep_svd_reference|deep_ros4_reference|deep_kde_reference)
        emit "$stem" "" "" ;;
      deep_dynamic_qr|deep_normalization|deep_stats_reference|deep_ros4_adaptive_reference)
        timeout 180 "$out/$stem" > "$out/$stem.stdout" 2> "$out/$stem.stderr" || rc=$?
        record cpp-native "$stem" "$rc" ;;
      deep_stats_dispersion_reference) compare "$stem" stdlib_deep_genomics_shrinkage ;;
      deep_stats_multiple_testing_reference) compare "$stem" stdlib_deep_stats_multiple_testing ;;
      deep_stats_count_glm_reference) compare "$stem" stdlib_deep_stats_count_glm ;;
      deep_stats_nb2_likelihood_reference) compare "$stem" stdlib_deep_stats_nb2_likelihood ;;
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
        record concordance "$stem" "$rc" ;;
      *) record unregistered-oracle "$stem" 1 ;;
    esac
done
rc=0
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.verify.txt" 2>&1 || rc=$?
record input-hashes unchanged "$rc"
rc=0
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.verify.txt" 2>&1 || rc=$?
record compiler-hash unchanged "$rc"
printf '%s\n' "$fail" > "$out/STATUS"
exit "$fail"
