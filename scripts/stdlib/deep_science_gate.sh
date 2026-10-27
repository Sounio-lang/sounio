#!/usr/bin/env bash
# Native Linux worker only. MADAROS_RAW_BIN must identify a source-built compiler.
set -uo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${STDLIB_DEEP_OUT:?set STDLIB_DEEP_OUT to a dedicated artifact directory}
[[ "$out" == /* && -x "${MADAROS_RAW_BIN:?set MADAROS_RAW_BIN}" ]] || exit 64
[[ -z "${SOUNIO_SOUC_BIN:-}" ]] || { echo "explicit raw override bypasses engine routing" >&2; exit 64; }
MADAROS_RAW_BIN=$(readlink -f "$MADAROS_RAW_BIN") || exit 64
export MADAROS_RAW_BIN
SEED_RAW_BIN=$(readlink -f "${SEED_RAW_BIN:-$repo/bin/souc-lean-single-x86_64}") || exit 64
[[ -x "$SEED_RAW_BIN" ]] || exit 64
export SEED_RAW_BIN
time_bin=$(readlink -f "${STDLIB_TIME_BIN:-/usr/bin/time}") || { echo "missing executable GNU time; set STDLIB_TIME_BIN" >&2; exit 64; }
[[ -x "$time_bin" ]] || { echo "missing executable GNU time; set STDLIB_TIME_BIN" >&2; exit 64; }
out=$(realpath -m "$out") || exit 64
case "$out/" in "$repo/stdlib/"*|"$repo/tests/"*|"$repo/examples/"*|"$repo/scripts/stdlib/"*|"$repo/scripts/lib/"*) echo "output overlaps input catalog" >&2; exit 64;; esac
mkdir -p "$out" || exit 65
[[ -z "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "refusing nonempty output directory" >&2; exit 65; }
"$time_bin" --version > "$out/TIME_VERSION.txt" 2>&1 || exit 64
grep -q GNU "$out/TIME_VERSION.txt" || { echo "GNU time required" >&2;exit 64; }
sha256sum "$time_bin" > "$out/TIME.sha256" || exit 65
cd "$repo" || exit 65
export SOUNIO_STDLIB_PATH="$repo/stdlib" SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288
printf 'kind\tsource\texit_status\n' > "$out/RESULTS.tsv"
mkdir -p "$out/result-rows" || exit 65
sha256sum "$MADAROS_RAW_BIN" "$SEED_RAW_BIN" bin/souc-lean-single-x86_64 > "$out/COMPILER.sha256" || exit 65
# Include callers, routing, the runner and every focused gate. Rebuild the
# catalog after execution so added or removed files cannot escape sha256sum -c.
write_inputs() {
    find stdlib tests examples scripts/stdlib scripts/lib tools/science_boundary runtime/stdlib_platform.cpp docs/evidence/stdlib-massive-2026-10-26 \( -type f -o -type l \) -print0 | LC_ALL=C sort -z | xargs -0 sha256sum || return $?
    sha256sum bin/souc bin/madaros bin/souc-lean-single-x86_64 scripts/ci/souc-native-wrapper.sh \
      docs/evidence/stdlib-SIO-172/before_variance/variance_witness.sio \
      docs/evidence/stdlib-SIO-172/before_variance/phonon_original.sio \
      docs/evidence/stdlib-SIO-174/before/callback_time.sio
}
write_inputs > "$out/INPUTS.sha256" || exit 65
g++ --version > "$out/CXX_TOOLCHAIN.txt" || exit 65
# Independent references require Boost.Multiprecision headers. Refuse before
# compiling the reference corpus when a sparse worker lacks them.
printf '%s\n' '#include <boost/multiprecision/cpp_int.hpp>' '#include <boost/multiprecision/cpp_dec_float.hpp>' |
    g++ -std=c++23 -x c++ -fsyntax-only - > "$out/CXX_DEPENDENCIES.stdout" 2> "$out/CXX_DEPENDENCIES.stderr" ||
    { echo "missing C++23 reference dependencies; inspect CXX_DEPENDENCIES.stderr" >&2; exit 64; }
# Bind the actual compiler helpers and their dynamic runtime dependencies.
cxx_manifest() {
    local tool resolved
    for tool in "$(command -v g++)" "$(g++ -print-prog-name=cc1plus)" \
      "$(g++ -print-prog-name=collect2)" "$(g++ -print-prog-name=as)" \
      "$(g++ -print-prog-name=ld)" "$(g++ -print-file-name=libstdc++.so)" \
      "$(g++ -print-file-name=libgcc_s.so.1)" "$(g++ -print-file-name=libquadmath.so.0)"; do
        if [[ "$tool" != */* ]]; then tool=$(command -v "$tool") || return 1; fi
        resolved=$(readlink -f "$tool") || return 1
        [[ -f "$resolved" ]] || return 1
        printf '%s\n' "$resolved"
        ldd "$resolved" > "$out/.cxx-ldd" 2>&1 || return 1
        if grep -q 'not found' "$out/.cxx-ldd"; then return 1; fi
        awk '/=> \// {print $3} /^[[:space:]]*\// {print $1}' "$out/.cxx-ldd"
    done | LC_ALL=C sort -u | xargs -r sha256sum
}
cxx_manifest > "$out/CXX_RUNTIME.sha256" || exit 65
[[ -s "$out/CXX_RUNTIME.sha256" ]] || exit 65
fail=0
record() {
    local tmp
    tmp="$(mktemp "$out/result-rows/row.XXXXXX")" || { fail=1; return 1; }
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" > "$tmp" || { fail=1; return 1; }
    mv -- "$tmp" "$tmp.tsv" || { fail=1; return 1; }
    [[ "$3" == 0 ]] || fail=1
    return 0
}
resolve_jobs() {
    local cpus memory_bytes memory_current host_memory_bytes memory_jobs jobs
    if [[ -n "${STDLIB_DEEP_JOBS:-}" ]]; then
        [[ "$STDLIB_DEEP_JOBS" =~ ^[1-9][0-9]*$ ]] || return 2
        printf '%s\n' "$STDLIB_DEEP_JOBS"
        return 0
    fi
    cpus="$(nproc 2>/dev/null || echo 1)"
    if [[ -r /sys/fs/cgroup/memory.max ]]; then
        read -r memory_bytes < /sys/fs/cgroup/memory.max
        if [[ "$memory_bytes" == max ]]; then
            memory_bytes="$(awk '/^MemAvailable:/ { printf "%.0f\n", $2 * 1024; exit }' /proc/meminfo)"
        elif [[ "$memory_bytes" =~ ^[0-9]+$ && -r /sys/fs/cgroup/memory.current ]]; then
            read -r memory_current < /sys/fs/cgroup/memory.current
            if [[ "$memory_current" =~ ^[0-9]+$ && "$memory_current" -lt "$memory_bytes" ]]; then
                memory_bytes=$((memory_bytes - memory_current))
            else
                memory_bytes=1
            fi
        else
            [[ "$memory_bytes" =~ ^[0-9]+$ ]] || memory_bytes=""
        fi
    elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
        read -r memory_bytes < /sys/fs/cgroup/memory/memory.limit_in_bytes
        memory_current=0
        [[ ! -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]] \
            || read -r memory_current < /sys/fs/cgroup/memory/memory.usage_in_bytes
        if [[ "$memory_bytes" =~ ^[0-9]+$ && "$memory_current" =~ ^[0-9]+$ && "$memory_current" -lt "$memory_bytes" ]]; then
            memory_bytes=$((memory_bytes - memory_current))
            host_memory_bytes="$(awk '/^MemAvailable:/ { printf "%.0f\n", $2 * 1024; exit }' /proc/meminfo)"
            if [[ "$host_memory_bytes" =~ ^[1-9][0-9]*$ ]]; then
                if (( memory_bytes > host_memory_bytes )); then
                    memory_bytes="$host_memory_bytes"
                fi
            else
                memory_bytes=1
            fi
        else
            memory_bytes=1
        fi
    else
        memory_bytes="$(awk '/^MemAvailable:/ { printf "%.0f\n", $2 * 1024; exit }' /proc/meminfo)"
    fi
    [[ "$cpus" =~ ^[1-9][0-9]*$ ]] || cpus=1
    [[ "$memory_bytes" =~ ^[1-9][0-9]*$ ]] || memory_bytes=$((4 * 1024 * 1024 * 1024))
    memory_jobs=$((memory_bytes / (4 * 1024 * 1024 * 1024)))
    (( memory_jobs > 0 )) || memory_jobs=1
    jobs=2
    (( jobs > cpus )) && jobs="$cpus"
    (( jobs > memory_jobs )) && jobs="$memory_jobs"
    printf '%s\n' "$jobs"
}
jobs="$(resolve_jobs)" || { echo "invalid STDLIB_DEEP_JOBS (expected a positive integer)" >&2; exit 64; }
positive_diagnostics() { grep -Eq '^[[:space:]]*error:|^[[:space:]]*error\[E' "$1"; }
native() {
    local src=$1 name=$2 rc=0
    timeout 180 bin/souc check "$src" > "$out/$name.check.txt" 2>&1 || rc=$?
    if [[ $rc == 0 ]] && positive_diagnostics "$out/$name.check.txt"; then rc=1; fi
    record check "$src" "$rc"
    [[ $rc == 0 ]] || return
    timeout 240 bin/souc compile "$src" -o "$out/$name.elf" > "$out/$name.compile.txt" 2>&1 || rc=$?
    if [[ $rc == 0 ]] && positive_diagnostics "$out/$name.compile.txt"; then rc=1; fi
    record compile "$src" "$rc"
    [[ $rc == 0 ]] || return
    sha256sum "$src" "$out/$name.elf" > "$out/$name.identities" || { record native-artifact-seal "$src" 1; return; }
    # Each parallel worker owns its pre-execution manifest; aggregate after join.
    sha256sum "$out/$name.elf" > "$out/$name.native-elf.sha256" || { record native-artifact-seal "$src" 1; return; }
    "$time_bin" -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 300 "$out/$name.elf" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?
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
NATIVE_SOURCES=()
NATIVE_NAMES=()
queue_native() {
    NATIVE_SOURCES+=("$1")
    NATIVE_NAMES+=("$2")
}
emit_fixture() {
    local stem="$1" suffix="$2" mode="$3" rc=0
    if [[ -n "$mode" ]]; then
        timeout 180 "$out/$stem" "$mode" > "$out/$stem$suffix.sio" 2> "$out/$stem$suffix.emitter.stderr" || rc=$?
    else
        timeout 180 "$out/$stem" > "$out/$stem$suffix.sio" 2> "$out/$stem$suffix.emitter.stderr" || rc=$?
    fi
    record emit "$stem$suffix" "$rc"
    [[ "$rc" != 0 ]] || queue_native "$out/$stem$suffix.sio" "$stem$suffix.generated"
}
native_worker() {
    local src="$1" name="$2" idx="$3" rc=0 tmp
    fail=0
    native "$src" "$name" || rc=$?
    [[ "$fail" == 0 ]] || rc=1
    tmp="$(mktemp "$out/.native-status.XXXXXX")" || exit 65
    printf '%s\n' "$rc" > "$tmp" || exit 65
    mv -- "$tmp" "$out/native.$idx.status" || exit 65
    return "$rc"
}
run_native_batch() {
    local active=0 idx worker_rc
    printf '[deep-science] native workers=%s fixtures=%s\n' "$jobs" "${#NATIVE_SOURCES[@]}"
    for ((idx=0; idx<${#NATIVE_SOURCES[@]}; idx++)); do
        native_worker "${NATIVE_SOURCES[$idx]}" "${NATIVE_NAMES[$idx]}" "$idx" &
        active=$((active + 1))
        if (( active >= jobs )); then
            wait -n || true
            active=$((active - 1))
        fi
    done
    while (( active > 0 )); do
        wait -n || true
        active=$((active - 1))
    done
    for ((idx=0; idx<${#NATIVE_SOURCES[@]}; idx++)); do
        if [[ -f "$out/native.$idx.status" ]]; then
            IFS= read -r worker_rc < "$out/native.$idx.status" || worker_rc=""
            if [[ "$worker_rc" =~ ^[0-9]+$ ]]; then
                record native-worker "${NATIVE_NAMES[$idx]}" "$worker_rc"
            else
                record native-worker "${NATIVE_NAMES[$idx]}:invalid-status" 1
            fi
        else
            record native-worker "${NATIVE_NAMES[$idx]}:missing-status" 1
        fi
    done
}
for src in tests/run-pass/stdlib_deep_*.sio; do queue_native "$src" "$(basename "$src" .sio)"; done
for name in stdlib_scale_normalization_into stdlib_scale_normalization_status stdlib_scale_count_likelihood_difference stdlib_scale_count_profile_nested_bootstrap; do queue_native "tests/run-pass/$name.sio" "$name"; done
# Legacy render paths are replayed in the canonical engine as well as the
# Stage2 full suite: these names exposed nested-label and first-tick defects.
for name in stats_inference_forest viz_bootstrap_forest viz_count_matrix_qc \
    viz_entropy_covariance viz_information_channels viz_scatter_viewport \
    viz_scientific_clip viz_statistical_interval_forest viz_bca_mean \
    viz_permutation_mc viz_axis_first_tick_guard; do
    queue_native "tests/run-pass/$name.sio" "$name"
done
for name in viz_dynamic_count_workflow viz_dynamic_ode_diagnostics; do
    queue_native "examples/$name.sio" "$name"
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
    local stem=$1 suffix=$2
    [[ -f "$out/$stem$suffix.generated.stdout" ]] \
        || record emit "$stem$suffix:missing-native-fixture" 1
}

compile_cpp() {
    local src="$1" stem="$2" extra="$3" rc=0
    timeout 180 g++ -std=c++23 -O2 -Wall -Wextra "$src" ${extra:+$extra} \
        -o "$out/$stem" > "$out/$stem.cpp.txt" 2>&1 || rc=$?
    record cpp-compile "$src" "$rc"
    if [[ $rc == 0 ]]; then
        sha256sum "$out/$stem" >> "$out/REFERENCE_ELFS.sha256" \
            || record reference-artifact-seal "$src" 1
    fi
}

# Build every independent oracle before starting native fixture work or comparisons.
for src in tests/reference/deep_*.cpp; do
    compile_cpp "$src" "$(basename "$src" .cpp)" "-lquadmath"
done
for stem in count_dispersion_increment_reference count_profile_lr_fixed_reference \
    count_profile_lr_v10_reference count_likelihood_difference_reference; do
    compile_cpp "tests/reference/$stem.cpp" "$stem" "-lquadmath"
done
compile_cpp tests/reference/display_font_orientation_reference.cpp display_font_orientation_reference ""
compile_cpp tests/reference/normalization_into_reference.cpp normalization_into_reference ""
for src in tests/compile-fail/stdlib_deep_*.sio; do
    name=$(basename "$src" .sio); rc=0
    timeout 120 bin/souc check "$src" > "$out/$name.check.txt" 2>&1 || rc=$?
    pattern=linear
    [[ "$name" != *forge* ]] || pattern=private
    [[ "$name" != *callback_private* ]] || pattern=E175
    if [[ $rc != 1 ]] || ! grep -qi "$pattern" "$out/$name.check.txt"; then record negative "$src" 1
    else record negative "$src" 0; fi
done
queue_native tests/run-pass/display_font_orientation.sio display_font_orientation
for stem in deep_ode_reference deep_special_reference deep_ode_dynamic_reference \
    deep_factor_reference deep_svd_reference deep_ros4_reference deep_kde_reference \
    deep_dynamic_count_tsv_reference deep_numeric_select_reference; do
    emit_fixture "$stem" "" ""
done
emit_fixture deep_count_sampling_reference "" ""
emit_fixture deep_count_sampling_reference .moments --emit-moments
for pair in "pcg.into emit-pcg-into" "count.into emit-count-into-million" "ptrs.into emit-ptrs-into-million"; do
    read -r suffix mode <<< "$pair"
    emit_fixture deep_count_sampling_reference ".$suffix" "--$mode"
done
run_native_batch

oracle_deep_case() {
    local stem="$1" rc=0 expected=0 verified=0 rejected=0 mode suffix generator checker pair
    fail=0
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
      deep_count_dataset_reference)
        compare "$stem" stdlib_deep_count_dataset_reference
        compare_mode "$stem" --workflow viz_dynamic_count_workflow ;;
      deep_viz_scientific_document)
        compare "$stem" stdlib_deep_viz_scientific_document
        compare_mode "$stem" --count stdlib_deep_viz_count_diagnostics ;;
      deep_ode_diagnostic_observer_reference)
        compare "$stem" stdlib_deep_ode_diagnostic_observer
        compare_mode "$stem" --figure viz_dynamic_ode_diagnostics ;;
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
    return "$fail"
}
oracle_tail_case() {
    local stem="$1" rc=0 lr_data=tests/reference/data/count_profile_lr_20261024 rejected=0
    fail=0
    case "$stem" in
      count_dispersion_increment_reference)
        timeout 600 "$out/$stem" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
        record concordance "$stem" "$rc"
        rc=0
        timeout 180 "$out/$stem" --mutant-sign > "$out/$stem.mutant.stdout" 2> "$out/$stem.mutant.stderr" || rc=$?
        rejected=1; [[ "$rc" != 2 ]] || rejected=0
        record mutant-rejection "$stem:orientation:expected=2:actual=$rc" "$rejected"
        ;;
      count_profile_lr_fixed_reference)
        timeout 180 "$out/$stem" "$lr_data/case-3.stdout" "$lr_data/case-11.stdout" "$lr_data/case-13.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
        record concordance "$stem" "$rc"
        rc=0
        timeout 180 "$out/$stem" --mutant-sign "$lr_data/case-3.stdout" "$lr_data/case-11.stdout" "$lr_data/case-13.stdout" > "$out/$stem.mutant.stdout" 2> "$out/$stem.mutant.stderr" || rc=$?
        rejected=1; [[ "$rc" != 2 ]] || rejected=0
        record mutant-rejection "$stem:orientation:expected=2:actual=$rc" "$rejected"
        ;;
      count_profile_lr_v10_reference)
        timeout 180 "$out/$stem" "$lr_data/v10.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
        record concordance "$stem" "$rc"
        ;;
      count_likelihood_difference_reference)
        timeout 180 "$out/$stem" --native "$out/stdlib_scale_count_likelihood_difference.stdout" "$lr_data/case-3.stdout" "$lr_data/case-11.stdout" "$lr_data/case-13.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
        record concordance "$stem" "$rc"
        ;;
      display_font_orientation_reference)
        timeout 180 "$out/$stem" --canonical tests/reference/data/font8x8_basic.h "$out/display_font_orientation.stdout" > "$out/$stem.concordance.stdout" 2> "$out/$stem.concordance.stderr" || rc=$?
        record concordance font-canonical "$rc"
        ;;
      normalization_into_reference)
        compare "$stem" stdlib_scale_normalization_into
        ;;
    esac
    return "$fail"
}

ORACLE_KINDS=()
ORACLE_STEMS=()
queue_oracle() {
    ORACLE_KINDS+=("$1")
    ORACLE_STEMS+=("$2")
}
for src in tests/reference/deep_*.cpp; do
    queue_oracle deep "$(basename "$src" .cpp)"
done
for stem in count_dispersion_increment_reference count_profile_lr_fixed_reference \
    count_profile_lr_v10_reference count_likelihood_difference_reference \
    display_font_orientation_reference normalization_into_reference; do
    queue_oracle tail "$stem"
done
oracle_worker() {
    local kind="$1" stem="$2" idx="$3" rc=0 tmp
    fail=0
    if [[ "$kind" == deep ]]; then
        oracle_deep_case "$stem" || rc=$?
    else
        oracle_tail_case "$stem" || rc=$?
    fi
    [[ "$fail" == 0 ]] || rc=1
    tmp="$(mktemp "$out/.oracle-status.XXXXXX")" || exit 65
    printf '%s\n' "$rc" > "$tmp" || exit 65
    mv -- "$tmp" "$out/oracle.$idx.status" || exit 65
    return "$rc"
}
run_oracle_batch() {
    local active=0 idx worker_rc
    printf '[deep-science] oracle workers=%s cases=%s\n' "$jobs" "${#ORACLE_STEMS[@]}"
    for ((idx=0; idx<${#ORACLE_STEMS[@]}; idx++)); do
        oracle_worker "${ORACLE_KINDS[$idx]}" "${ORACLE_STEMS[$idx]}" "$idx" &
        active=$((active + 1))
        if (( active >= jobs )); then
            wait -n || true
            active=$((active - 1))
        fi
    done
    while (( active > 0 )); do
        wait -n || true
        active=$((active - 1))
    done
    for ((idx=0; idx<${#ORACLE_STEMS[@]}; idx++)); do
        if [[ -f "$out/oracle.$idx.status" ]]; then
            IFS= read -r worker_rc < "$out/oracle.$idx.status" || worker_rc=""
            if [[ "$worker_rc" =~ ^[0-9]+$ ]]; then
                record oracle-worker "${ORACLE_KINDS[$idx]}:${ORACLE_STEMS[$idx]}" "$worker_rc"
            else
                record oracle-worker "${ORACLE_STEMS[$idx]}:invalid-status" 1
            fi
        else
            record oracle-worker "${ORACLE_STEMS[$idx]}:missing-status" 1
        fi
    done
}
run_oracle_batch

# Checked BDF1 numerical concordance and refusal controls use actual native rows.
rc=0
timeout 300 bash scripts/stdlib/ode_bdf_checked_gate.sh "$out/checked-bdf" > "$out/checked-bdf.gate.stdout" 2> "$out/checked-bdf.gate.stderr" || rc=$?
record focused-gate checked-bdf "$rc"
# Genuine NCT tails, power and minimum-n: own generated native/strict-seed
# fixtures plus independent C++23 actual-output oracles and rejected mutants.
rc=0
timeout 900 bash scripts/stdlib/noncentral_t_power_gate.sh "$out/noncentral_t_power" > "$out/noncentral_t_power.focused.stdout" 2> "$out/noncentral_t_power.focused.stderr" || rc=$?
record focused-gate noncentral_t_power "$rc"
# Dynamic named counts, byte-exact identities/joins and validated CSR.
# Child receipt separates bootstrap numerical support from canonical privacy.
rc=0
MADAROS_RAW_BIN="$MADAROS_RAW_BIN" SEED_RAW_BIN="${SEED_RAW_BIN:-$repo/bin/souc-lean-single-x86_64}" timeout 1200 bash scripts/stdlib/genomics_count_identity_gate.sh "$out/genomics-count-identities" > "$out/genomics-count-identities.gate.stdout" 2> "$out/genomics-count-identities.gate.stderr" || rc=$?
record focused-gate genomics-count-identities "$rc"

# Reusable named results: canonical framing across all chunk sizes and long streams.
rc=0
MADAROS_RAW_BIN="$MADAROS_RAW_BIN" SEED_RAW_BIN="${SEED_RAW_BIN:-$repo/bin/souc-lean-single-x86_64}" timeout 600 bash scripts/stdlib/genomics_named_stream_gate.sh "$out/genomics-named-stream" > "$out/genomics-named-stream.gate.stdout" 2> "$out/genomics-named-stream.gate.stderr" || rc=$?
record focused-gate genomics-named-stream "$rc"

# Conditional exact signed ranks, explicit approximation, counted scratch
# and independent C++ integer/tail controls execute in both compiler engines.
rc=0
timeout 1200 bash scripts/stdlib/wilcoxon_checked_gate.sh "$out/wilcoxon_checked" > "$out/wilcoxon_checked.focused.stdout" 2> "$out/wilcoxon_checked.focused.stderr" || rc=$?
record focused-gate wilcoxon_checked "$rc"

# Full pooled/group right-censored risk curves, checked logrank tails,
# and independently qualified median boundaries execute in both engines.
rc=0
timeout 1200 bash scripts/stdlib/survival_checked_gate.sh "$out/survival_checked" > "$out/survival_checked.focused.stdout" 2> "$out/survival_checked.focused.stderr" || rc=$?
record focused-gate survival_checked "$rc"

# Checked FEM / Poisson manufactured solutions, legacy imports and work controls.
rc=0
MADAROS_RAW_BIN="$MADAROS_RAW_BIN" bash scripts/stdlib/fem_poisson_gate.sh "$out/fem-poisson" > "$out/fem-poisson.gate.stdout" 2> "$out/fem-poisson.gate.stderr" || rc=$?
record focused-gate fem-poisson "$rc"

# DP5(4) FSAL, quartic dense output and event contracts: actual native + C++.
# Optional DP54_SEED_BIN selects a second independently receipted engine.
rc=0
DP54_GATE_OUTPUT="$out/dp54-focused" DP54_SEED_BIN="$SEED_RAW_BIN" MADAROS_RAW_BIN="$MADAROS_RAW_BIN" \
 timeout 900 bash scripts/stdlib/dp54_dense_events_gate.sh > "$out/dp54-focused.stdout" 2> "$out/dp54-focused.stderr" || rc=$?
record focused-gate dp54-dense-events "$rc"

# Prepared viewport-bound contour geometry and independent native pixel/oracle controls.
rc=0
timeout 900 env MADAROS_RAW_BIN="$MADAROS_RAW_BIN" bash scripts/stdlib/prepared_contour_gate.sh "$out/prepared-contour" > "$out/prepared-contour.gate.stdout" 2> "$out/prepared-contour.gate.stderr" || rc=$?
record focused-gate prepared-contour "$rc"

# Canonical palette primary data, actual Canvas/SVG colors and strict controls.
rc=0
MADAROS_RAW_BIN="$MADAROS_RAW_BIN" bash scripts/stdlib/canonical_palette_gate.sh "$out/canonical-palettes" > "$out/canonical-palettes.gate.stdout" 2> "$out/canonical-palettes.gate.stderr" || rc=$?
record focused-gate canonical-palettes "$rc"

# Checked transforms, finite differences, physical uncertainty and distributions.
rc=0
timeout 900 bash scripts/stdlib/math_fft_checked_gate.sh "$out/math-fft" > "$out/math-fft.gate.stdout" 2> "$out/math-fft.gate.stderr" || rc=$?
record focused-gate math-fft "$rc"
rc=0
NUMDIFF_SEED_BIN="$SEED_RAW_BIN" timeout 900 bash scripts/stdlib/math_numdiff_checked_gate.sh "$out/math-numdiff" > "$out/math-numdiff.gate.stdout" 2> "$out/math-numdiff.gate.stderr" || rc=$?
record focused-gate math-numdiff "$rc"
rc=0
timeout 600 bash scripts/stdlib/physics_uncertainty_gate.sh "$out/physics-uncertainty" > "$out/physics-uncertainty.gate.stdout" 2> "$out/physics-uncertainty.gate.stderr" || rc=$?
record focused-gate physics-uncertainty "$rc"
rc=0
timeout 900 bash scripts/stdlib/checked_distributions_gate.sh "$out/checked-distributions" > "$out/checked-distributions.gate.stdout" 2> "$out/checked-distributions.gate.stderr" || rc=$?
record focused-gate checked-distributions "$rc"

# Genomics batch fit/prior/refit/tests/FDR/report: both native engines + C++23.
rc=0
timeout 1200 bash scripts/stdlib/genomics_batch_gate.sh "$out/genomics_batch" > "$out/genomics_batch.focused.stdout" 2> "$out/genomics_batch.focused.stderr" || rc=$?
record focused-gate genomics_batch "$rc"

# Four real scientific producers, shared renderers and actual exported pixels.
rc=0
SCIENCE_NATIVE_OUTPUT="$out/scientific-native-workflows" SCIENCE_NATIVE_SEED_BIN="$SEED_RAW_BIN" MADAROS_RAW_BIN="$MADAROS_RAW_BIN" timeout 1200 bash scripts/stdlib/scientific_native_workflows_gate.sh > "$out/scientific-native-workflows.gate.stdout" 2> "$out/scientific-native-workflows.gate.stderr" || rc=$?
record focused-gate scientific-native-workflows "$rc"

# ADR-019 structural shape contract; directed arithmetic is not qualified here.
rc=0
INTERVAL_CONTRACT_OUT="$out/interval-contract" timeout 240 bash scripts/stdlib/interval_contract_gate.sh > "$out/interval-contract.gate.stdout" 2> "$out/interval-contract.gate.stderr" || rc=$?
record focused-gate interval-contract "$rc"

# Phonon means/covariance, transport and original production-import controls.
rc=0
PHONON_GATE_OUTPUT="$out/phonon-factor" timeout 600 bash scripts/stdlib/phonon_factor_gate.sh > "$out/phonon-factor.gate.stdout" 2> "$out/phonon-factor.gate.stderr" || rc=$?
record focused-gate phonon-factor "$rc"

rc=0
STDLIB_MASSIVE_OUT="$out/massive-sprint" timeout 900 bash scripts/stdlib/massive_sprint_gate.sh > "$out/massive-sprint.gate.stdout" 2> "$out/massive-sprint.gate.stderr" || rc=$?
record focused-gate massive-sprint "$rc"

# Every child seals native programs and C++ oracles before first use. Repeat
# those checks after all children, so later phases cannot mutate an earlier ELF.
: > "$out/CHILD_ELFS.verify.txt"
for child in checked-bdf noncentral_t_power genomics-count-identities genomics-named-stream \
 wilcoxon_checked survival_checked fem-poisson dp54-focused prepared-contour canonical-palettes \
 math-fft math-numdiff physics-uncertainty checked-distributions genomics_batch \
 scientific-native-workflows interval-contract phonon-factor; do
 rc=0
 if [[ -s "$out/$child/ELFS.sha256" ]]; then
  sha256sum -c "$out/$child/ELFS.sha256" >> "$out/CHILD_ELFS.verify.txt" 2>&1 || rc=$?
 else
  printf 'missing child executable manifest: %s\n' "$child" >> "$out/CHILD_ELFS.verify.txt"
  rc=1
 fi
 record child-artifact-hashes "$child" "$rc"
done

rc=0
cat "$out"/*.native-elf.sha256 > "$out/NATIVE_ELFS.sha256" 2> "$out/NATIVE_ELFS.catalog.stderr" || rc=$?
record native-artifact-catalog joined "$rc"
rc=0
write_inputs > "$out/INPUTS.final.sha256" || rc=$?
if [[ $rc == 0 ]]; then cmp "$out/INPUTS.sha256" "$out/INPUTS.final.sha256" > "$out/INPUTS.catalog-cmp.txt" 2>&1 || rc=$?; fi
record input-catalog unchanged "$rc"
rc=0
sha256sum -c "$out/NATIVE_ELFS.sha256" > "$out/NATIVE_ELFS.verify.txt" 2>&1 || rc=$?
record native-artifact-hashes unchanged "$rc"
rc=0
sha256sum -c "$out/TIME.sha256" > "$out/TIME.final.verify" || rc=$?
record time-tool-hash unchanged "$rc"
rc=0
sha256sum -c "$out/REFERENCE_ELFS.sha256" > "$out/REFERENCE_ELFS.verify.txt" 2>&1 || rc=$?
record reference-artifact-hashes unchanged "$rc"
rc=0
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.verify.txt" 2>&1 || rc=$?
record input-hashes unchanged "$rc"
rc=0
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.verify.txt" 2>&1 || rc=$?
record compiler-hash unchanged "$rc"

rc=0
sha256sum -c "$out/CXX_RUNTIME.sha256" > "$out/CXX_RUNTIME.verify.txt" 2>&1 || rc=$?
record cxx-runtime-hashes unchanged "$rc"

results_tmp="$out/RESULTS.tsv.tmp"
{
    printf 'kind\tsource\texit_status\n'
    for row in "$out"/result-rows/*.tsv; do
        [[ -f "$row" ]] && cat "$row"
    done
} > "$out/RESULTS.unsorted.tsv"
{
    head -n 1 "$out/RESULTS.unsorted.tsv"
    tail -n +2 "$out/RESULTS.unsorted.tsv" | LC_ALL=C sort -t $'\t' -k1,1 -k2,2 -k3,3
} > "$results_tmp"
mv -- "$results_tmp" "$out/RESULTS.tsv"

audit_fail=0
if [[ "$(head -n 1 "$out/RESULTS.tsv")" != $'kind\tsource\texit_status' ]]; then
    audit_fail=1
fi
while IFS=$'\t' read -r kind source exit_status; do
    [[ -n "$kind" && -n "$source" && "$exit_status" =~ ^[0-9]+$ ]] || {
        audit_fail=1
        continue
    }
    [[ "$exit_status" == 0 ]] || audit_fail=1
done < <(tail -n +2 "$out/RESULTS.tsv")
[[ -s "$out/RESULTS.tsv" ]] && [[ "$(tail -n +2 "$out/RESULTS.tsv" | wc -l)" -gt 0 ]] || audit_fail=1
(( audit_fail == 0 )) || fail=1
status_tmp="$(mktemp "$out/.status.XXXXXX")" || exit 65
printf '%s\n' "$fail" > "$status_tmp" || exit 65
mv -- "$status_tmp" "$out/STATUS" || exit 65
exit "$fail"
