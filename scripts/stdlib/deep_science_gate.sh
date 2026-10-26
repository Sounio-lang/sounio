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
mkdir -p "$out/result-rows" || exit 65
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256" || exit 65
find stdlib tests -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256" || exit 65
g++ --version > "$out/CXX_TOOLCHAIN.txt" || exit 65
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

# Arrow IPC is Madaros-only (lean_single has no bits_to_f64). The full suite
# skips these fixtures via //@ requires: madaros. This job already holds the
# fresh Madaros ELF, so the gate runs here without a workflow-file edit.
arrow_rc=0
bash "$repo/scripts/arrow_ipc_gate.sh" > "$out/arrow-ipc-gate.log" 2>&1 || arrow_rc=$?
record arrow-gate scripts/arrow_ipc_gate.sh "$arrow_rc"

rc=0
sha256sum -c "$out/INPUTS.sha256" > "$out/INPUTS.verify.txt" 2>&1 || rc=$?
record input-hashes unchanged "$rc"
rc=0
sha256sum -c "$out/COMPILER.sha256" > "$out/COMPILER.verify.txt" 2>&1 || rc=$?
record compiler-hash unchanged "$rc"

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
