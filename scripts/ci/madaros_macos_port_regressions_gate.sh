#!/usr/bin/env bash
# Extra discriminators discovered during the 200-program port. Not a replacement
# or extension of the pinned corpus denominator.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPILER="${1:?pass the native Madaros candidate}"
case "$COMPILER" in /*) ;; *) COMPILER="$PWD/$COMPILER" ;; esac
WORK="${2:-$(mktemp -d "${TMPDIR:-/tmp}/madaros-port-regressions.XXXXXX")}"
mkdir -p "$WORK"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
# Simulate a clean macOS PATH lacking timeout without hiding uname/dirname.
mkdir -p "$WORK/without-timeout"
ln -s /usr/bin/uname "$WORK/without-timeout/uname"
ln -s /usr/bin/dirname "$WORK/without-timeout/dirname"
set +e
PATH="$WORK/without-timeout" /bin/bash scripts/ci/build_madaros_macos_arm64.sh \
    "$WORK/nonexistent-seed" > "$WORK/timeout-preflight.log" 2>&1
preflight_status=$?
set -e
test "$preflight_status" = 2
grep -Fq 'brew install coreutils' "$WORK/timeout-preflight.log"
echo 'PASS negative: missing timeout refused before compiler build'
"${CXX:-/usr/bin/clang++}" -std=c++23 -O2 -ffp-contract=off -Wall -Wextra -Werror \
    scripts/ci/knowledge_measure_variance_oracle.cpp -o "$WORK/variance-oracle"
"$WORK/variance-oracle" > "$WORK/variance-oracle.log"
for test in a64_tuple_call_snapshot a64_box_large_field_table \
    a64_array_element_borrow madaros_nested_float_array_rank \
    madaros_generic_hof_scalar_inference madaros_generic_bool_literal_inference \
    madaros_flat_arg_dep_reset \
    madaros_measure_literal_variance_bound \
    madaros_escaping_float_closure madaros_measure_variance_joins \
    rapamycin_epistemic_pbpk gum_fo_closure_call gum_fo_for_loop_call \
    gum_fo_match_call gum_fo_ref_local_field gum_fo_while_call; do
    source_path="tests/run-pass/$test.sio"
    timeout 90 "$COMPILER" build "$source_path" --target aarch64-macos -o "$WORK/$test" > "$WORK/$test.build.log" 2>&1
    chmod +x "$WORK/$test"
    codesign -s - -f "$WORK/$test"
    codesign --verify "$WORK/$test"
    timeout 30 "$WORK/$test" > "$WORK/$test.stdout.log" 2> "$WORK/$test.stderr.log"
    sed -n 's|^//@ expect-stdout: ||p' "$source_path" > "$WORK/$test.expected.log"
    test -s "$WORK/$test.expected.log"
    while IFS= read -r expected; do
        grep -Fq -- "$expected" "$WORK/$test.stdout.log"
    done < "$WORK/$test.expected.log"
    echo "PASS positive: $test"
    if [[ "$test" = madaros_flat_arg_dep_reset ]]; then
        grep -Fq 'lower_array: arena_reset_ok module ' "$WORK/$test.build.log"
        if grep -Fq 'arena_reset_skipped' "$WORK/$test.build.log"; then
            echo 'FAIL: dependency Box arena was not reclaimed' >&2
            exit 1
        fi
    fi
done
for test in madaros_closure_inferred_numeric_rejects_bool \
    madaros_generic_hof_inference_mismatch \
    madaros_measure_unknown_variance_not_attested \
    madaros_measure_excess_variance_not_attested \
    madaros_measure_sum_variance_not_attested \
    madaros_measure_correlated_variance_not_attested \
    madaros_measure_unknown_sum_not_attested \
    madaros_measure_large_sum_not_attested \
    madaros_measure_add_payload_mismatch \
    ontology_epistemic_closure_return_downgrade_reject; do
    source_path="tests/compile-fail/$test.sio"
    set +e
    timeout 90 "$COMPILER" check "$source_path" > "$WORK/$test.check.log" 2>&1
    check_status=$?
    set -e
    # Require a normal diagnostic refusal, never a crash or timeout.
    test "$check_status" = 1
    grep -Eq 'error\[E[0-9]+\]' "$WORK/$test.check.log"
    sed -n 's|^//@ error-pattern: ||p' "$source_path" > "$WORK/$test.expected.log"
    while IFS= read -r expected; do
        grep -Fq -- "$expected" "$WORK/$test.check.log"
    done < "$WORK/$test.expected.log"
    echo "PASS negative: $test"
done
set +e
timeout 90 "$COMPILER" build tests/compile-fail/madaros_closure_capture_capacity_refuses.sio \
    --target aarch64-macos -o "$WORK/capture-overflow" > "$WORK/capture-overflow.build.log" 2>&1
overflow_status=$?
set -e
test "$overflow_status" = 1
test ! -s "$WORK/capture-overflow"
grep -Fq 'closure capture capacity exceeded' "$WORK/capture-overflow.build.log"
echo 'PASS negative: closure capture capacity refuses without output'
echo "PASS: native port extra positive/negative discriminators; artifacts=$WORK"
