#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() {
    printf 'MADAROS_AFFINE_SEMANTICS_FAIL reason=%s\n' "$1" >&2
    exit 1
}

portable_sha256() {
    local output digest
    if command -v sha256sum >/dev/null 2>&1; then
        output="$(LC_ALL=C sha256sum "$1" 2>/dev/null)" || return 1
    elif command -v shasum >/dev/null 2>&1; then
        output="$(LC_ALL=C shasum -a 256 "$1" 2>/dev/null)" || return 1
    else
        return 1
    fi
    digest="${output%%[[:space:]]*}"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || return 1
    printf '%s\n' "$digest"
}

is_elf_binary() {
    [[ "$(od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n')" == "7f454c46" ]]
}

is_elf64_x86_64_le() {
    local class_and_data machine
    class_and_data="$(od -An -tx1 -j4 -N2 "$1" 2>/dev/null | tr -d ' \n')"
    machine="$(od -An -tx1 -j18 -N2 "$1" 2>/dev/null | tr -d ' \n')"
    [[ "$class_and_data" == "0201" && "$machine" == "3e00" ]]
}

declare -A RECEIPT_FIELDS=()
declare -A RECEIPT_COUNTS=()

load_receipt() {
    local key value extra
    while IFS=$'\t' read -r key value extra || [[ -n "$key" ]]; do
        [[ -n "$key" && -z "$extra" ]] || fail build_receipt_malformed_row
        [[ "$key" =~ ^[a-z0-9_]+$ ]] || fail build_receipt_invalid_key
        RECEIPT_COUNTS["$key"]=$(( ${RECEIPT_COUNTS["$key"]:-0} + 1 ))
        RECEIPT_FIELDS["$key"]="$value"
    done <"$BUILD_RECEIPT"
}

receipt_value() {
    local key="$1"
    local count
    count="${RECEIPT_COUNTS["$key"]:-0}"
    [[ "$count" == "1" ]] || fail "build_receipt_${key}_count_${count}"
    printf '%s\n' "${RECEIPT_FIELDS["$key"]}"
}

MADAROS="${SOUNIO_AFFINE_MADAROS_RAW_BIN:-}"
BUILD_RECEIPT="${SOUNIO_AFFINE_BUILD_RECEIPT:-}"
DEV_MODE="${SOUNIO_AFFINE_DEV_MODE:-0}"
CASE_TIMEOUT_SECONDS="${SOUNIO_AFFINE_CASE_TIMEOUT_SECONDS:-10}"
STDLIB_PATH="${SOUNIO_AFFINE_STDLIB_PATH:-$ROOT_DIR/stdlib}"

[[ "$DEV_MODE" == "0" || "$DEV_MODE" == "1" ]] || fail invalid_dev_mode
[[ "$CASE_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail invalid_case_timeout
(( CASE_TIMEOUT_SECONDS <= 60 )) || fail case_timeout_exceeds_cap_60
[[ -n "$MADAROS" ]] || fail missing_explicit_madaros_bin
[[ "$MADAROS" == /* ]] || fail madaros_bin_must_be_absolute
[[ -f "$MADAROS" && -r "$MADAROS" && -x "$MADAROS" ]] || fail madaros_bin_not_executable
is_elf_binary "$MADAROS" || fail madaros_bin_is_not_elf
is_elf64_x86_64_le "$MADAROS" || fail madaros_bin_wrong_elf_class_data_or_machine
[[ "$STDLIB_PATH" == /* ]] || fail stdlib_path_must_be_absolute
[[ -d "$STDLIB_PATH" && -r "$STDLIB_PATH" ]] || fail stdlib_path_unreadable
STDLIB_PATH="$(realpath "$STDLIB_PATH")"
export SOUNIO_STDLIB_PATH="$STDLIB_PATH"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/madaros-affine-semantics.XXXXXX")"
chmod 700 "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

MADAROS_ORIGINAL="$(realpath "$MADAROS")"
compiler_original_sha256="$(portable_sha256 "$MADAROS_ORIGINAL")"
cp -- "$MADAROS_ORIGINAL" "$TMP_DIR/madaros.elf"
chmod 500 "$TMP_DIR/madaros.elf"
MADAROS="$TMP_DIR/madaros.elf"
compiler_sha256="$(portable_sha256 "$MADAROS")"
[[ "$compiler_sha256" == "$compiler_original_sha256" ]] || fail madaros_changed_while_snapshotting

set +e
identity="$(timeout --signal=TERM --kill-after=2s "${CASE_TIMEOUT_SECONDS}s" \
    "$MADAROS" --version 2>&1)"
identity_rc=$?
set -e
[[ "$identity_rc" == "0" ]] || fail madaros_identity_command_failed
EXPECTED_IDENTITY=$'Madaros v0.80.0 -- the Sounio self-hosted compiler\nthe bare highland that does not negotiate with ill-formed code -- Sfakia, Crete\nHorizon 3: self-hosted primary compiler.'
[[ "$identity" == "$EXPECTED_IDENTITY" ]] \
    || fail madaros_identity_banner_mismatch

source_git_sha="$(git rev-parse HEAD)"
source_tree_sha="$(git rev-parse 'HEAD^{tree}')"
authority="dev-unbound"
merge_ready=0
BUILD_RECEIPT_ORIGINAL=""
build_receipt_original_sha256=""

if [[ "$DEV_MODE" == "0" ]]; then
    [[ -n "${SOUNIO_AFFINE_EXPECTED_SOURCE_SHA:-}" ]] || fail missing_expected_source_git_sha
    [[ "$SOUNIO_AFFINE_EXPECTED_SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || fail invalid_expected_source_git_sha
    [[ -n "$BUILD_RECEIPT" ]] || fail missing_build_receipt
    [[ -f "$BUILD_RECEIPT" && -r "$BUILD_RECEIPT" ]] || fail build_receipt_unreadable
    BUILD_RECEIPT_ORIGINAL="$(realpath "$BUILD_RECEIPT")"
    build_receipt_original_sha256="$(portable_sha256 "$BUILD_RECEIPT_ORIGINAL")"
    cp -- "$BUILD_RECEIPT_ORIGINAL" "$TMP_DIR/build-receipt.tsv"
    BUILD_RECEIPT="$TMP_DIR/build-receipt.tsv"
    [[ "$(portable_sha256 "$BUILD_RECEIPT")" == "$build_receipt_original_sha256" ]] \
        || fail build_receipt_changed_while_snapshotting
    load_receipt
    [[ "$(receipt_value schema)" == "sounio.madaros.build-receipt.v2" ]] || fail build_receipt_schema
    [[ "$(receipt_value build_strategy)" == "derived-current-lean-single" ]] || fail build_receipt_strategy_not_current_source
    [[ "$(receipt_value worktree_clean_before)" == "1" ]] || fail build_receipt_worktree_was_dirty_before
    [[ "$(receipt_value worktree_clean_after)" == "1" ]] || fail build_receipt_worktree_was_dirty_after
    [[ "$(receipt_value source_stable)" == "1" ]] || fail build_receipt_source_changed_during_build
    [[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail current_worktree_dirty
    [[ "$(receipt_value source_git_sha_before)" == "$source_git_sha" ]] || fail build_receipt_source_git_sha_before_mismatch
    [[ "$(receipt_value source_git_sha_after)" == "$source_git_sha" ]] || fail build_receipt_source_git_sha_after_mismatch
    [[ "$(receipt_value source_tree_sha_before)" == "$source_tree_sha" ]] || fail build_receipt_source_tree_sha_before_mismatch
    [[ "$(receipt_value source_tree_sha_after)" == "$source_tree_sha" ]] || fail build_receipt_source_tree_sha_after_mismatch
    [[ "$(receipt_value output_path)" == "$MADAROS_ORIGINAL" ]] || fail build_receipt_output_path_mismatch
    [[ "$(receipt_value output_sha256)" == "$compiler_sha256" ]] || fail build_receipt_output_sha256_mismatch
    [[ "$(receipt_value lean_source_sha256_before)" == "$(portable_sha256 self-hosted/compiler/lean_single.sio)" ]] \
        || fail build_receipt_lean_source_sha256_before_mismatch
    [[ "$(receipt_value lean_source_sha256_after)" == "$(portable_sha256 self-hosted/compiler/lean_single.sio)" ]] \
        || fail build_receipt_lean_source_sha256_after_mismatch
    [[ "$(receipt_value modular_source_sha256_before)" == "$(portable_sha256 self-hosted/compiler/main.sio)" ]] \
        || fail build_receipt_modular_source_sha256_before_mismatch
    [[ "$(receipt_value modular_source_sha256_after)" == "$(portable_sha256 self-hosted/compiler/main.sio)" ]] \
        || fail build_receipt_modular_source_sha256_after_mismatch
    [[ "$(receipt_value build_script_sha256_before)" == "$(portable_sha256 scripts/ci/build_modular_madaros.sh)" ]] \
        || fail build_receipt_build_script_sha256_before_mismatch
    [[ "$(receipt_value build_script_sha256_after)" == "$(portable_sha256 scripts/ci/build_modular_madaros.sh)" ]] \
        || fail build_receipt_build_script_sha256_after_mismatch
    [[ "$source_git_sha" == "$SOUNIO_AFFINE_EXPECTED_SOURCE_SHA" ]] || fail expected_source_git_sha_mismatch
    if [[ -n "${SOUNIO_AFFINE_EXPECTED_COMPILER_SHA256:-}" ]]; then
        [[ "$SOUNIO_AFFINE_EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] \
            || fail invalid_expected_compiler_sha256
        [[ "$compiler_sha256" == "$SOUNIO_AFFINE_EXPECTED_COMPILER_SHA256" ]] || fail expected_compiler_sha256_mismatch
    fi
    authority="current-source-content-bound"
    merge_ready=1
fi

failures=0
positives=0
negatives=0
EXPECTED_POSITIVES=32
EXPECTED_NEGATIVES=48

record_failure() {
    printf 'MADAROS_AFFINE_SEMANTICS_CASE_FAIL label=%s reason=%s\n' "$1" "$2" >&2
    failures=$((failures + 1))
}

fatal_log() {
    grep -Eqi 'segmentation fault|core dumped|internal compiler error|(^|[^a-z])panic([^a-z]|$)|bus error|illegal instruction|assertion failed|stack overflow|out of memory|addresssanitizer|undefinedbehaviorsanitizer|threadsanitizer|memorysanitizer|(^|[^a-z])aborted([^a-z]|$)' "$1"
}

expect_check() {
    local label="$1"
    local source="$2"
    local log="$TMP_DIR/$label.check.log"
    [[ -f "$source" ]] || {
        record_failure "$label" missing_source
        return
    }
    positives=$((positives + 1))
    set +e
    timeout --signal=TERM --kill-after=2s "${CASE_TIMEOUT_SECONDS}s" \
        "$MADAROS" --check "$source" >"$log" 2>&1
    local rc=$?
    set -e
    if [[ "$rc" == "124" ]]; then
        cat "$log" >&2
        record_failure "$label" timeout
    elif [[ "$rc" != "0" ]]; then
        cat "$log" >&2
        record_failure "$label" "unexpected_rc_$rc"
    elif fatal_log "$log"; then
        cat "$log" >&2
        record_failure "$label" fatal_output
    elif ! grep -Fxq 'check: OK' "$log"; then
        cat "$log" >&2
        record_failure "$label" missing_check_receipt
    elif grep -Eq '^error\[E[0-9]{3}' "$log"; then
        cat "$log" >&2
        record_failure "$label" positive_emitted_error
    fi
}

expect_rejection() {
    local label="$1"
    local source="$2"
    local code="$3"
    local message="$4"
    local log="$TMP_DIR/$label.reject.log"
    [[ -f "$source" ]] || {
        record_failure "$label" missing_source
        return
    }
    negatives=$((negatives + 1))
    set +e
    timeout --signal=TERM --kill-after=2s "${CASE_TIMEOUT_SECONDS}s" \
        "$MADAROS" --check "$source" >"$log" 2>&1
    local rc=$?
    set -e
    if [[ "$rc" == "124" ]]; then
        cat "$log" >&2
        record_failure "$label" timeout
        return
    fi
    if [[ "$rc" != "1" ]]; then
        cat "$log" >&2
        record_failure "$label" "expected_semantic_rc_1_actual_$rc"
        return
    fi
    if fatal_log "$log"; then
        cat "$log" >&2
        record_failure "$label" fatal_output
        return
    fi
    local diagnostic_count
    diagnostic_count="$(grep -Ec '^error\[E[0-9]{3}' "$log" || true)"
    if [[ "$diagnostic_count" != "1" ]]; then
        cat "$log" >&2
        record_failure "$label" "expected_one_diagnostic_actual_$diagnostic_count"
        return
    fi
    # Madaros currently wraps some diagnostic closing brackets onto the next
    # line. Anchor the complete numeric code without accepting E0399/E0400.
    if ! grep -Eq "^error\\[$code(\\]|$)" "$log"; then
        cat "$log" >&2
        record_failure "$label" "missing_$code"
    fi
    if ! grep -Fq "$message" "$log"; then
        cat "$log" >&2
        record_failure "$label" missing_exact_diagnostic
    fi
}

# Cross the former 256-entry ownership-visit boundary without committing a
# 300-struct fixture. The last type must still retain affine ownership and the
# second move must be rejected rather than indexing beyond the visit set.
large_struct_source="$TMP_DIR/affine_many_structs_reuse.sio"
printf '%s\n' '//@ compile-fail' >"$large_struct_source"
for ((struct_index = 0; struct_index < 300; struct_index++)); do
    printf 'struct Padding%s { value: i64 }\n' "$struct_index" >>"$large_struct_source"
done
printf '%s\n' \
    'affine struct Permit { id: i64 }' \
    'fn spend(p: Permit) -> i64 { p.id }' \
    'fn main() -> i32 { let p = Permit { id: 61 }; let a = spend(p); let b = spend(p); (a + b) as i32 }' \
    >>"$large_struct_source"

# Cross the borrow environment's explicit 128-binding capacity. The 129th
# binding must reject with a compiler-resource diagnostic instead of becoming
# invisible to ownership and borrow checking.
borrow_capacity_source="$TMP_DIR/affine_borrow_tracking_capacity.sio"
printf '%s\n' \
    '//@ compile-fail' \
    'affine struct Permit { id: i64 }' \
    'fn main() -> i32 {' \
    >"$borrow_capacity_source"
for ((binding_index = 0; binding_index < 128; binding_index++)); do
    printf '    let filler%s = %s\n' "$binding_index" "$binding_index" >>"$borrow_capacity_source"
done
printf '%s\n' \
    '    let permit = Permit { id: 67 }' \
    '    0' \
    '}' \
    >>"$borrow_capacity_source"

# Cross the former depth-16 reference search through named declarations. The
# branch result carries a borrow 18 structs deep; that borrow must survive the
# join and keep the affine referent unavailable for a move.
deep_reference_source="$TMP_DIR/affine_deep_reference_survives_join.sio"
printf '%s\n' \
    '//@ compile-fail' \
    'affine struct Permit { id: i64 }' \
    'struct Ref0 { permit: &Permit }' \
    >"$deep_reference_source"
for ((reference_depth = 1; reference_depth < 18; reference_depth++)); do
    printf 'struct Ref%s { inner: Ref%s }\n' "$reference_depth" "$((reference_depth - 1))" >>"$deep_reference_source"
done
deep_reference_expr='Ref0 { permit: &permit }'
deep_reference_access='carried'
for ((reference_depth = 1; reference_depth < 18; reference_depth++)); do
    deep_reference_expr="Ref${reference_depth} { inner: ${deep_reference_expr} }"
    deep_reference_access="${deep_reference_access}.inner"
done
deep_reference_access="${deep_reference_access}.permit"
printf '%s\n' \
    'fn spend(p: Permit) -> i64 { p.id }' \
    'fn main() -> i32 {' \
    '    let permit = Permit { id: 71 }' \
    '    let return_early = false' \
    "    let carried = if return_early { return 0 } else { ${deep_reference_expr} }" \
    "    let observed = (*${deep_reference_access}).id" \
    '    let consumed = spend(permit)' \
    '    (observed + consumed) as i32' \
    '}' \
    >>"$deep_reference_source"

# Affine means at most once: abandonment is legal, duplication is not.
expect_check affine_drop tests/run-pass/affine_can_drop.sio
expect_check affine_once tests/run-pass/affine_consume_once.sio
expect_rejection affine_double_use tests/compile-fail/affine_double_use.sio E039 \
    'affine value has already been used'
expect_rejection affine_direct_binding_reuse tests/compile-fail/affine_direct_binding_reuse.sio E039 \
    'affine value has already been used'
expect_check affine_generic_identity tests/run-pass/affine_generic_identity.sio
expect_rejection affine_generic_duplicate tests/compile-fail/affine_generic_duplicate.sio E039 \
    'affine value has already been used'
expect_check affine_generic_struct_annotation tests/run-pass/affine_generic_struct_annotation.sio
expect_check affine_generic_struct_literal_once tests/run-pass/affine_generic_struct_literal_once.sio
expect_rejection affine_generic_struct_reuse tests/compile-fail/affine_generic_struct_reuse.sio E039 \
    'affine value has already been used'
expect_rejection affine_nested_generic_struct_reuse tests/compile-fail/affine_nested_generic_struct_reuse.sio E039 \
    'affine value has already been used'
expect_rejection affine_many_structs_reuse "$large_struct_source" E039 \
    'affine value has already been used'
expect_rejection affine_borrow_tracking_capacity "$borrow_capacity_source" E064 \
    'type complexity budget exceeded'
expect_check affine_imported_once tests/run-pass/affine_imported_consume_once.sio
expect_rejection affine_imported_reuse tests/compile-fail/affine_imported_reuse.sio E039 \
    'affine value has already been used'

# Closure ownership is the ownership of moved captures. Borrow-only captures
# remain reusable but keep the referent unavailable for a move.
expect_check affine_closure_once tests/run-pass/affine_closure_capture_once.sio
expect_rejection affine_closure_reuse tests/compile-fail/affine_closure_capture_reuse.sio E039 \
    'affine value has already been used'
expect_check linear_closure_once tests/run-pass/linear_closure_capture_once.sio
expect_rejection linear_closure_unconsumed tests/compile-fail/linear_closure_capture_unconsumed.sio E040 \
    'linear value not consumed'
expect_check affine_closure_borrow_reusable tests/run-pass/affine_closure_borrow_reusable.sio
expect_rejection affine_closure_borrow_then_move tests/compile-fail/affine_closure_borrow_then_move.sio E038 \
    'cannot move affine value while borrowed'
expect_check closure_ignores_uncaptured_linear tests/run-pass/closure_ignores_uncaptured_linear.sio
expect_rejection affine_closure_cannot_erase_to_fn tests/compile-fail/affine_closure_cannot_erase_to_fn.sio E009 \
    'argument type does not match parameter'
expect_check affine_closure_parameter_once tests/run-pass/affine_closure_parameter_once_per_call.sio
expect_rejection affine_closure_parameter_reuse tests/compile-fail/affine_closure_parameter_reuse.sio E039 \
    'affine value has already been used'
expect_rejection linear_closure_parameter_unconsumed tests/compile-fail/linear_closure_parameter_unconsumed.sio E040 \
    'linear value not consumed'

# Assignment places observe their root without moving it, but the root must be
# live and unborrowed. Index computations retain ordinary ownership effects.
expect_check affine_bare_reassignment tests/run-pass/affine_bare_reassignment.sio
expect_rejection affine_borrowed_bare_reassignment tests/compile-fail/affine_borrowed_bare_reassignment.sio E038 \
    'cannot move affine value while borrowed'
expect_rejection affine_moved_field_assignment tests/compile-fail/affine_moved_field_assignment.sio E039 \
    'affine value has already been used'
expect_rejection linear_field_overwrite tests/compile-fail/linear_field_overwrite_unconsumed.sio E040 \
    'linear value not consumed'
expect_rejection affine_moved_array_assignment tests/compile-fail/affine_moved_array_element_assignment.sio E039 \
    'affine value has already been used'
expect_rejection affine_assignment_index_reuse tests/compile-fail/affine_assignment_index_expression_reuse.sio E039 \
    'affine value has already been used'

# Array repetition is a semantic duplication site even though its element AST
# appears only once.
expect_check affine_array_repeat_once tests/run-pass/affine_array_repeat_once.sio
expect_check affine_array_repeat_zero tests/run-pass/affine_array_repeat_zero.sio
expect_rejection affine_array_repeat_duplicate tests/compile-fail/affine_array_repeat_duplicate.sio E039 \
    'affine value has already been used'
expect_rejection linear_array_repeat_zero tests/compile-fail/linear_array_repeat_zero.sio E040 \
    'linear value not consumed'
expect_rejection affine_tuple_element_seventeen_reuse tests/compile-fail/affine_tuple_element_seventeen_reuse.sio E039 \
    'affine value has already been used'

# Owning wrappers preserve the ownership mode of their payload.
expect_check affine_owned_wrappers_once tests/run-pass/affine_owned_wrappers_once.sio
expect_rejection affine_option_duplicate tests/compile-fail/affine_option_duplicate.sio E039 \
    'affine value has already been used'
expect_rejection affine_box_duplicate tests/compile-fail/affine_box_duplicate.sio E039 \
    'affine value has already been used'
expect_rejection affine_enum_payload_reuse tests/compile-fail/affine_enum_payload_reuse.sio E039 \
    'affine value has already been used'
expect_check unrestricted_recursive_owned_type tests/run-pass/unrestricted_recursive_owned_type.sio

# A one-sided branch move is valid for affine values, but poisons the join.
# Non-fallthrough alternatives do not participate in that join.
expect_check affine_branch_optional tests/run-pass/affine_branch_optional_consume.sio
expect_check affine_if_branch_borrow_isolated tests/run-pass/affine_if_branch_borrow_isolated.sio
expect_check affine_if_return_path_does_not_poison_join tests/run-pass/affine_if_return_path_does_not_poison_join.sio
expect_rejection affine_if_reference_survives_divergent_join tests/compile-fail/affine_if_reference_survives_divergent_join.sio E038 \
    'cannot move affine value while borrowed'
expect_rejection affine_if_tuple_reference_survives_join tests/compile-fail/affine_if_tuple_reference_survives_join.sio E038 \
    'cannot move affine value while borrowed'
expect_rejection affine_if_enum_reference_survives_join tests/compile-fail/affine_if_enum_reference_survives_join.sio E038 \
    'cannot move affine value while borrowed'
expect_rejection affine_deep_reference_survives_join "$deep_reference_source" E038 \
    'cannot move affine value while borrowed'
expect_rejection affine_branch_post_join_reuse tests/compile-fail/affine_branch_post_join_reuse.sio E039 \
    'affine value has already been used'
expect_check affine_match_optional tests/run-pass/affine_match_optional_consume.sio
expect_check linear_match_return_path_does_not_join tests/run-pass/linear_match_return_path_does_not_join.sio
expect_rejection affine_match_reference_union_survives_join tests/compile-fail/affine_match_reference_union_survives_join.sio E038 \
    'cannot move affine value while borrowed'
expect_rejection affine_match_post_join_reuse tests/compile-fail/affine_match_post_join_reuse.sio E039 \
    'affine value has already been used'
expect_check affine_match_arm_borrow_isolated tests/run-pass/affine_match_arm_borrow_isolated.sio
expect_rejection affine_match_guard_move tests/compile-fail/affine_match_guard_move.sio E039 \
    'affine value cannot be consumed in a match guard'

# Return, borrow, loop, and transitive-field ownership each have a positive and
# a discriminating negative control.
expect_check affine_return tests/run-pass/affine_return_value.sio
expect_rejection affine_return_reuse tests/compile-fail/affine_return_reuse.sio E039 \
    'affine value has already been used'
expect_check affine_borrow_then_move tests/run-pass/affine_borrow_call_then_move.sio
expect_rejection affine_move_while_borrowed tests/compile-fail/affine_move_while_borrowed.sio E038 \
    'cannot move affine value while borrowed'
expect_check affine_loop_local tests/run-pass/affine_created_and_consumed_in_loop.sio
expect_rejection affine_loop_outer_move tests/compile-fail/affine_outer_move_in_loop.sio E039 \
    'affine value consumed in potentially repeating loop'
expect_check affine_transitive_field tests/run-pass/affine_transitive_field_consume.sio
expect_rejection affine_transitive_field_reuse tests/compile-fail/affine_transitive_field_reuse.sio E039 \
    'affine value has already been used'

# Linear remains exactly once on every path.
expect_check linear_once tests/run-pass/linear_consume_once.sio
expect_check linear_return tests/run-pass/linear_return_value.sio
expect_check linear_balanced_branches tests/run-pass/linear_balanced_branches.sio
expect_check linear_balanced_match tests/run-pass/linear_match_balanced_ownership.sio
expect_check linear_field_access tests/run-pass/linear_struct_field_access.sio
expect_check borrow_call_release tests/run-pass/borrow_call_explicit_release.sio
expect_rejection linear_double_use tests/compile-fail/linear_double_use.sio E039 \
    'linear value has already been used'
expect_rejection linear_unconsumed tests/compile-fail/linear_not_consumed.sio E040 \
    'linear value not consumed'
expect_rejection linear_branch_asymmetry tests/compile-fail/linear_branch_asymmetry.sio E040 \
    'linear value consumed in then-branch but not else-branch'
expect_rejection linear_match_asymmetry tests/compile-fail/linear_match_ownership_asymmetry.sio E040 \
    'linear value consumed in some match arms but not all match arms'
expect_rejection linear_parameter_unused tests/compile-fail/linear_parameter_unused.sio E040 \
    'linear value not consumed'
expect_rejection linear_loop_outer_move tests/compile-fail/linear_loop_consume.sio E039 \
    'linear value consumed in potentially repeating loop'
expect_rejection linear_transitive_field_unconsumed tests/compile-fail/linear_field_unconsumed.sio E040 \
    'linear value not consumed'
expect_rejection linear_early_return tests/compile-fail/linear_early_return.sio E040 \
    'linear value not consumed'
expect_rejection linear_reassignment_loss tests/compile-fail/linear_reassign_lost.sio E040 \
    'linear value not consumed'
expect_rejection linear_field_reuse tests/compile-fail/linear_multiple_field_access.sio E039 \
    'linear value has already been used'
expect_rejection linear_move_while_borrowed tests/compile-fail/ownership_move_while_borrowed.sio E038 \
    'cannot move linear value while borrowed'

if [[ "$positives" != "$EXPECTED_POSITIVES" || "$negatives" != "$EXPECTED_NEGATIVES" ]]; then
    record_failure matrix_cardinality \
        "expected_${EXPECTED_POSITIVES}_${EXPECTED_NEGATIVES}_actual_${positives}_${negatives}"
fi

[[ "$(portable_sha256 "$MADAROS")" == "$compiler_sha256" ]] \
    || fail compiler_snapshot_changed_during_gate
[[ "$(portable_sha256 "$MADAROS_ORIGINAL")" == "$compiler_original_sha256" ]] \
    || fail compiler_source_changed_during_gate
if [[ "$DEV_MODE" == "0" ]]; then
    [[ "$(portable_sha256 "$BUILD_RECEIPT_ORIGINAL")" == "$build_receipt_original_sha256" ]] \
        || fail build_receipt_source_changed_during_gate
    [[ "$(git rev-parse HEAD)" == "$source_git_sha" ]] || fail source_git_sha_changed_during_gate
    [[ "$(git rev-parse 'HEAD^{tree}')" == "$source_tree_sha" ]] || fail source_tree_sha_changed_during_gate
    [[ -z "$(git status --porcelain --untracked-files=all)" ]] || fail worktree_changed_during_gate
fi

if [[ "$failures" != "0" ]]; then
    fail "matrix_failures_${failures}_positives_${positives}_negatives_${negatives}"
fi

printf 'MADAROS_AFFINE_SEMANTICS_PASS authority=%s merge_ready=%s positives=%s negatives=%s compiler=%s compiler_sha256=%s source_git_sha=%s source_tree_sha=%s stdlib=%s fallback=0\n' \
    "$authority" "$merge_ready" "$positives" "$negatives" "$MADAROS_ORIGINAL" "$compiler_sha256" \
    "$source_git_sha" "$source_tree_sha" "$STDLIB_PATH"
