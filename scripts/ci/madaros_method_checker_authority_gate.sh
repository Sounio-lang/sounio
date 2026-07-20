#!/usr/bin/env bash
# Prove receiver-first method lookup, visibility filtering, and structured failure.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WRAPPER="$ROOT_DIR/bin/madaros"
RAW_MADAROS="${SOUNIO_MADAROS_METHOD_CHECKER_RAW_BIN:-}"
KEEP_WORK="${SOUNIO_MADAROS_METHOD_CHECKER_KEEP:-0}"
TIMEOUT_SECONDS="${SOUNIO_MADAROS_METHOD_CHECKER_TIMEOUT_SECONDS:-180}"
FIXTURES="$ROOT_DIR/tests/compiler/madaros_method_checker_authority"
ASSOCIATED_REEXPORT="$ROOT_DIR/tests/compiler/module_graph_impl_authority_reexport/main.sio"
VISIBILITY_FIXTURES="$ROOT_DIR/tests/multimodule"

fail() {
  printf '[madaros-method-checker-authority] FAIL: %s\n' "$1" >&2
  exit 1
}

is_fatal_log() {
  grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$1"
}

run_check() {
  local label="$1"
  local source="$2"
  local log="$WORK/$label.log"
  local rc=0

  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
    "$WRAPPER" --science-boundary off check "$source" >"$log" 2>&1
  rc=$?
  set -e
  printf '%s' "$rc" >"$WORK/$label.rc"
  if is_fatal_log "$log"; then
    fail "${label}_fatal"
  fi
  return 0
}

expect_success() {
  local label="$1"
  local source="$2"
  local log="$WORK/$label.log"

  run_check "$label" "$source"
  [[ "$(cat "$WORK/$label.rc")" -eq 0 ]] || {
    cat "$log" >&2 || true
    fail "${label}_expected_success"
  }
  grep -Fq 'run_check_mode: verdict=0' "$log" || fail "${label}_verdict_missing"
  grep -Fq 'check: OK' "$log" || fail "${label}_ok_missing"
  ! grep -Fq 'error[E' "$log" || fail "${label}_diagnostic_on_success"
}

expect_single_success() {
  local label="$1"
  local source="$2"
  local log="$WORK/$label.log"

  run_check "$label" "$source"
  [[ "$(cat "$WORK/$label.rc")" -eq 0 ]] || {
    cat "$log" >&2 || true
    fail "${label}_expected_success"
  }
  grep -Fq 'check: OK' "$log" || fail "${label}_ok_missing"
  ! grep -Fq 'error[E' "$log" || fail "${label}_diagnostic_on_success"
  ! grep -Fq 'run_check_mode: about to check' "$log" || fail "${label}_unexpected_multimodule_path"
}

expect_single_rejection() {
  local label="$1"
  local source="$2"
  local code="$3"
  local message="$4"
  local log="$WORK/$label.log"

  run_check "$label" "$source"
  [[ "$(cat "$WORK/$label.rc")" -eq 1 ]] || {
    cat "$log" >&2 || true
    fail "${label}_expected_rc_1"
  }
  [[ "$(grep -Fc "error[$code" "$log" || true)" -eq 1 ]] || fail "${label}_${code}_count"
  [[ "$(grep -Fc "$message" "$log" || true)" -eq 1 ]] || fail "${label}_message_count"
  [[ "$(grep -Fc 'error[E' "$log" || true)" -eq 1 ]] || fail "${label}_diagnostic_count"
  ! grep -Fq 'check: OK' "$log" || fail "${label}_unexpected_ok"
  ! grep -Fq 'run_check_mode: about to check' "$log" || fail "${label}_unexpected_multimodule_path"
}

expect_rejection() {
  local label="$1"
  local source="$2"
  local code="$3"
  local message="$4"
  local log="$WORK/$label.log"

  run_check "$label" "$source"
  [[ "$(cat "$WORK/$label.rc")" -eq 1 ]] || {
    cat "$log" >&2 || true
    fail "${label}_expected_rc_1"
  }
  [[ "$(grep -Fc "error[$code" "$log" || true)" -eq 1 ]] || fail "${label}_${code}_count"
  [[ "$(grep -Fc "$message" "$log" || true)" -eq 1 ]] || fail "${label}_message_count"
  [[ "$(grep -Fc 'error[E' "$log" || true)" -eq 1 ]] || fail "${label}_diagnostic_count"
  grep -Fq 'run_check_mode: verdict=1' "$log" || fail "${label}_verdict_missing"
  ! grep -Fq 'check: OK' "$log" || fail "${label}_unexpected_ok"
}

if [[ -n "${SOUNIO_MADAROS_METHOD_CHECKER_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_METHOD_CHECKER_DIR"
  [[ ! -e "$WORK" ]] || fail "work_directory_already_exists path=$WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-madaros-method-checker-authority.XXXXXX")"
fi

if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail x86_64_required ;;
esac
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail invalid_timeout_seconds
[[ -x "$WRAPPER" ]] || fail madaros_wrapper_missing
[[ -n "$RAW_MADAROS" ]] || fail explicit_raw_required
[[ -x "$RAW_MADAROS" ]] || fail explicit_raw_missing_or_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ "$(od -An -tx1 -N4 "$RAW_MADAROS" | tr -d ' \n')" == 7f454c46 ]] || fail raw_compiler_must_be_elf
[[ -f "$ASSOCIATED_REEXPORT" ]] || fail associated_reexport_fixture_missing
for fixture in visibility_fn_private_main.sio visibility_struct_private_main.sio visibility_enum_private_main.sio; do
  [[ -f "$VISIBILITY_FIXTURES/$fixture" ]] || fail "visibility_fixture_missing_$fixture"
done

for fixture in \
  public_unique_main.sio public_unique_leaf.sio \
  private_cross_module_main.sio private_cross_module_leaf.sio \
  ambiguous_remote_main.sio ambiguous_remote_reversed_main.sio \
  ambiguous_remote_target.sio ambiguous_remote_distractor.sio \
  private_distractor_main.sio private_distractor_reversed_main.sio \
  private_distractor_target.sio private_distractor_unrelated.sio \
  ambiguous_same_receiver_main.sio ambiguous_same_receiver_marker.sio \
  method_body_contract_main.sio method_body_contract_leaf.sio \
  local_precedence_main.sio local_precedence_distractor.sio \
  single_local_helper_main.sio single_undefined_main.sio; do
  [[ -f "$FIXTURES/$fixture" ]] || fail "fixture_missing_$fixture"
done

# Source-shape guard: the active in-place path and every retained by-value path
# route through one structured resolver. Six expression paths map structured
# failures to diagnostics; two impl-body paths bind definitions separately.
[[ "$(grep -Fc 'let method_lookup = method_lookup_resolve(' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 6 ]] \
  || fail method_lookup_callsite_count
[[ "$(grep -Fc 'method_lookup_error_code(method_lookup' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 6 ]] \
  || fail method_lookup_diagnostic_callsite_count
[[ "$(grep -Fc 'let sig_id = method_definition_sig_id(' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 2 ]] \
  || fail method_definition_binding_callsite_count
[[ "$(grep -Fc '&& sig.source_span_start == source_span_start' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 1 ]] \
  || fail method_definition_source_identity_missing
[[ "$(grep -Fc 'METHOD_LOOKUP_FOUND' "$ROOT_DIR/self-hosted/check/check.sio")" -ge 2 ]] \
  || fail method_lookup_found_state_missing
[[ "$(grep -Fc 'METHOD_LOOKUP_PRIVATE' "$ROOT_DIR/self-hosted/check/check.sio")" -ge 2 ]] \
  || fail method_lookup_private_state_missing
[[ "$(grep -Fc 'METHOD_LOOKUP_AMBIGUOUS' "$ROOT_DIR/self-hosted/check/check.sio")" -ge 3 ]] \
  || fail method_lookup_ambiguous_state_missing
[[ "$(grep -Fc 'METHOD_LOOKUP_MISSING' "$ROOT_DIR/self-hosted/check/check.sio")" -ge 3 ]] \
  || fail method_lookup_missing_state_missing
[[ "$(grep -Ec 'fn_sig_table_find_method(_semantic)?\(' "$ROOT_DIR/self-hosted/check/check.sio" || true)" -eq 0 ]] \
  || fail legacy_method_lookup_reintroduced
[[ "$(grep -Fc 'return sig_module_id >= 0 && sig_module_id == receiver_module_id' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 1 ]] \
  || fail nominal_receiver_identity_guard_missing
[[ "$(grep -Fc 'if visible_count == 1' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 1 ]] \
  || fail visibility_before_uniqueness_guard_missing
[[ "$(grep -Fc 'if inaccessible_count > 0' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 1 ]] \
  || fail private_outcome_guard_missing
[[ "$(grep -Fc 'else if code == 219 { print("method resolution is ambiguous for this receiver type") }' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 1 ]] \
  || fail ambiguity_diagnostic_missing
[[ "$(grep -Fc 'current_module_id: CHECK_MODULE_ID_UNKNOWN' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 1 ]] \
  || fail checker_value_init_unknown_missing
[[ "$(grep -Fc 'current_module_id = CHECK_MODULE_ID_UNKNOWN' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 1 ]] \
  || fail checker_inplace_init_unknown_missing
[[ "$(grep -Fc 'visibility_allows_access_with_module_identity(' "$ROOT_DIR/self-hosted/check/check.sio")" -ge 12 ]] \
  || fail module_identity_visibility_not_generalized
[[ "$(grep -Ec 'visibility_allows_access\(' "$ROOT_DIR/self-hosted/check/check.sio")" -eq 2 ]] \
  || fail direct_path_visibility_reintroduced
[[ "$(grep -Fc 'check_items_verdict_boot4_with_identity(items, empty_path(), CHECK_MODULE_ID_UNKNOWN)' "$ROOT_DIR/self-hosted/check/mod.sio")" -eq 1 ]] \
  || fail merged_items_unknown_identity_missing
[[ "$(grep -Fc 'check_items_verdict_boot4_with_identity(items, module_path, 0)' "$ROOT_DIR/self-hosted/check/mod.sio")" -eq 1 ]] \
  || fail standalone_root_identity_missing

expect_success public_unique "$FIXTURES/public_unique_main.sio"
expect_single_success single_local_helper "$FIXTURES/single_local_helper_main.sio"
expect_single_rejection single_undefined "$FIXTURES/single_undefined_main.sio" \
  E137 'use of undeclared variable'
expect_success local_precedence "$FIXTURES/local_precedence_main.sio"
expect_success nominal_target_distractor_first "$FIXTURES/ambiguous_remote_main.sio"
expect_success nominal_target_target_first "$FIXTURES/ambiguous_remote_reversed_main.sio"
expect_success private_distractor_first "$FIXTURES/private_distractor_main.sio"
expect_success private_distractor_target_first "$FIXTURES/private_distractor_reversed_main.sio"
expect_success associated_reexport_compat "$ASSOCIATED_REEXPORT"
expect_rejection private_function "$VISIBILITY_FIXTURES/visibility_fn_private_main.sio" \
  E175 'function is private in its defining module'
expect_rejection private_struct "$VISIBILITY_FIXTURES/visibility_struct_private_main.sio" \
  E176 'struct constructor is private in its defining module'
expect_rejection private_enum "$VISIBILITY_FIXTURES/visibility_enum_private_main.sio" \
  E177 'enum constructor is private in its defining module'
expect_rejection private_cross_module "$FIXTURES/private_cross_module_main.sio" \
  E175 'function is private in its defining module'
expect_rejection ambiguous_same_receiver "$FIXTURES/ambiguous_same_receiver_main.sio" \
  E219 'method resolution is ambiguous for this receiver type'
expect_rejection method_body_contract "$FIXTURES/method_body_contract_main.sio" \
  E008 "return value does not match function's declared return type"

printf 'MADAROS_METHOD_CHECKER_AUTHORITY_PASS public_unique=pass single_local_helper=pass single_undefined=E137 root_module_id=0 merged_items_module_id=UNKNOWN local_precedence=pass nominal_identity_orders=passx2 private_distractor_orders=passx2 associated_reexport_compat=pass private_surfaces=fn:E175+struct:E176+enum:E177+method:E175 ambiguity=E219 method_body_contract=E008 method_lookup=receiver-first+visibility-before-uniqueness outcomes=FOUND+PRIVATE+AMBIGUOUS+MISSING legacy_fallback=witness-probe-unknown checker_paths=inplace-executed+by-value-source-routed selective_import_authority=not_claimed generic_mangling_identity=not_claimed forward_reference_backpatch=not_claimed\n'
