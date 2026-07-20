#!/usr/bin/env bash
# Prove selective-import and qualified-call authority from AST closure to ELF.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FRONTEND="$ROOT_DIR/self-hosted/compiler/module_frontend.sio"
LOWER="$ROOT_DIR/self-hosted/ir/lower.sio"
FIXTURES="$ROOT_DIR/tests/compiler/madaros_import_call_authority"
WRAPPER="$ROOT_DIR/bin/madaros"
RAW_MADAROS="${SOUNIO_IMPORT_AUTHORITY_RAW_BIN:-}"
EXPECTED_COMPILER_SHA256="${SOUNIO_IMPORT_AUTHORITY_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_IMPORT_AUTHORITY_KEEP:-0}"
TIMEOUT_SECONDS="${SOUNIO_IMPORT_AUTHORITY_TIMEOUT_SECONDS:-180}"
MODE="source-only"

fail() {
  printf 'MADAROS_IMPORT_CALL_AUTHORITY_FAIL reason=%s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<'EOF'
usage: scripts/ci/madaros_import_call_authority_gate.sh [--source-only|--runtime]

  --source-only  Verify source authority wiring and the complete fixture matrix.
  --runtime      Require an explicit raw source-fresh ELF plus exact SHA-256,
                 compile all witnesses, and execute the accepted ELFs.

Runtime mode requires:
  SOUNIO_IMPORT_AUTHORITY_RAW_BIN=/path/to/source-fresh/madaros
  SOUNIO_IMPORT_AUTHORITY_EXPECTED_SHA256=<64 lowercase hex characters>
EOF
}

case "${1:-}" in
  ""|--source-only) MODE="source-only" ;;
  --runtime) MODE="runtime" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; fail unexpected_argument ;;
esac
[[ $# -le 1 ]] || fail unexpected_argument

for path in \
  "$FRONTEND" \
  "$LOWER" \
  "$FIXTURES/selective_negative/main.sio" \
  "$FIXTURES/selective_negative/leaf.sio" \
  "$FIXTURES/qualified/main.sio" \
  "$FIXTURES/qualified/leaf.sio" \
  "$FIXTURES/duplicate_order/left.sio" \
  "$FIXTURES/duplicate_order/right.sio" \
  "$FIXTURES/duplicate_order/left_first.sio" \
  "$FIXTURES/duplicate_order/right_first.sio"; do
  [[ -f "$path" ]] || fail "missing_${path#"$ROOT_DIR"/}"
done

binding_shape="$(sed -n '/^pub struct LowerExternBindingHandle {/,/^}/p' "$LOWER")"
collector_shape="$(sed -n '/^pub fn module_frontend_collect_ast_closure_programs_into(/,/^\/\/ Validate that the collection generation/p' "$FRONTEND")"
authority_shape="$(sed -n '/^fn module_frontend_prepare_import_authority(/,/^fn module_frontend_assign_function_provenance(/p' "$FRONTEND")"
compile_shape="$(sed -n '/^pub fn module_frontend_compile_collected_to_file(/,/^pub fn module_frontend_imported_native_compile(/p' "$FRONTEND")"
lower_binding_shape="$(sed -n '/^fn lowerer_extern_binding_index(/,/^fn lowerer_preseed_external_impl_method_items_mut(/p' "$LOWER")"

for field in caller_module_id local_name defining_module_id export_name qualifier_path; do
  grep -Fq "pub $field:" <<<"$binding_shape" || fail "binding_field_${field}_missing"
done
grep -Fq 'MF_IMPORT_AUTHORITY_PREPARED_COLLECTION_ID = (*out).collection_id' <<<"$collector_shape" ||
  fail collector_authority_receipt_missing
grep -Fq 'module_frontend_prepare_import_authority(' <<<"$collector_shape" ||
  fail collector_authority_prepare_missing
grep -Fq 'module_frontend_authority_rewrite_item_list_opt(' <<<"$authority_shape" ||
  fail authority_ast_rewrite_missing
grep -Fq '(*callee).kind == ExprKind::ExprPath' "$FRONTEND" || fail qualified_path_classifier_missing
grep -Fq '(*rewritten).kind = ExprKind::ExprIdent' "$FRONTEND" || fail qualified_path_rewrite_missing
grep -Fq 'module_frontend_prepend_used_checker_stubs(' <<<"$authority_shape" ||
  fail checker_stub_adapter_missing
grep -Fq 'visibility_is_pub((*item).visibility)' "$FRONTEND" || fail checker_stub_public_guard_missing
grep -Fq 'MADAROS_IMPORT_AUTHORITY_REFUSAL schema=1 global_unique_fallback=0' <<<"$authority_shape" ||
  fail refusal_receipt_missing
grep -Fq 'MADAROS_IMPORT_AUTHORITY_RECEIPT schema=1 modules=' <<<"$authority_shape" ||
  fail acceptance_receipt_missing
grep -Fq 'module_frontend_strip_import_authority_stubs(programs, loaded)' <<<"$compile_shape" ||
  fail checker_stub_strip_missing
grep -Fq 'trace.last_stage = "lower_merge"' <<<"$compile_shape" || fail lower_boundary_missing

for guard in \
  '(*binding).caller_module_id == caller_module_id' \
  '(*binding).defining_module_id == defining_module_id' \
  'ir_name_eq((*binding).export_name, export_name)'; do
  grep -Fq "$guard" <<<"$lower_binding_shape" || fail "lower_binding_guard_missing_${guard//[^a-zA-Z0-9]/_}"
done
if grep -Fq 'if found_count == 1' "$FRONTEND"; then
  fail global_unique_fallback_reintroduced
fi

grep -Fq 'use leaf::{allowed}' "$FIXTURES/selective_negative/main.sio" || fail selective_fixture_import
grep -Fq 'leaked()' "$FIXTURES/selective_negative/main.sio" || fail selective_fixture_leak_call
grep -Fq 'leaf::answer()' "$FIXTURES/qualified/main.sio" || fail qualified_fixture_call
[[ "$(grep -Fc 'pub fn picked()' "$FIXTURES/duplicate_order/left.sio")" -eq 1 ]] || fail left_picked_missing
[[ "$(grep -Fc 'pub fn picked()' "$FIXTURES/duplicate_order/right.sio")" -eq 1 ]] || fail right_picked_missing

left_first_left_line="$(grep -nF 'use left::{picked}' "$FIXTURES/duplicate_order/left_first.sio" | cut -d: -f1)"
left_first_right_line="$(grep -nF 'use right::{right_anchor}' "$FIXTURES/duplicate_order/left_first.sio" | cut -d: -f1)"
right_first_left_line="$(grep -nF 'use left::{picked}' "$FIXTURES/duplicate_order/right_first.sio" | cut -d: -f1)"
right_first_right_line="$(grep -nF 'use right::{right_anchor}' "$FIXTURES/duplicate_order/right_first.sio" | cut -d: -f1)"
(( left_first_left_line < left_first_right_line )) || fail left_first_fixture_order
(( right_first_right_line < right_first_left_line )) || fail right_first_fixture_order

source_sha256="$({
  sha256sum "$FRONTEND" | awk '{print $1}'
  sha256sum "$LOWER" | awk '{print $1}'
} | sha256sum | awk '{print $1}')"

printf 'MADAROS_IMPORT_CALL_AUTHORITY_SOURCE_PASS binding=caller+local+defining+export+qualifier checker_adapter=collection-owned lowering=explicit-only global_unique_fallback=absent fixtures=selective-reject,qualified-call,duplicate-order-x2 source_sha256=%s\n' "$source_sha256"

if [[ "$MODE" == "source-only" ]]; then
  exit 0
fi

[[ "$(uname -s 2>/dev/null || true)" == Linux ]] || fail linux_required
case "$(uname -m 2>/dev/null || true)" in
  x86_64|amd64) ;;
  *) fail x86_64_required ;;
esac
[[ "$TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || fail invalid_timeout_seconds
[[ -x "$WRAPPER" ]] || fail madaros_wrapper_missing
[[ -n "$RAW_MADAROS" ]] || fail explicit_source_fresh_raw_required
[[ -x "$RAW_MADAROS" ]] || fail source_fresh_raw_not_executable
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ "$(od -An -tx1 -N4 "$RAW_MADAROS" | tr -d ' \n')" == 7f454c46 ]] || fail source_fresh_raw_must_be_elf
[[ "$EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail expected_compiler_sha256_required
compiler_sha256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] || fail source_fresh_compiler_sha256_mismatch

if [[ -n "${SOUNIO_IMPORT_AUTHORITY_WORK_DIR:-}" ]]; then
  WORK="$SOUNIO_IMPORT_AUTHORITY_WORK_DIR"
  [[ ! -e "$WORK" ]] || fail work_directory_exists
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-import-authority.XXXXXX")"
fi
if [[ "$KEEP_WORK" != 1 ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RUNNER=(env "MADAROS_RAW_BIN=$RAW_MADAROS" "$WRAPPER" --science-boundary off)

compile_case() {
  local label="$1"
  local source="$2"
  local elf="$WORK/$label.elf"
  local log="$WORK/$label.compile.log"

  if ! timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
      "${RUNNER[@]}" compile "$source" -o "$elf" >"$log" 2>&1; then
    cat "$log" >&2 || true
    fail "${label}_compile_failed"
  fi
  [[ -s "$elf" ]] || fail "${label}_elf_missing"
  [[ "$(od -An -tx1 -N4 "$elf" | tr -d ' \n')" == 7f454c46 ]] || fail "${label}_output_not_elf"
  grep -Fq 'MADAROS_IMPORT_AUTHORITY_RECEIPT schema=1' "$log" || fail "${label}_authority_receipt_missing"
  grep -Fq 'global_unique_fallback=0' "$log" || fail "${label}_fallback_receipt_missing"
}

run_exit_42() {
  local label="$1"
  local elf="$WORK/$label.elf"
  local rc=0

  set +e
  timeout --signal=TERM --kill-after=5s 30s "$elf" >"$WORK/$label.stdout" 2>"$WORK/$label.stderr"
  rc=$?
  set -e
  [[ "$rc" -eq 42 ]] || {
    cat "$WORK/$label.stdout" >&2 || true
    cat "$WORK/$label.stderr" >&2 || true
    fail "${label}_runtime_rc_${rc}"
  }
}

negative_elf="$WORK/selective_negative.elf"
negative_log="$WORK/selective_negative.compile.log"
negative_rc=0
set +e
timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
  "${RUNNER[@]}" compile "$FIXTURES/selective_negative/main.sio" -o "$negative_elf" >"$negative_log" 2>&1
negative_rc=$?
set -e
[[ "$negative_rc" -ne 0 ]] || fail selective_negative_unexpected_accept
[[ ! -e "$negative_elf" ]] || fail selective_negative_left_final_elf
[[ "$(grep -Fc 'error[E137]' "$negative_log" || true)" -eq 1 ]] || fail selective_negative_e137_count
grep -Fq 'reason=not_selected' "$negative_log" || fail selective_negative_reason_missing
grep -Fq 'MADAROS_IMPORT_AUTHORITY_REFUSAL schema=1 global_unique_fallback=0 errors=1' "$negative_log" ||
  fail selective_negative_refusal_receipt_missing

compile_case qualified "$FIXTURES/qualified/main.sio"
grep -Eq 'qualified_rewrites=[1-9][0-9]*' "$WORK/qualified.compile.log" || fail qualified_rewrite_receipt_missing
run_exit_42 qualified

compile_case duplicate_left_first "$FIXTURES/duplicate_order/left_first.sio"
run_exit_42 duplicate_left_first
compile_case duplicate_right_first "$FIXTURES/duplicate_order/right_first.sio"
run_exit_42 duplicate_right_first

printf 'MADAROS_IMPORT_CALL_AUTHORITY_RUNTIME_PASS compiler_provenance=source-fresh compiler_sha256=%s source_sha256=%s selective_negative=E137+no-elf qualified=42 duplicate_left_first=42 duplicate_right_first=42 closure_order=both\n' \
  "$compiler_sha256" "$source_sha256"
