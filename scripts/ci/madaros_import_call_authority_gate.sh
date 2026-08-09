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

has_forbidden_path() {
  grep -Eq \
    'native_prebundle:|falling back to full IR path|compact modular IR table path|legacy compact IR differential enabled|SELFHOST=fallback|driver_orchestration.*status=fallback' \
    "$1"
}

has_fatal_log() {
  grep -Eiq \
    'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' \
    "$1"
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
  "$FIXTURES/qualified_shadow/main.sio" \
  "$FIXTURES/qualified_shadow/leaf.sio" \
  "$FIXTURES/glob_multi/main.sio" \
  "$FIXTURES/glob_multi/leaf.sio" \
  "$FIXTURES/duplicate_order/left.sio" \
  "$FIXTURES/duplicate_order/right.sio" \
  "$FIXTURES/duplicate_order/left_first.sio" \
  "$FIXTURES/duplicate_order/right_first.sio"; do
  [[ -f "$path" ]] || fail "missing_${path#"$ROOT_DIR"/}"
done

binding_shape="$(sed -n '/^pub struct LowerExternBindingHandle {/,/^}/p' "$LOWER")"
collector_shape="$(sed -n '/^pub fn module_frontend_collect_ast_closure_programs_into(/,/^\/\/ Validate that the collection generation/p' "$FRONTEND")"
authority_shape="$(sed -n '/^fn module_frontend_prepare_import_authority(/,/^fn module_frontend_assign_function_provenance(/p' "$FRONTEND")"
stub_shape="$(sed -n '/^fn module_frontend_prepend_used_checker_stubs(/,/^fn module_frontend_drop_item_prefix(/p' "$FRONTEND")"
strip_shape="$(sed -n '/^fn module_frontend_strip_import_authority_stubs(/,/^fn module_frontend_prepare_import_authority(/p' "$FRONTEND")"
compile_shape="$(sed -n '/^pub fn module_frontend_compile_collected_to_file(/,/^pub fn module_frontend_imported_native_compile(/p' "$FRONTEND")"
lower_array_shape="$(sed -n '/^fn module_frontend_lower_programs_array_direct_box(/,/^fn module_frontend_lower_programs_array_boxed(/p' "$FRONTEND")"
lower_binding_shape="$(sed -n '/^fn lowerer_extern_binding_index(/,/^fn lowerer_preseed_external_impl_method_items_mut(/p' "$LOWER")"

for field in caller_module_id local_name defining_module_id export_name qualified_name qualifier_path; do
  grep -Fq "pub $field:" <<<"$binding_shape" || fail "binding_field_${field}_missing"
done
grep -Fq 'MF_IMPORT_AUTHORITY_PREPARED_COLLECTION_ID = (*out).collection_id' <<<"$collector_shape" ||
  fail collector_authority_receipt_missing
grep -Fq 'module_frontend_publish_program_item_handle(0, &(*root_program))' <<<"$collector_shape" ||
  fail collector_root_item_handle_publish_missing
[[ "$(grep -Fc 'module_frontend_publish_program_item_handle(' <<<"$collector_shape")" -ge 2 ]] ||
  fail collector_dependency_item_handle_publish_missing
grep -Fq 'module_frontend_imports_from_program_item_handle(work_front)' <<<"$collector_shape" ||
  fail collector_import_scan_not_handle_owned
grep -Fq 'module_frontend_program_item_handles_available((*out).node_count)' <<<"$collector_shape" ||
  fail collector_item_handle_completeness_guard_missing
grep -Fq 'var MF_CHECKER_ITEM_PTRS: [i64; 256]' "$FRONTEND" ||
  fail checker_item_handle_table_missing
grep -Fq '(*checker_handle).items = rewritten_items' <<<"$authority_shape" ||
  fail checker_rewrite_view_not_pinned
grep -Fq '(*programs)[caller_module_id as usize].items = (*checker_handle).items' <<<"$authority_shape" ||
  fail checker_program_view_not_field_scoped
if grep -Fq '(*caller_handle).items = rewritten_items' <<<"$authority_shape"; then
  fail authored_item_handle_rewritten
fi
if grep -Fq 'module_frontend_publish_program_item_handle(' <<<"$strip_shape"; then
  fail checker_stub_strip_overwrites_authored_handle
fi
grep -Fq 'let seed_item_handle = MF_PROGRAM_ITEM_PTRS[0] as *mut LowerProgramItemsHandle' "$FRONTEND" ||
  fail seed_lowering_item_handle_missing
grep -Fq 'let dep_item_handle = MF_PROGRAM_ITEM_PTRS[i as usize] as *mut LowerProgramItemsHandle' "$FRONTEND" ||
  fail dependency_lowering_item_handle_missing
for authoritative_shape in "$collector_shape" "$authority_shape" "$lower_array_shape"; do
  if grep -Fq 'module_frontend_snapshot_program_item_handles(' <<<"$authoritative_shape"; then
    fail authoritative_program_resnapshot_reintroduced
  fi
done
grep -Fq 'module_frontend_prepare_import_authority(' <<<"$collector_shape" ||
  fail collector_authority_prepare_missing
grep -Fq 'module_frontend_authority_rewrite_item_list_opt(' <<<"$authority_shape" ||
  fail authority_ast_rewrite_missing
grep -Fq '(*callee).kind == ExprKind::ExprPath' "$FRONTEND" || fail qualified_path_classifier_missing
grep -Fq '(*callee).kind = ExprKind::ExprIdent' "$FRONTEND" || fail qualified_path_rewrite_missing
grep -Fq '(*callee).name = make_name(MF_EXTERN_BINDING_LOCAL_NAMES[binding_index as usize])' "$FRONTEND" ||
  fail qualified_path_exact_local_name_missing
if grep -Fq '(*stage).e.left = Some(Box::new(rewritten))' "$FRONTEND"; then
  fail qualified_path_reboxing_reintroduced
fi
grep -Fq 'MF_EXTERN_BINDING_QUALIFIER_PATHS[qualifier_sidecar_index as usize] = requested_import' "$FRONTEND" ||
  fail qualifier_text_not_derived_from_authored_import
grep -Fq 'MF_EXTERN_BINDING_QUALIFIED_NAMES[qualifier_sidecar_index as usize] = str_from_bytes(qualified_name.buf, qualified_name.len)' "$FRONTEND" ||
  fail qualified_lower_name_not_derived_from_authored_import
for sidecar in \
  MF_EXTERN_BINDING_LOCAL_NAMES \
  MF_EXTERN_BINDING_EXPORT_NAMES \
  MF_EXTERN_BINDING_QUALIFIER_PATHS \
  MF_EXTERN_BINDING_QUALIFIED_NAMES; do
  grep -Fq "var $sidecar: [string; 2048]" "$FRONTEND" || fail "binding_sidecar_${sidecar}_missing"
done
grep -Fq 'fn module_frontend_call_text_selects_binding(' "$FRONTEND" || fail qualified_exact_path_selection_missing
grep -Fq 'ir_name_eq((*binding).local_name, local_name)' "$FRONTEND" || fail bare_exact_local_name_selection_missing
grep -Fq 'call_len != qualifier_len + 2 + export_len' "$FRONTEND" || fail qualified_exact_path_length_missing
grep -Fq 'str_char_at(call_path, qualifier_len) != 58' "$FRONTEND" || fail qualified_separator_comparison_missing
grep -Fq 'module_frontend_prepend_used_checker_stubs(' <<<"$authority_shape" ||
  fail checker_stub_adapter_missing
grep -Fq 'visibility_is_pub((*item).visibility)' "$FRONTEND" || fail checker_stub_public_guard_missing
grep -Fq '(*programs)[caller_module_id as usize].items = Some(Box::new(ItemList {' <<<"$stub_shape" ||
  fail checker_stub_prepend_not_field_scoped
grep -Fq '(*programs)[i as usize].items = module_frontend_drop_item_prefix(items, count)' <<<"$strip_shape" ||
  fail checker_stub_strip_not_field_scoped
if grep -Eq 'var (program|caller) = \(\*programs\)' <<<"${authority_shape}${stub_shape}${strip_shape}"; then
  fail authority_whole_program_write_reintroduced
fi
grep -Fq 'MADAROS_IMPORT_AUTHORITY_REFUSAL schema=1 global_unique_fallback=0' <<<"$authority_shape" ||
  fail refusal_receipt_missing
grep -Fq 'MADAROS_IMPORT_AUTHORITY_RECEIPT schema=1 modules=' <<<"$authority_shape" ||
  fail acceptance_receipt_missing
grep -Fq 'fn module_frontend_rebind_qualified_calls_to_external_identity(' "$FRONTEND" ||
  fail qualified_ir_identity_rebind_missing
grep -Fq 'ir_merge_find_function_identity_index(' "$FRONTEND" ||
  fail qualified_ir_exact_identity_lookup_missing
grep -Fq 'MADAROS_IMPORT_AUTHORITY_IR_REBIND_RECEIPT schema=1 caller=' "$FRONTEND" ||
  fail qualified_ir_rebind_receipt_missing
grep -Fq 'fn module_frontend_rebind_merged_calls_for_caller(' "$FRONTEND" ||
  fail post_merge_caller_scoped_rebind_missing
grep -Fq 'func.defining_module_id == caller_module_id' "$FRONTEND" ||
  fail post_merge_caller_identity_guard_missing
grep -Fq 'func.instrs[ii as usize].fn_id < 0' "$FRONTEND" ||
  fail post_merge_unresolved_call_guard_missing
grep -Fq '(*binding).export_name,' "$FRONTEND" ||
  fail post_merge_export_identity_lookup_missing
grep -Fq 'let post_merge_rebind_count = module_frontend_rebind_merged_import_calls(' <<<"$lower_array_shape" ||
  fail post_merge_rebind_not_wired_after_merge
grep -Fq 'post_merge_import_call_identity_rebind_failed' <<<"$lower_array_shape" ||
  fail post_merge_rebind_not_fail_closed
grep -Fq 'MADAROS_IMPORT_AUTHORITY_POST_MERGE_REBIND_TOTAL schema=1 callers=' "$FRONTEND" ||
  fail post_merge_rebind_total_receipt_missing
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
grep -Fq 'answer() + leaf::answer()' "$FIXTURES/qualified_shadow/main.sio" ||
  fail qualified_shadow_fixture_calls
grep -Fq 'fn answer() -> i64' "$FIXTURES/qualified_shadow/main.sio" ||
  fail qualified_shadow_local_definition
grep -Fq 'use leaf::*' "$FIXTURES/glob_multi/main.sio" || fail glob_fixture_import
grep -Fq 'second()' "$FIXTURES/glob_multi/main.sio" || fail glob_fixture_second_call
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

printf 'MADAROS_IMPORT_CALL_AUTHORITY_SOURCE_PASS binding=caller+local+defining+export+qualifier checker_adapter=collection-owned lowering=explicit-only+post-merge-exact global_unique_fallback=absent fixtures=selective-reject,qualified-call,qualified-shadow,duplicate-order-x2,glob-multi source_sha256=%s\n' "$source_sha256"

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

INFO_LOG="$WORK/wrapper.info"
if ! env \
    -u SOUNIO_MADAROS_BIN \
    -u SOUNIO_SOUC_BIN \
    "MADAROS_RAW_BIN=$RAW_MADAROS" \
    "$WRAPPER" info >"$INFO_LOG" 2>&1; then
  cat "$INFO_LOG" >&2 || true
  fail wrapper_info_failed
fi
grep -Fxq "raw_elf:      $RAW_MADAROS" "$INFO_LOG" || fail wrapper_raw_identity_mismatch

RUNNER=(env \
  -u SOUNIO_MADAROS_BIN \
  -u SOUNIO_SOUC_BIN \
  "MADAROS_RAW_BIN=$RAW_MADAROS" \
  "SOUNIO_STDLIB_PATH=$ROOT_DIR/stdlib" \
  SOUNIO_SOUC_ENGINE=madaros \
  SOUNIO_ENABLE_COMPACT_IMPORTED_IR=0 \
  OMEGA_SOUC_ALLOW_LOCAL_FALLBACK=0 \
  "$WRAPPER" \
  --science-boundary off)

compile_case() {
  local label="$1"
  local source="$2"
  local elf="$WORK/$label.elf"
  local log="$WORK/$label.compile.log"

  rm -f "$elf"
  if ! timeout --signal=TERM --kill-after=10s "$TIMEOUT_SECONDS" \
      "${RUNNER[@]}" compile "$source" -o "$elf" >"$log" 2>&1; then
    cat "$log" >&2 || true
    fail "${label}_compile_failed"
  fi
  has_fatal_log "$log" && fail "${label}_compiler_crash"
  has_forbidden_path "$log" && fail "${label}_compact_or_fallback_path"
  if grep -Fq 'error[E' "$log" || grep -Eq '^error:' "$log"; then
    cat "$log" >&2 || true
    fail "${label}_diagnostic_on_success"
  fi
  [[ -s "$elf" && -x "$elf" ]] || fail "${label}_elf_missing_or_not_executable"
  [[ "$(od -An -tx1 -N4 "$elf" | tr -d ' \n')" == 7f454c46 ]] || fail "${label}_output_not_elf"
  grep -Fq 'MADAROS_IMPORT_AUTHORITY_RECEIPT schema=1' "$log" || fail "${label}_authority_receipt_missing"
  grep -Fq 'global_unique_fallback=0' "$log" || fail "${label}_fallback_receipt_missing"
  grep -Fq 'MADAROS_IMPORT_AUTHORITY_IR_REBIND_RECEIPT schema=1' "$log" || fail "${label}_ir_rebind_receipt_missing"
  grep -Fq 'MADAROS_IMPORT_AUTHORITY_POST_MERGE_REBIND_TOTAL schema=1' "$log" || fail "${label}_post_merge_rebind_receipt_missing"
  grep -Eq '^Merged IR: [1-9][0-9]*$' "$log" || fail "${label}_merged_ir_receipt_missing"
}

run_exit_code() {
  local label="$1"
  local expected="$2"
  local elf="$WORK/$label.elf"
  local rc=0

  set +e
  timeout --signal=TERM --kill-after=5s 30s "$elf" >"$WORK/$label.stdout" 2>"$WORK/$label.stderr"
  rc=$?
  set -e
  [[ "$rc" -eq "$expected" ]] || {
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
[[ "$negative_rc" -eq 1 ]] || fail "selective_negative_unexpected_rc_${negative_rc}"
[[ ! -e "$negative_elf" ]] || fail selective_negative_left_final_elf
has_fatal_log "$negative_log" && fail selective_negative_compiler_crash
has_forbidden_path "$negative_log" && fail selective_negative_compact_or_fallback_path
[[ "$(grep -Fc 'error[E137]' "$negative_log" || true)" -eq 1 ]] || fail selective_negative_e137_count
grep -Fq 'reason=not_selected' "$negative_log" || fail selective_negative_reason_missing
grep -Fq 'MADAROS_IMPORT_AUTHORITY_REFUSAL schema=1 global_unique_fallback=0 errors=1' "$negative_log" ||
  fail selective_negative_refusal_receipt_missing
if grep -Eq 'imported_compile: lower_begin|lower_array:|Merged IR:|Compilation successful!' "$negative_log"; then
  fail selective_negative_reached_lowering
fi

compile_case qualified "$FIXTURES/qualified/main.sio"
grep -Eq 'qualified_rewrites=[1-9][0-9]*' "$WORK/qualified.compile.log" || fail qualified_rewrite_receipt_missing
grep -Eq '^MADAROS_IMPORT_AUTHORITY_POST_MERGE_REBIND_RECEIPT schema=1 caller=0 bindings=[1-9][0-9]* rewrites=1 global_unique_fallback=0$' "$WORK/qualified.compile.log" ||
  fail qualified_ir_rewrite_count
run_exit_code qualified 42

compile_case qualified_shadow "$FIXTURES/qualified_shadow/main.sio"
grep -Eq 'qualified_rewrites=[1-9][0-9]*' "$WORK/qualified_shadow.compile.log" ||
  fail qualified_shadow_rewrite_receipt_missing
grep -Eq '^MADAROS_IMPORT_AUTHORITY_POST_MERGE_REBIND_RECEIPT schema=1 caller=0 bindings=[1-9][0-9]* rewrites=1 global_unique_fallback=0$' "$WORK/qualified_shadow.compile.log" ||
  fail qualified_shadow_ir_rewrite_count
run_exit_code qualified_shadow 44

compile_case duplicate_left_first "$FIXTURES/duplicate_order/left_first.sio"
run_exit_code duplicate_left_first 42
compile_case duplicate_right_first "$FIXTURES/duplicate_order/right_first.sio"
run_exit_code duplicate_right_first 42
compile_case glob_multi "$FIXTURES/glob_multi/main.sio"
run_exit_code glob_multi 42

final_compiler_sha256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
[[ "$final_compiler_sha256" == "$compiler_sha256" ]] || fail compiler_changed_during_gate
printf 'MADAROS_IMPORT_CALL_AUTHORITY_RUNTIME_PASS compiler_provenance=source-fresh compiler_sha256=%s source_sha256=%s selective_negative=E137+no-elf+pre-lowering qualified=42 qualified_shadow=44+ir-rebind1 duplicate_left_first=42 duplicate_right_first=42 glob_multi=42 closure_order=both compact=disabled fallback=none\n' \
  "$compiler_sha256" "$source_sha256"
