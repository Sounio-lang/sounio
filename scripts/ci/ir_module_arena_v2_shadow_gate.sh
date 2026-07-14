#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOURCE="self-hosted/ir/arena_v2_shadow.sio"
WITNESS="tests/native-v2/ir_module_arena_v2_shadow_witness.sio"
BASE="${IR_MODULE_ARENA_V2_BASE_SHA:-17b0858f6e7d75c9cfc9e545b1b9f0805fa9d5d6}"
SOUC="${SOUC_BIN:-$ROOT/bin/souc}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-ir-module-arena-v2.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

fail() {
  printf 'IR_MODULE_ARENA_V2_SHADOW_FAIL reason=%s\n' "$1" >&2
  exit 1
}

[[ -f "$SOURCE" ]] || fail source_missing
[[ -f "$WITNESS" ]] || fail witness_missing
[[ -x "$SOUC" ]] || fail compiler_missing

if grep -Eq '^[[:space:]]*use (native::runtime_context|ir::heap_storage|compiler::lean_single)' "$SOURCE" ||
   grep -Eq '(^|[^[:alnum:]_])(heap_alloc|heap_free)[[:space:]]*\(' "$SOURCE"; then
  fail forbidden_storage_dependency
fi
if grep -Eq ':[[:space:]]*(IrModule|IrFunction|IrInstr)([[:space:]],}]|$)' "$SOURCE"; then
  fail aggregate_payload_storage_detected
fi

grep -Fq 'pub struct IrModuleArenaV2FunctionHandle' "$SOURCE" || fail typed_function_handle_missing
grep -Fq 'pub struct IrModuleArenaV2InstrHandle' "$SOURCE" || fail typed_instr_handle_missing
grep -Fq 'function_generation: [i64; 4]' "$SOURCE" || fail function_generation_missing
grep -Fq 'instr_generation: [i64; 8]' "$SOURCE" || fail instr_generation_missing
grep -Fq 'fn ir_module_arena_v2_set_function_reg_count_validated' "$SOURCE" || fail private_function_setter_missing
grep -Fq 'fn ir_module_arena_v2_set_instr_payload_validated' "$SOURCE" || fail private_instr_setter_missing
if grep -Eq '^pub fn ir_module_arena_v2_set_' "$SOURCE"; then
  fail raw_setter_is_public
fi

for default_surface in self-hosted/ir/mod.sio self-hosted/compiler/main.sio scripts/bootstrap/bootstrap_concat.sh; do
  if grep -Fq 'arena_v2_shadow' "$default_surface"; then
    fail "shadow_imported_by_${default_surface//\//_}"
  fi
done

if git cat-file -e "$BASE^{commit}" 2>/dev/null; then
  if ! git diff --quiet "$BASE" -- \
      self-hosted/ir/heap_storage.sio \
      self-hosted/ir/lower.sio \
      self-hosted/native/codegen_x86_linux.sio \
      self-hosted/native/runtime_context.sio \
      self-hosted/ir/mod.sio \
      self-hosted/compiler/main.sio \
      scripts/bootstrap/bootstrap_concat.sh; then
    fail legacy_or_default_surface_changed
  fi
else
  fail base_sha_unavailable
fi

"$SOUC" check "$SOURCE" >"$TMP/source-check.log" 2>&1 || {
  cat "$TMP/source-check.log" >&2
  fail source_check
}

# The shadow is deliberately absent from the compiler's canonical module
# bundle. Compose only inside this throwaway gate directory so checker/runtime
# evidence does not require a default-pipeline import.
COMPOSITE="$TMP/ir_module_arena_v2_shadow_composite.sio"
sed '/^module ir::arena_v2_shadow$/d' "$SOURCE" >"$COMPOSITE"
sed '/^use ir::arena_v2_shadow::\*$/d' "$WITNESS" >>"$COMPOSITE"

"$SOUC" check "$COMPOSITE" >"$TMP/composite-check.log" 2>&1 || {
  cat "$TMP/composite-check.log" >&2
  fail composite_check
}

set +e
"$SOUC" run "$COMPOSITE" >"$TMP/witness.out" 2>"$TMP/witness.err"
run_rc=$?
set -e
if [[ "$run_rc" -ne 0 ]]; then
  cat "$TMP/witness.out" >&2
  cat "$TMP/witness.err" >&2
  fail "witness_rc_$run_rc"
fi
grep -Fxq 'IR_MODULE_ARENA_V2_SHADOW_PASS' "$TMP/witness.out" || {
  cat "$TMP/witness.out" >&2
  fail witness_pass_marker_missing
}
if grep -Fq 'IR_MODULE_ARENA_V2_SHADOW_FAIL' "$TMP/witness.out"; then
  cat "$TMP/witness.out" >&2
  fail contradictory_witness_receipt
fi

compiler_sha256="$(sha256sum "$SOUC" | awk '{print $1}')"
printf '%s\n' 'IR_MODULE_ARENA_V2_SHADOW_CHECK source=pass composite=pass'
printf '%s\n' 'IR_MODULE_ARENA_V2_SHADOW_PROBE aba_function=pass aba_instr=pass cross_arena=pass cross_module=pass capacity=pass setter_bounds=pass stale_after_release=pass double_release=pass'
printf '%s\n' 'IR_MODULE_ARENA_V2_SHADOW_DIFF legacy_surfaces=unchanged default_pipeline=unwired'
printf '%s\n' 'IR_MODULE_ARENA_V2_SHADOW_SCOPE backing=bounded_scalar_tables authority=arena state=free_live_released handles=typed_generation_checked mutation=validated_scalar_only'
printf '%s\n' 'IR_MODULE_ARENA_V2_SHADOW_ORACLE legacy=self-hosted/ir/heap_storage.sio mode=differential kept=yes replaced=no'
printf '%s\n' 'IR_MODULE_ARENA_V2_SHADOW_NOT_SOLVED issue_877=compile_time_noncopyable_authority issue_884=runtime_managed_handle_lifecycle'
printf 'IR_MODULE_ARENA_V2_SHADOW_PASS compiler=%s compiler_sha256=%s base=%s\n' "$SOUC" "$compiler_sha256" "$BASE"
