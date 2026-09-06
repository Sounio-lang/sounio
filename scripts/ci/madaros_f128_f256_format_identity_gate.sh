#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

STRUCTURAL_ONLY=0
if [[ "${1:-}" == "--structural-only" ]]; then
  STRUCTURAL_ONLY=1
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--structural-only]" >&2
  exit 64
fi

if [[ "$STRUCTURAL_ONLY" -eq 0 && -z "${SOUNIO_F128_COMPILER:-}" ]]; then
  echo "BLOCKED manual gate requires SOUNIO_F128_COMPILER=/path/to/source-fresh-madaros-elf" >&2
  exit 2
fi
COMPILER=""
COMPILER_CLI="$ROOT_DIR/bin/madaros"
if [[ "$STRUCTURAL_ONLY" -eq 0 ]]; then
  COMPILER="$(realpath "$SOUNIO_F128_COMPILER")"
fi
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"
SOURCE_HEAD="$(git rev-parse HEAD)"
COMPILER_SOURCE_SHA="${SOUNIO_F128_COMPILER_SOURCE_SHA:-}"

for binary in "$SEED_COMPILER" ${COMPILER:+"$COMPILER"}; do
  if [[ ! -x "$binary" ]]; then
    echo "FAIL compiler is not executable: $binary" >&2
    exit 2
  fi
  if [[ "$(head -c2 "$binary" 2>/dev/null)" == '#!' ]]; then
    echo "FAIL gate requires a resolved ELF, not a wrapper: $binary" >&2
    exit 2
  fi
done
if [[ "$STRUCTURAL_ONLY" -eq 0 && ! -x "$COMPILER_CLI" ]]; then
  echo "FAIL canonical Madaros CLI is not executable: $COMPILER_CLI" >&2
  exit 2
fi
if [[ "$STRUCTURAL_ONLY" -eq 0 && -z "$COMPILER_SOURCE_SHA" ]]; then
  echo "BLOCKED set SOUNIO_F128_COMPILER_SOURCE_SHA to the source SHA used to build the Madaros ELF" >&2
  exit 2
fi
if [[ "$STRUCTURAL_ONLY" -eq 0 && "$COMPILER_SOURCE_SHA" != "$SOURCE_HEAD" ]]; then
  echo "FAIL compiler/source pin mismatch: compiler=$COMPILER_SOURCE_SHA source=$SOURCE_HEAD" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ "$STRUCTURAL_ONLY" -eq 0 ]]; then
  echo "compiler_elf=$COMPILER"
  echo "compiler_sha256=$(sha256sum "$COMPILER" | awk '{print $1}')"
  echo "compiler_source_sha=$COMPILER_SOURCE_SHA"
  echo "compiler_cli=$COMPILER_CLI"
fi
echo "seed_elf=$SEED_COMPILER"
echo "seed_sha256=$(sha256sum "$SEED_COMPILER" | awk '{print $1}')"
echo "source_head=$SOURCE_HEAD"
if [[ "$STRUCTURAL_ONLY" -eq 1 ]]; then
  echo "gate_mode=structural_only"
else
  echo "gate_mode=manual_source_fresh_evidence"
fi

DESCRIPTOR="self-hosted/compiler/f128_f256_format_descriptor_probe.sio"
probe_elf="$TMP_DIR/f128-f256-identity-probe.elf"
probe_build_log="$TMP_DIR/identity-probe-build.log"
probe_run_log="$TMP_DIR/identity-probe-run.log"
if ! "$SEED_COMPILER" "$DESCRIPTOR" "$probe_elf" >"$probe_build_log" 2>&1; then
  echo "FAIL source-owned identity probe did not compile with the bootstrap seed" >&2
  cat "$probe_build_log" >&2
  exit 1
fi
chmod +x "$probe_elf"
if ! "$probe_elf" >"$probe_run_log" 2>&1; then
  echo "FAIL source-owned identity probe returned nonzero" >&2
  cat "$probe_run_log" >&2
  exit 1
fi
if ! grep -Fxq 'PASS f128_f256_format_descriptor_probe' "$probe_run_log"; then
  echo "FAIL source-owned identity probe omitted exact PASS receipt" >&2
  cat "$probe_run_log" >&2
  exit 1
fi
echo "PASS internal identity: TyRawPtr=96 TyF128=97 TyF256=98 descriptors=exact compatibility=identity-only names=distinct"

assert_structural_containment() {
  function_body() {
    local fn_name="$1"
    awk -v fn_name="$fn_name" '
      $0 ~ "^pub fn " fn_name "\\(" { capture=1 }
      capture { print }
      capture && /^}/ { exit }
    ' self-hosted/check/compat.sio
  }

  if function_body is_numeric_type | grep -Eq 'TyF128|TyF256'; then
    echo "FAIL wide floats must stay out of is_numeric_type (use wide_float_* dispatch)" >&2
    exit 1
  fi
  if function_body is_float_type | grep -Eq 'TyF128|TyF256'; then
    echo "FAIL wide floats must stay out of is_float_type (use wide_float_* dispatch)" >&2
    exit 1
  fi
  if ! function_body is_wide_float_type | grep -Fq 'TyF128'; then
    echo "FAIL is_wide_float_type omits TyF128" >&2
    exit 1
  fi
  if ! function_body is_wide_float_type | grep -Fq 'TyF256'; then
    echo "FAIL is_wide_float_type omits TyF256" >&2
    exit 1
  fi
  if ! grep -Fq 'wide_float_binary_result_type' self-hosted/check/compat.sio; then
    echo "FAIL wide_float_binary_result_type missing (V0-E.2 ops dispatch)" >&2
    exit 1
  fi
  if ! grep -Fq 'if is_wide_float_type(left) || is_wide_float_type(right)' self-hosted/check/compat.sio; then
    echo "FAIL binary operator wide-float dispatch guard missing" >&2
    exit 1
  fi
  if ! grep -Fq 'if is_wide_float_type(operand)' self-hosted/check/compat.sio; then
    echo "FAIL unary operator wide-float dispatch guard missing" >&2
    exit 1
  fi
  if [[ "$(grep -Fc 'checker_report_error_at_inplace(c, te.span, 249' self-hosted/check/check.sio)" -ne 0 ]]; then
    echo "FAIL in-place TypeExpr still emits live E249 after V0-B lift" >&2
    exit 1
  fi
  if [[ "$(grep -Fc 'c.report_error_at(te.span, 249' self-hosted/check/check.sio)" -ne 0 ]]; then
    echo "FAIL by-value TypeExpr still emits live E249 after V0-B lift" >&2
    exit 1
  fi
  if [[ "$(grep -Fc 'parser_reject_reserved_wide_float_path(' self-hosted/parser/types.sio)" -ne 3 ]]; then
    echo "FAIL parser TypeNamed constructors do not keep reject-helper call sites" >&2
    exit 1
  fi
  if ! grep -Fq 'fn parser_reject_reserved_wide_float_path' self-hosted/parser/types.sio; then
    echo "FAIL parser_reject_reserved_wide_float_path helper missing" >&2
    exit 1
  fi
  for exact in \
    'TypeKind::TyF128 {' \
    'TypeKind::TyF256 {' \
    '// "f128"' \
    '// "f256"'; do
    if ! grep -Fq "$exact" self-hosted/check/check.sio; then
      echo "FAIL distinct diagnostic/mangle identity branch missing: $exact" >&2
      exit 1
    fi
  done
  # Full-line comments may discuss the boundary without implementing a carrier.
  # Score only source-bearing hits; otherwise a documentation improvement makes
  # this gate claim the IR schema changed.
  if rg -n '\b(f128|f256)\b|TyF128|TyF256' \
      self-hosted/ir self-hosted/native self-hosted/main.sio self-hosted/compiler/main.sio \
      2>&1 | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//' \
      >"$TMP_DIR/carrier-leak.log"; then
    echo "FAIL V0-A introduced a wide value carrier below the checker" >&2
    cat "$TMP_DIR/carrier-leak.log" >&2
    exit 1
  fi
  echo "PASS structural containment remains meaningful after commit"
}

assert_structural_containment
if [[ "$STRUCTURAL_ONLY" -eq 1 ]]; then
  echo "PASS madaros_f128_f256_format_identity_gate structural-only"
  echo "source_fresh_E249=not_run"
  exit 0
fi

RESERVED_MESSAGE='f128/f256 is reserved for compiler-owned format identity; source values are unavailable in V0-A'

run_compiler() {
  MADAROS_RAW_BIN="$COMPILER" "$COMPILER_CLI" "$@"
}

assert_check_reserved() {
  local source="$1"
  local label="$2"
  local log="$TMP_DIR/$label.check.log"
  if run_compiler check "$source" >"$log" 2>&1; then
    echo "FAIL $label unexpectedly type-checked" >&2
    cat "$log" >&2
    exit 1
  fi
  if ! grep -Fq 'error[E249' "$log" || ! grep -Fq "$RESERVED_MESSAGE" "$log"; then
    echo "FAIL $label did not emit stable E249" >&2
    cat "$log" >&2
    exit 1
  fi
  echo "PASS $label rejected as fresh source with E249"
}

assert_check_reserved tests/compile-fail/f128_f256_source_signature_reserved.sio signature
assert_check_reserved tests/compile-fail/f128_struct_field_reserved.sio struct_field
assert_check_reserved tests/compile-fail/f256_enum_field_reserved.sio enum_field
assert_check_reserved tests/compile-fail/f128_type_alias_reserved.sio type_alias
assert_check_reserved tests/compile-fail/f128_global_let_reserved.sio global_let
assert_check_reserved tests/compile-fail/f256_global_var_nested_reserved.sio nested_global_var
assert_check_reserved tests/compile-fail/f128_f256_literal_unimplemented.sio local_binding
assert_check_reserved tests/compile-fail/f128_nested_knowledge_reserved.sio nested_epistemic
assert_check_reserved tests/compile-fail/f128_unknown_generic_arg_reserved.sio unknown_generic_arg
assert_check_reserved tests/compile-fail/f128_cast_from_f64_unimplemented.sio cast_target
assert_check_reserved tests/compile-fail/f128_cast_to_f64_unimplemented.sio cast_source
assert_check_reserved tests/compile-fail/f128_f256_arithmetic_unimplemented.sio arithmetic_surface
assert_check_reserved tests/compile-fail/f256_comparison_unimplemented.sio comparison_surface
assert_check_reserved tests/compile-fail/f128_f256_implicit_conversion_unimplemented.sio conversion_surface
assert_check_reserved tests/native-v2/f128_format_identity_imported_containment.sio imported_signature

typed_global_control_log="$TMP_DIR/non-wide-typed-global.check.log"
if ! run_compiler check tests/compiler/fixtures/monolithic_public_lower_call/bss_typed_adds.sio \
    >"$typed_global_control_log" 2>&1; then
  echo "FAIL preserving global annotations regressed ordinary f64/i64 typed globals" >&2
  cat "$typed_global_control_log" >&2
  exit 1
fi
echo "PASS non-wide typed-global regression control"

assert_compile_reserved() {
  local source="$1"
  local label="$2"
  local out="$TMP_DIR/$label.elf"
  local log="$TMP_DIR/$label.compile.log"
  if run_compiler compile "$source" -o "$out" >"$log" 2>&1; then
    echo "FAIL $label unexpectedly compiled" >&2
    cat "$log" >&2
    exit 1
  fi
  if ! grep -Fq 'error[E249' "$log" || ! grep -Fq "$RESERVED_MESSAGE" "$log"; then
    echo "FAIL $label compilation did not stop at E249" >&2
    cat "$log" >&2
    exit 1
  fi
  if [[ -e "$out" ]]; then
    echo "FAIL $label produced an artifact despite E249" >&2
    ls -l "$out" >&2
    exit 1
  fi
  echo "PASS $label produced no requested native artifact after E249"
}

assert_compile_reserved tests/compile-fail/f128_f256_source_signature_reserved.sio single_module
assert_compile_reserved tests/native-v2/f128_format_identity_imported_containment.sio imported_module

render_control_path='examples/render/../../tests/selfhost/native_runtime/println_int_42.sio'
render_wide_path='examples/render/../../tests/compile-fail/f128_f256_source_signature_reserved.sio'
if [[ "$render_control_path" != examples/render/* || "$render_wide_path" != examples/render/* ]]; then
  echo "FAIL streaming probes must preserve the lexical examples/render/ prefix" >&2
  exit 1
fi
if [[ "$(realpath "$render_control_path")" != "$(realpath tests/selfhost/native_runtime/println_int_42.sio)" ]]; then
  echo "FAIL render control path does not resolve to the intended non-wide fixture" >&2
  exit 1
fi
if [[ "$(realpath "$render_wide_path")" != "$(realpath tests/compile-fail/f128_f256_source_signature_reserved.sio)" ]]; then
  echo "FAIL render wide path does not resolve to the intended reserved-wide fixture" >&2
  exit 1
fi

streaming_control_log="$TMP_DIR/streaming-render-control.log"
run_compiler --probe-native-streaming "$render_control_path" >"$streaming_control_log" 2>&1 || true
if ! grep -Fxq 'probe_native_streaming: typecheck_skipped' "$streaming_control_log"; then
  echo "FAIL non-wide render control did not prove the checker-skip branch is live" >&2
  cat "$streaming_control_log" >&2
  exit 1
fi
echo "PASS lexical render control activated probe_native_streaming: typecheck_skipped"

streaming_log="$TMP_DIR/streaming-render-wide.log"
if run_compiler --probe-native-streaming "$render_wide_path" >"$streaming_log" 2>&1; then
  echo "FAIL render streaming probe accepted reserved wide source" >&2
  cat "$streaming_log" >&2
  exit 1
fi
if ! grep -Fq 'error[E249' "$streaming_log" || ! grep -Fq "$RESERVED_MESSAGE" "$streaming_log"; then
  echo "FAIL render streaming probe did not stop at parser E249" >&2
  cat "$streaming_log" >&2
  exit 1
fi
for forbidden_stage in \
  'probe_native_streaming: typecheck_begin' \
  'probe_native_streaming: typecheck_skipped' \
  'probe_native_streaming: summary_begin' \
  'probe_native_streaming: body_begin'; do
  if grep -Fq "$forbidden_stage" "$streaming_log"; then
    echo "FAIL reserved-wide render probe reached forbidden post-parser stage: $forbidden_stage" >&2
    cat "$streaming_log" >&2
    exit 1
  fi
done
echo "PASS live render skip-check route stopped at parser E249 before typecheck/summary/body"

echo "PASS madaros_f128_f256_format_identity_gate"
echo "claim=internal_format_identity_plus_fresh_source_canonical_cli_containment"
echo "source_values=parser_E249_with_checker_defense_on_canonical_cli_paths"
echo "ir_schema=unchanged"
echo "value_transport=not_implemented"
echo "ci_enforcement=madaros_witness_gate"
echo "future_preparsed_cache=out_of_scope_must_version_or_revalidate"
echo "internal_direct_load_lower=not_claimed_must_check_parser_errors"
