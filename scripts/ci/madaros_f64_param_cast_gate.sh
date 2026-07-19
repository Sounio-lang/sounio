#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOWER="$ROOT_DIR/self-hosted/ir/lower.sio"
PARAM_SOURCE="$ROOT_DIR/tests/regression/f64_param_cast_numeric_truncation.sio"
PARAM_EXPECTED="$ROOT_DIR/tests/regression/f64_param_cast_numeric_truncation.stdout"
CONTROL_SOURCE="$ROOT_DIR/tests/regression/f64_param_cast_controls.sio"
CONTROL_EXPECTED="$ROOT_DIR/tests/regression/f64_param_cast_controls.stdout"
RAW_COMPILER="${SOUNIO_MADAROS_F64_PARAM_CAST_GATE_BIN:-}"
EXPECTED_COMPILER_SHA256="${SOUNIO_MADAROS_F64_PARAM_CAST_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_MADAROS_F64_PARAM_CAST_GATE_KEEP:-0}"
MODE="source-only"

fail() {
  printf 'MADAROS_F64_PARAM_CAST_FAIL reason=%s\n' "$1" >&2
  exit 1
}

usage() {
  cat <<'EOF'
usage: scripts/ci/madaros_f64_param_cast_gate.sh [--source-only|--classify-default|--source-fresh]

  --source-only       Verify the mutation-safe source shape and fixture coverage.
  --classify-default  Run the checkout's default compiler as stale/default diagnostic evidence.
  --source-fresh      Require an explicit raw Madaros ELF and exact SHA-256, then enforce runtime output.

Source-fresh mode requires:
  SOUNIO_MADAROS_F64_PARAM_CAST_GATE_BIN=/path/to/madaros
  SOUNIO_MADAROS_F64_PARAM_CAST_EXPECTED_SHA256=<64 lowercase hex characters>
EOF
}

case "${1:-}" in
  ""|--source-only) MODE="source-only" ;;
  --classify-default) MODE="classify-default" ;;
  --source-fresh) MODE="source-fresh" ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; fail unexpected_argument ;;
esac
[[ $# -le 1 ]] || fail unexpected_argument

for path in "$LOWER" "$PARAM_SOURCE" "$PARAM_EXPECTED" "$CONTROL_SOURCE" "$CONTROL_EXPECTED"; do
  [[ -f "$path" ]] || fail "missing_${path#"$ROOT_DIR"/}"
done

helper_body="$(sed -n '/^fn lowerer_mark_local_scalar_kind_mut(/,/^fn lowerer_bind_local_fixed_array_len_mut(/p' "$LOWER")"
param_body="$(sed -n '/^fn lowerer_lower_fn_params_mut(/,/^fn lowerer_lower_fn_item_mut(/p' "$LOWER")"

[[ "$(grep -Fc 'fn lowerer_mark_local_scalar_kind_mut(' "$LOWER")" -eq 1 ]] ||
  fail mutation_helper_count
grep -Fq 'var locals_box = (*lo).locals' <<<"$helper_body" || fail mutation_helper_box_extract
grep -Fq '(*locals_box).scalar_kind[i as usize] = kind' <<<"$helper_body" || fail mutation_helper_scalar_store
grep -Fq '(*lo).locals = locals_box' <<<"$helper_body" || fail mutation_helper_box_writeback

expected_call='lowerer_mark_local_scalar_kind_mut(lo, (*list).head.name, 2)'
[[ "$(grep -Fc "$expected_call" <<<"$param_body")" -eq 1 ]] || fail parameter_mutation_call_count
if grep -Eq '^[[:space:]]*\(\*lo\)[[:space:]]*=[[:space:]]*\(\*lo\)\.bind_local_scalar_kind' <<<"$param_body"; then
  fail parameter_by_value_rmw_present
fi

for anchor in \
  'fn direct_param_cast(x: f64) -> i64' \
  'fn copied_param_cast(x: f64) -> i64' \
  'fn second_position_param_cast(first: i64, x: f64) -> i64' \
  'fn negative_param_cast(x: f64) -> i64'; do
  grep -Fq "$anchor" "$PARAM_SOURCE" || fail "missing_parameter_fixture_${anchor//[^a-zA-Z0-9]/_}"
done
[[ "$(grep -Fc 'return x as i64' "$PARAM_SOURCE")" -eq 3 ]] || fail parameter_direct_cast_count
[[ "$(grep -Fc 'let copied = x' "$PARAM_SOURCE")" -eq 1 ]] || fail parameter_copy_count
[[ "$(grep -Fc 'return copied as i64' "$PARAM_SOURCE")" -eq 1 ]] || fail parameter_copied_cast_count
for anchor in \
  'fn local_f64_cast() -> i64' \
  'fn arithmetic_f64_cast(x: f64) -> i64' \
  'fn int_identity_cast(x: i64) -> i64'; do
  grep -Fq "$anchor" "$CONTROL_SOURCE" || fail "missing_control_fixture_${anchor//[^a-zA-Z0-9]/_}"
done
grep -Fq 'let local: f64 = 6.75' "$CONTROL_SOURCE" || fail local_control_value
grep -Fq 'return local as i64' "$CONTROL_SOURCE" || fail local_control_cast
grep -Fq 'return (x + 0.0) as i64' "$CONTROL_SOURCE" || fail arithmetic_control_cast
grep -Fq 'return x as i64' "$CONTROL_SOURCE" || fail int_control_cast

[[ "$(cat "$PARAM_EXPECTED")" == $'direct=4\ncopied=9\nsecond=3\nnegative=-2\nF64_PARAM_CAST_NUMERIC_TRUNCATION_PASS' ]] ||
  fail parameter_expected_output
[[ "$(cat "$CONTROL_EXPECTED")" == $'local=6\narithmetic=8\nint=-11\nF64_PARAM_CAST_CONTROLS_PASS' ]] ||
  fail control_expected_output

printf 'MADAROS_F64_PARAM_CAST_SOURCE_PASS helper=existing_single_level_box_mutation parameter_rmw=removed cases=direct,copied,second-position,negative controls=local,arithmetic,int\n'

if [[ "$MODE" == "source-only" ]]; then
  exit 0
fi

if [[ -n "${SOUNIO_MADAROS_F64_PARAM_CAST_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_F64_PARAM_CAST_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail work_directory_exists
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-f64-param-cast.XXXXXX")"
fi
if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ "$MODE" == "source-fresh" ]]; then
  [[ -n "$RAW_COMPILER" ]] || fail explicit_source_fresh_compiler_required
  [[ -x "$RAW_COMPILER" ]] || fail source_fresh_compiler_not_executable
  [[ "$(od -An -tx1 -N4 "$RAW_COMPILER" | tr -d ' \n')" == "7f454c46" ]] ||
    fail source_fresh_compiler_must_be_elf
  [[ "$EXPECTED_COMPILER_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail expected_compiler_sha256_required
  compiler_sha256="$(sha256sum "$RAW_COMPILER" | awk '{print $1}')"
  [[ "$compiler_sha256" == "$EXPECTED_COMPILER_SHA256" ]] || fail source_fresh_compiler_sha256_mismatch
  RUNNER=(env "MADAROS_RAW_BIN=$RAW_COMPILER" "$ROOT_DIR/bin/madaros")
  PROVENANCE="source-fresh"
  COMPILER_SURFACE="$RAW_COMPILER"
else
  RUNNER=("$ROOT_DIR/bin/souc")
  PROVENANCE="stale/default"
  if "${RUNNER[@]}" info >"$WORK/default.info" 2>&1; then
    default_raw="$(awk '$1 == "raw_elf:" { print $2; exit }' "$WORK/default.info")"
  else
    default_raw=""
  fi
  if [[ -n "$default_raw" && -x "$default_raw" ]]; then
    COMPILER_SURFACE="$default_raw"
  else
    COMPILER_SURFACE="$ROOT_DIR/bin/souc"
  fi
  compiler_sha256="$(sha256sum "$COMPILER_SURFACE" | awk '{print $1}')"
fi

run_case() {
  local label="$1"
  local source="$2"
  local expected="$3"
  local elf="$WORK/$label.elf"
  local actual="$WORK/$label.stdout"
  local stderr="$WORK/$label.stderr"
  local rc=0

  if ! timeout --signal=TERM --kill-after=5s 30s \
      "${RUNNER[@]}" check "$source" >"$WORK/$label.check.log" 2>&1; then
    printf 'case=%s status=check-failed\n' "$label" >&2
    cat "$WORK/$label.check.log" >&2
    return 2
  fi
  if ! timeout --signal=TERM --kill-after=5s 30s \
      "${RUNNER[@]}" compile "$source" -o "$elf" >"$WORK/$label.compile.log" 2>&1; then
    printf 'case=%s status=compile-failed\n' "$label" >&2
    cat "$WORK/$label.compile.log" >&2
    return 3
  fi
  [[ -s "$elf" ]] || return 4
  chmod +x "$elf"

  if timeout --signal=TERM --kill-after=5s 15s "$elf" >"$actual" 2>"$stderr"; then
    rc=0
  else
    rc=$?
  fi
  if [[ "$rc" -ne 0 ]]; then
    printf 'case=%s status=runtime-failed rc=%s\n' "$label" "$rc" >&2
    cat "$actual" >&2 || true
    cat "$stderr" >&2 || true
    return 5
  fi
  if ! cmp -s "$expected" "$actual"; then
    printf 'case=%s status=stdout-mismatch\n' "$label" >&2
    diff -u "$expected" "$actual" >&2 || true
    return 6
  fi
  printf 'MADAROS_F64_PARAM_CAST_CASE_PASS case=%s compiler_provenance=%s\n' "$label" "$PROVENANCE"
}

if [[ "$MODE" == "source-fresh" ]]; then
  run_case parameter "$PARAM_SOURCE" "$PARAM_EXPECTED" || fail source_fresh_parameter_case
  run_case controls "$CONTROL_SOURCE" "$CONTROL_EXPECTED" || fail source_fresh_control_case
  printf 'MADAROS_F64_PARAM_CAST_PASS compiler_provenance=source-fresh compiler_surface=%s compiler_sha256=%s casts=direct:4,copied:9,second:3,negative:-2 controls=local:6,arithmetic:8,int:-11\n' \
    "$COMPILER_SURFACE" "$compiler_sha256"
  exit 0
fi

if run_case parameter "$PARAM_SOURCE" "$PARAM_EXPECTED"; then
  parameter_rc=0
else
  parameter_rc=$?
fi
if run_case controls "$CONTROL_SOURCE" "$CONTROL_EXPECTED"; then
  controls_rc=0
else
  controls_rc=$?
fi

printf 'MADAROS_F64_PARAM_CAST_CLASSIFY compiler_provenance=stale/default authority=diagnostic-only compiler_surface=%s compiler_sha256=%s parameter_rc=%s controls_rc=%s source_fresh_acceptance=not-run\n' \
  "$COMPILER_SURFACE" "$compiler_sha256" "$parameter_rc" "$controls_rc"
