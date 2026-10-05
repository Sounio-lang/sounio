#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 2

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export SOUNIO_MODULE_FRONTEND_LOWER_TRACE="${SOUNIO_MODULE_FRONTEND_LOWER_TRACE:-1}"
export SOUNIO_LOWER_RSS_TRACE="${SOUNIO_LOWER_RSS_TRACE:-1}"

OUT_DIR="${SOUNIO_IMPORTED_SEED_LOWERING_DIR:-$(mktemp -d /tmp/sounio-imported-seed-lowering.XXXXXX)}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
mkdir -p "$OUT_DIR"

run_compile() {
  local label="$1"
  local src="$2"
  local out="$OUT_DIR/$label.elf"
  local log="$OUT_DIR/$label.compile.log"
  set +e
  "$SOUC" compile "$src" -o "$out" >"$log" 2>&1
  local rc=$?
  set -e
  printf '%s_compile_rc=%s\n' "$label" "$rc"
  printf '%s_log=%s\n' "$label" "$log"
  return "$rc"
}

expect_compile_ok() {
  local label="$1"
  local src="$2"
  if ! run_compile "$label" "$src"; then
    echo "FAIL $label expected compile success"
    tail -n 80 "$OUT_DIR/$label.compile.log"
    return 1
  fi
  if ! grep -q 'Compilation successful!' "$OUT_DIR/$label.compile.log"; then
    echo "FAIL $label missing success marker"
    tail -n 80 "$OUT_DIR/$label.compile.log"
    return 1
  fi
  return 0
}

expect_run_output() {
  local label="$1"
  local expected="$2"
  local out="$OUT_DIR/$label.elf"
  local run_log="$OUT_DIR/$label.run.log"
  chmod +x "$out" 2>/dev/null || true
  set +e
  "$out" >"$run_log" 2>&1
  local rc=$?
  set -e
  printf '%s_run_rc=%s\n' "$label" "$rc"
  printf '%s_run_log=%s\n' "$label" "$run_log"
  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL $label expected run rc=0"
    tail -n 80 "$run_log"
    return 1
  fi
  if [[ "$(cat "$run_log")" != "$expected" ]]; then
    echo "FAIL $label expected output: $expected"
    tail -n 80 "$run_log"
    return 1
  fi
  return 0
}

expect_compile_ok_and_run_output() {
  local label="$1"
  local src="$2"
  local expected="$3"
  expect_compile_ok "$label" "$src" || return 1
  expect_run_output "$label" "$expected" || return 1
  echo "FIXED $label no_lowering_sigsegv output_ok"
  return 0
}

expect_compile_ok_and_run_rc() {
  local label="$1"
  local src="$2"
  local expected_rc="$3"
  expect_compile_ok "$label" "$src" || return 1
  local out="$OUT_DIR/$label.elf"
  local run_log="$OUT_DIR/$label.run.log"
  chmod +x "$out" 2>/dev/null || true
  set +e
  "$out" >"$run_log" 2>&1
  local rc=$?
  set -e
  printf '%s_run_rc=%s\n' "$label" "$rc"
  printf '%s_run_log=%s\n' "$label" "$run_log"
  if [[ "$rc" -ne "$expected_rc" ]]; then
    echo "FAIL $label expected run rc=$expected_rc"
    tail -n 80 "$run_log"
    return 1
  fi
  echo "FIXED $label deterministic_runtime_rc=$expected_rc"
  return 0
}

status=0
expect_compile_ok_and_run_output imported_seq_count tests/compiler/imported_seed_lowering/seq_count_main.sio "0" || status=1
expect_compile_ok_and_run_output imported_seq_push tests/compiler/imported_seed_lowering/seq_push_main.sio "5" || status=1
expect_compile_ok_and_run_output imported_seq_field_push tests/compiler/imported_seed_lowering/seq_field_push_main.sio "28" || status=1
expect_compile_ok_and_run_output imported_seq_len_unwrap tests/compiler/imported_seed_lowering/seq_len_unwrap_main.sio "33" || status=1
expect_compile_ok_and_run_output imported_seq_free_builtins tests/compiler/imported_seed_lowering/seq_free_builtins_main.sio "81" || status=1
expect_compile_ok_and_run_output imported_seq_access tests/compiler/imported_seed_lowering/seq_access_main.sio "11" || status=1
expect_compile_ok_and_run_output imported_seq_struct_access tests/compiler/imported_seed_lowering/seq_struct_access_main.sio "11" || status=1
expect_compile_ok_and_run_rc imported_seq_oob_get tests/compiler/imported_seed_lowering/seq_oob_get_main.sio 1 || status=1

if [[ "$status" -eq 0 ]]; then
  echo "PASS madaros_imported_seed_lowering_gate Seq<T> lowering SIGSEGV reduced"
else
  echo "FAIL madaros_imported_seed_lowering_gate"
fi
echo "artifact_dir=$OUT_DIR"
exit "$status"
