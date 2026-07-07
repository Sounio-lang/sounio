#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CASE_TIMEOUT="${EISA_MADAROS_NATIVE_FAIL_CLOSED_TIMEOUT:-300}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-eisa-native-fail-closed.XXXXXX")"
LOG_PATH="$WORK_DIR/native_compile.log"

cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

fail() {
  echo "[eisa-madaros-native-fail-closed] FAIL: $*" >&2
  if [[ -s "$LOG_PATH" ]]; then
    echo "[eisa-madaros-native-fail-closed] --- native compile log ---" >&2
    tail -120 "$LOG_PATH" >&2
  fi
  exit 1
}

make_overcapacity_case() {
  local case_dir="$WORK_DIR/overcapacity"
  mkdir -p "$case_dir/overcap"
  {
    echo 'use overcap::mod::*'
    echo
    echo 'fn main() -> i64 {'
    echo '    var total: i64 = 0'
    local j=0
    while [[ "$j" -lt 130 ]]; do
      printf '    total = total + f%03d()\n' "$j"
      j=$((j + 1))
    done
    echo '    total'
    echo '}'
  } >"$case_dir/main.sio"
  : >"$case_dir/overcap/mod.sio"
  local i=0
  while [[ "$i" -lt 130 ]]; do
    printf 'fn f%03d() -> i64 {\n    %d\n}\n\n' "$i" "$i" >>"$case_dir/overcap/mod.sio"
    i=$((i + 1))
  done
  printf '%s\n' "$case_dir/main.sio"
}

run_native_compile_case() {
  local name="$1"
  local program="$2"
  local require_overcapacity="$3"
  local out_path="$WORK_DIR/${name}.elf"
  LOG_PATH="$WORK_DIR/${name}.native_compile.log"

  echo "[eisa-madaros-native-fail-closed] case=$name source-fresh-native-compile"
  set +e
  SOUNIO_SOUC_ENGINE=lean_single \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  SOUNIO_MODULE_FRONTEND_LOWER_TRACE=1 \
  timeout "$CASE_TIMEOUT" \
    bash scripts/lib/run_selfhost_fresh.sh "$ROOT_DIR/bin/souc" self-hosted/compiler/lean.sio -- \
      --native-compile "$program" -o "$out_path" \
    >"$LOG_PATH" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 124 ]]; then
    fail "$name timed out after ${CASE_TIMEOUT}s"
  fi

  if [[ "$rc" -eq 139 ]] || grep -Eiq 'segmentation fault|core dumped' "$LOG_PATH"; then
    fail "$name regressed to SIGSEGV instead of a classified failure"
  fi

  if [[ "$rc" -eq 0 ]]; then
    if [[ "$require_overcapacity" == "1" ]]; then
      fail "$name unexpectedly compiled; this generated witness must remain over capacity"
    fi
    echo "[eisa-madaros-native-fail-closed] PASS: $name succeeded cleanly"
    return
  fi

  grep -q 'imported_simple_ir_over_capacity' "$LOG_PATH" \
    || fail "$name nonzero compile must be classified as imported_simple_ir_over_capacity"

  if [[ -e "$out_path" ]]; then
    fail "$name wrote an output artifact despite nonzero native compile"
  fi

  echo "[eisa-madaros-native-fail-closed] PASS: $name fails closed with imported_simple_ir_over_capacity (rc=$rc)"
}

OVERCAPACITY_MAIN="$(make_overcapacity_case)"
run_native_compile_case generated_overcapacity "$OVERCAPACITY_MAIN" 1
run_native_compile_case test_eisa_v1e_showcase tests/stdlib/eisa/test_eisa_v1e_showcase.sio 0
