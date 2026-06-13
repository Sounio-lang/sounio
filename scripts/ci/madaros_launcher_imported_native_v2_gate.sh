#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s 2>/dev/null || echo unknown):$(uname -m 2>/dev/null || echo unknown)" in
  Linux:x86_64|Linux:amd64) ;;
  *)
    echo "[madaros-launcher-imported-native-v2] SKIP: x86-64 Linux-only gate"
    exit 0
    ;;
esac

MADAROS="$ROOT_DIR/bin/madaros"
TMP_DIR="$(mktemp -d /tmp/sounio-madaros-launcher-imported-native-v2.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[madaros-launcher-imported-native-v2] FAIL: $*" >&2
  exit 1
}

run_case() {
  local name="$1"
  local src="$2"
  local expected_fns="$3"
  local log="$TMP_DIR/$name.log"

  echo "[madaros-launcher-imported-native-v2] running $name"
  set +e
  "$MADAROS" run "$src" >"$log" 2>&1
  local rc=$?
  set -e

  [[ "$rc" -eq 42 ]] || {
    cat "$log" >&2
    fail "$name expected exit 42, got $rc"
  }
  grep -q "Merged IR: $expected_fns" "$log" || {
    cat "$log" >&2
    fail "$name did not use multimodule IR preflight"
  }
  grep -q 'native_v2_compile: emitted path=' "$log" || {
    cat "$log" >&2
    fail "$name did not report native-v2 emission"
  }
  if grep -q 'module_native_driver: imported source uses compact modular IR table path' "$log"; then
    cat "$log" >&2
    fail "$name went through compact imported emitter"
  fi
  if grep -q 'falling back to full IR path' "$log"; then
    cat "$log" >&2
    fail "$name used imported native fallback"
  fi
  if grep -q 'native_prebundle:' "$log"; then
    cat "$log" >&2
    fail "$name used native prebundle"
  fi
}

run_exit_case() {
  local name="$1"
  local src="$2"
  local expected_rc="$3"
  local expected_fns="$4"
  local log="$TMP_DIR/$name.log"

  echo "[madaros-launcher-imported-native-v2] running $name"
  set +e
  "$MADAROS" run "$src" >"$log" 2>&1
  local rc=$?
  set -e

  [[ "$rc" -eq "$expected_rc" ]] || {
    cat "$log" >&2
    fail "$name expected exit $expected_rc, got $rc"
  }
  grep -q "Merged IR: $expected_fns" "$log" || {
    cat "$log" >&2
    fail "$name did not use multimodule IR preflight"
  }
  grep -q 'native_v2_compile: emitted path=' "$log" || {
    cat "$log" >&2
    fail "$name did not report native-v2 emission"
  }
  if grep -q 'module_native_driver: imported source uses compact modular IR table path' "$log"; then
    cat "$log" >&2
    fail "$name went through compact imported emitter"
  fi
  if grep -q 'falling back to full IR path' "$log"; then
    cat "$log" >&2
    fail "$name used imported native fallback"
  fi
  if grep -q 'native_prebundle:' "$log"; then
    cat "$log" >&2
    fail "$name used native prebundle"
  fi
}

run_stdout_case() {
  local name="$1"
  local target="$2"
  local expected_text="$3"
  local expected_fns="$4"
  local log="$TMP_DIR/$name.log"

  echo "[madaros-launcher-imported-native-v2] running $name"
  set +e
  "$MADAROS" run "$target" >"$log" 2>&1
  local rc=$?
  set -e

  [[ "$rc" -eq 0 ]] || {
    cat "$log" >&2
    fail "$name expected exit 0, got $rc"
  }
  grep -q "$expected_text" "$log" || {
    cat "$log" >&2
    fail "$name did not print expected text: $expected_text"
  }
  grep -q "Merged IR: $expected_fns" "$log" || {
    cat "$log" >&2
    fail "$name did not use multimodule IR preflight"
  }
  grep -q 'native_v2_compile: emitted path=' "$log" || {
    cat "$log" >&2
    fail "$name did not report native-v2 emission"
  }
  if grep -q 'module_native_driver: imported source uses compact modular IR table path' "$log"; then
    cat "$log" >&2
    fail "$name went through compact imported emitter"
  fi
  if grep -q 'falling back to full IR path' "$log"; then
    cat "$log" >&2
    fail "$name used imported native fallback"
  fi
  if grep -q 'native_prebundle:' "$log"; then
    cat "$log" >&2
    fail "$name used native prebundle"
  fi
}

[[ -x "$MADAROS" ]] || fail "Madaros launcher is not executable: $MADAROS"

run_case imported_core tests/selfhost/native_runtime/import_core_abi_42.sio 6
run_case imported_hof tests/selfhost/native_runtime/import_hof_abi_42.sio 6
run_case imported_mixed tests/selfhost/native_runtime/import_body_lowering_42.sio 7
run_stdout_case hello_pkg examples/projects/hello_pkg 42 2
run_exit_case thin_single tests/multimodule/thin_single_main.sio 7 6
run_exit_case thin_sum5 tests/multimodule/thin_five_literal_args_main.sio 31 6
run_exit_case thin_sum6 tests/multimodule/thin_six_literal_args_main.sio 63 6
run_exit_case thin_sum7 tests/multimodule/thin_seven_literal_args_main.sio 127 6

echo "MADAROS_LAUNCHER_IMPORTED_NATIVE_V2_PASS"
