#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s 2>/dev/null || echo unknown):$(uname -m 2>/dev/null || echo unknown)" in
  Linux:x86_64|Linux:amd64) ;;
  *)
    echo "[native-v2-compile-imported-source] SKIP: x86-64 Linux-only gate"
    exit 0
    ;;
esac

MADAROS="${MADAROS_LAUNCHER:-$ROOT_DIR/bin/madaros}"
TMP_DIR="$(mktemp -d /tmp/sounio-native-v2-compile-imported-source.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[native-v2-compile-imported-source] FAIL: $*" >&2
  exit 1
}

run_case() {
  local name="$1"
  local src="$2"
  local expected_fns="$3"
  local elf="$TMP_DIR/$name.elf"
  local log="$TMP_DIR/$name.log"

  echo "[native-v2-compile-imported-source] running $name"
  "$MADAROS" --native-v2-compile "$src" -o "$elf" >"$log" 2>&1 || {
    cat "$log" >&2
    fail "$name compile failed"
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
  if grep -q 'native_prebundle:' "$log"; then
    cat "$log" >&2
    fail "$name used native prebundle"
  fi

  [[ -s "$elf" ]] || {
    cat "$log" >&2
    fail "$name did not produce an ELF"
  }
  chmod +x "$elf"

  set +e
  "$elf" >"$TMP_DIR/$name.stdout" 2>"$TMP_DIR/$name.stderr"
  local rc=$?
  set -e
  [[ "$rc" -eq 42 ]] || {
    cat "$log" >&2
    cat "$TMP_DIR/$name.stdout" >&2 || true
    cat "$TMP_DIR/$name.stderr" >&2 || true
    fail "$name expected exit 42, got $rc"
  }
}

[[ -x "$MADAROS" ]] || fail "Madaros launcher is not executable: $MADAROS"

run_case imported_core tests/selfhost/native_runtime/import_core_abi_42.sio 6
run_case imported_hof tests/selfhost/native_runtime/import_hof_abi_42.sio 6
run_case imported_mixed tests/selfhost/native_runtime/import_body_lowering_42.sio 7

echo "NATIVE_V2_COMPILE_IMPORTED_SOURCE_PASS"
