#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ROOT="${ZERO_EVENT_SOURCE_ROOT:-$ROOT_DIR}"
MADAROS_BIN="${MADAROS_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zero-native-privacy.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[zero-native-privacy] FAIL: $*" >&2
  exit 1
}

compile_rejects() {
  local name="$1"
  local source="$2"
  local marker="$3"
  local log="$TMP_DIR/$name.log"

  if (cd "$SOURCE_ROOT" && timeout 30 "$MADAROS_BIN" compile "$source" -o "$TMP_DIR/$name") >"$log" 2>&1; then
    cat "$log" >&2
    fail "$name unexpectedly compiled"
  fi
  grep -Fq "$marker" "$log" || {
    cat "$log" >&2
    fail "$name did not emit $marker"
  }
}

compile_accepts() {
  local name="$1"
  local source="$2"
  local log="$TMP_DIR/$name.log"

  (cd "$SOURCE_ROOT" && timeout 60 "$MADAROS_BIN" compile "$source" -o "$TMP_DIR/$name") >"$log" 2>&1 || {
    cat "$log" >&2
    fail "$name did not compile"
  }
  grep -Fq 'Compilation successful!' "$log" || {
    cat "$log" >&2
    fail "$name compiled without the native emission marker"
  }
  [[ -s "$TMP_DIR/$name" ]] || fail "$name reported success without an output artifact"
}

[[ -x "$MADAROS_BIN" ]] || fail "rebuilt Madaros not executable: $MADAROS_BIN"

compile_rejects receipt \
  tests/compile-fail/zero_event_direct_receipt_construction.sio \
  'error[E176'
compile_rejects erased \
  tests/compile-fail/zero_event_direct_erased_construction.sio \
  'error[E176'
compile_rejects private-struct \
  tests/multimodule/visibility_struct_private_main.sio \
  'error[E176'
compile_rejects private-fn \
  tests/multimodule/visibility_fn_private_main.sio \
  'error[E175'
compile_rejects private-enum \
  tests/multimodule/visibility_enum_private_main.sio \
  'error[E177'

compile_accepts public-struct tests/multimodule/visibility_struct_pub_main.sio
compile_accepts generic-public tests/multimodule/wp_a3/w2_main.sio
compile_accepts zero-event-positive tests/known_failures/zero_event_stdlib_native_v2_probe.sio
compile_accepts eisa-core tests/stdlib/eisa/test_eisa_core.sio

eisa_core_run_log="$TMP_DIR/eisa-core.run.log"
chmod +x "$TMP_DIR/eisa-core"
timeout 30 "$TMP_DIR/eisa-core" >"$eisa_core_run_log" 2>&1 || {
  cat "$eisa_core_run_log" >&2
  fail "eisa-core compiled but failed at runtime"
}
grep -Fq 'ALL PASS: eisa core W1 W2 W3 W4 W5' "$eisa_core_run_log" || {
  cat "$eisa_core_run_log" >&2
  fail "eisa-core runtime did not emit the W1-W5 receipt"
}

eisa_log="$TMP_DIR/eisa.log"
if (cd "$SOURCE_ROOT" && timeout 60 "$MADAROS_BIN" compile \
    tests/known_failures/eisa_zero_flags_native_v2_probe.sio \
    -o "$TMP_DIR/eisa") >"$eisa_log" 2>&1; then
  :
else
  grep -Fq 'run_check_mode: verdict=0' "$eisa_log" || {
    cat "$eisa_log" >&2
    fail "EISA did not pass the visibility preflight"
  }
  grep -Fq 'Failed to write native binary' "$eisa_log" || {
    cat "$eisa_log" >&2
    fail "EISA failure moved away from its classified backend frontier"
  }
fi

echo '[zero-native-privacy] PASS: private constructors rejected; public, generic, zero-event, and EISA compile paths preserved; EISA W1-W5 runtime receipt verified'
