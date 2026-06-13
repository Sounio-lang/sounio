#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s 2>/dev/null || echo unknown):$(uname -m 2>/dev/null || echo unknown)" in
  Linux:x86_64|Linux:amd64) ;;
  *)
    echo "[native-v2-global-struct-array] SKIP: x86-64 Linux-only probe"
    exit 0
    ;;
esac

SRC="tests/native-v2/global_struct_array_record_storage_witness.sio"
SOUC_NATIVE="${SOUC_NATIVE:-$ROOT_DIR/bin/souc}"
MADAROS="${MADAROS_LAUNCHER:-$ROOT_DIR/bin/madaros}"
TMP_DIR="$(mktemp -d /tmp/sounio-global-struct-array.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[native-v2-global-struct-array] FAIL: $*" >&2
  exit 1
}

[[ -x "$SOUC_NATIVE" ]] || fail "native compiler is not executable: $SOUC_NATIVE"
[[ -x "$MADAROS" ]] || fail "Madaros launcher is not executable: $MADAROS"
[[ -f "$SRC" ]] || fail "missing witness: $SRC"

echo "[native-v2-global-struct-array] witness=$SRC"

# First isolate the current Madaros source front-half boundary. This is broader
# than struct arrays: user-level globals are not yet visible to Madaros source
# typechecking.
cat >"$TMP_DIR/global_scalar.sio" <<'SRC'
var X: i64 = 0
fn main() -> i64 with Mut { X = 42; return X }
SRC

RAW_SCALAR_ELF="$TMP_DIR/raw_global_scalar.elf"
set +e
"$SOUC_NATIVE" "$TMP_DIR/global_scalar.sio" "$RAW_SCALAR_ELF" >"$TMP_DIR/raw_global_scalar.log" 2>&1
raw_scalar_compile_rc=$?
set -e

if [[ "$raw_scalar_compile_rc" -eq 0 && -s "$RAW_SCALAR_ELF" ]]; then
  chmod +x "$RAW_SCALAR_ELF"
  set +e
  "$RAW_SCALAR_ELF"
  raw_scalar_rc=$?
  set -e
  if [[ "$raw_scalar_rc" -eq 42 ]]; then
    echo "[native-v2-global-struct-array] raw native source globals: PASS"
  else
    echo "[native-v2-global-struct-array] raw native source globals: BLOCKED rc=$raw_scalar_rc"
  fi
else
  echo "[native-v2-global-struct-array] raw native source globals: BLOCKED compile_rc=$raw_scalar_compile_rc"
  grep -aE 'error\[|typecheck|failed' "$TMP_DIR/raw_global_scalar.log" || true
fi

set +e
"$MADAROS" run "$TMP_DIR/global_scalar.sio" >"$TMP_DIR/madaros_global_scalar.log" 2>&1
madaros_rc=$?
set -e

if [[ "$madaros_rc" -eq 42 ]]; then
  echo "[native-v2-global-struct-array] Madaros source globals: PASS"
else
  echo "[native-v2-global-struct-array] Madaros source globals: BLOCKED rc=$madaros_rc"
  grep -aE 'error\[|undeclared variable|type_check_failed' "$TMP_DIR/madaros_global_scalar.log" || true
fi

MADAROS_STRUCT_ELF="$TMP_DIR/madaros_global_struct_array.elf"
set +e
"$MADAROS" compile "$SRC" -o "$MADAROS_STRUCT_ELF" >"$TMP_DIR/madaros_global_struct_array.log" 2>&1
madaros_struct_compile_rc=$?
set -e

if [[ "$madaros_struct_compile_rc" -eq 0 && -s "$MADAROS_STRUCT_ELF" ]]; then
  chmod +x "$MADAROS_STRUCT_ELF"
  set +e
  "$MADAROS_STRUCT_ELF"
  madaros_struct_rc=$?
  set -e
  if [[ "$madaros_struct_rc" -eq 42 ]]; then
    echo "[native-v2-global-struct-array] Madaros global struct array: PASS"
  else
    echo "[native-v2-global-struct-array] Madaros global struct array: BLOCKED rc=$madaros_struct_rc"
  fi
else
  echo "[native-v2-global-struct-array] Madaros global struct array: BLOCKED compile_rc=$madaros_struct_compile_rc"
  grep -aE 'error\[|typecheck|failed' "$TMP_DIR/madaros_global_struct_array.log" || true
fi

# Then probe the lower-level native compiler path, which accepts the global
# aggregate source today but mis-emits the global array-of-struct write/read.
RAW_ELF="$TMP_DIR/global_struct_array.elf"
RAW_LOG="$TMP_DIR/raw_compile.log"
"$SOUC_NATIVE" "$SRC" "$RAW_ELF" >"$RAW_LOG" 2>&1 || {
  echo "[native-v2-global-struct-array] raw native compile: BLOCKED"
  cat "$RAW_LOG"
  exit 0
}
chmod +x "$RAW_ELF"

set +e
"$RAW_ELF"
raw_rc=$?
set -e

if [[ "$raw_rc" -eq 42 ]]; then
  echo "[native-v2-global-struct-array] raw native global struct array: PASS"
  if [[ "${REQUIRE_NATIVE_V2_GLOBAL_STRUCT_ARRAY_PASS:-0}" == "1" ]]; then
    exit 0
  fi
  exit 0
fi

echo "[native-v2-global-struct-array] raw native global struct array: BLOCKED rc=$raw_rc"
echo "[native-v2-global-struct-array] expected fixed exit=42; rc=10 means first stored field was not preserved"

if [[ "${REQUIRE_NATIVE_V2_GLOBAL_STRUCT_ARRAY_PASS:-0}" == "1" ]]; then
  exit 1
fi

exit 0
