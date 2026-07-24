#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
WORK="${SOUNIO_MADAROS_CALL_ARITY_13_DIR:-$(mktemp -d /tmp/sounio-madaros-call-arity-13.XXXXXX)}"
SOURCE="$ROOT_DIR/tests/multimodule/madaros_imported_call_arity_13_main.sio"
ELF="$WORK/call-arity-13.elf"

fail() {
  echo "[madaros-call-arity-13] FAIL: $*" >&2
  exit 1
}

[[ -n "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN must name a current-source Madaros ELF"
[[ -x "$RAW_MADAROS" ]] || fail "Madaros ELF is missing or not executable: $RAW_MADAROS"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
"$RAW_MADAROS" --native-compile "$SOURCE" -o "$ELF" >"$WORK/compile.log" 2>&1 || {
  cat "$WORK/compile.log" >&2
  fail "imported 13-argument witness did not compile"
}
[[ -s "$ELF" ]] || fail "compiler did not emit an ELF"
chmod +x "$ELF"

"$ELF" >"$WORK/run.log" 2>&1 || {
  cat "$WORK/run.log" >&2
  fail "imported 13-argument witness did not execute"
}
grep -Fxq 'MADAROS_IMPORTED_CALL_ARITY_13_OK' "$WORK/run.log" || {
  cat "$WORK/run.log" >&2
  fail "runtime marker is missing"
}

echo "[madaros-call-arity-13] PASS: imported call preserved all 13 ordered arguments"
