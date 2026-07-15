#!/usr/bin/env bash
# Prove imported struct-return identity survives and changes across assignment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${SOUNIO_IMPORTED_STRUCT_REASSIGN_MADAROS:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
WITNESS="$ROOT_DIR/tests/compiler/imported_struct_reassignment/main.sio"
WORK="$(mktemp -d /tmp/sounio-imported-struct-reassign.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "[imported-struct-reassignment] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros wrapper is missing: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit current-source Madaros ELF'
[[ -x "$RAW_MADAROS" ]] || fail "current-source Madaros is missing: $RAW_MADAROS"

export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" build "$WITNESS" -o "$WORK/witness" >"$WORK/build.log" 2>&1 || {
  cat "$WORK/build.log" >&2
  fail 'witness did not compile'
}
chmod +x "$WORK/witness"

set +e
"$WORK/witness" >"$WORK/run.log" 2>&1
rc=$?
set -e
if [[ "$rc" -ne 42 ]]; then
  cat "$WORK/build.log" >&2
  cat "$WORK/run.log" >&2
  fail "witness returned rc=$rc instead of 42"
fi

echo '[imported-struct-reassignment] receipt initial_let=PASS same_type_reassign=PASS replacement_type_reassign=PASS noncall_scalar=PASS nominal_fallback=FORBIDDEN legacy_raw_field_extension=FINAL'
