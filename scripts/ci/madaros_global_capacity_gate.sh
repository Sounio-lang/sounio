#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_KEEP:-0}"
SOURCE="$ROOT_DIR/tests/compile-fail/door1_too_many_globals_1025.sio"

fail() {
  echo "[madaros-global-capacity] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-global-capacity.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_BIN:-$WORK/madaros}"
OUT="$WORK/door1.elf"
LOG="$WORK/compile.log"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "rebuilt Madaros is missing or not executable: $MADAROS_ELF"

set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$SOURCE" -o "$OUT" >"$LOG" 2>&1
compile_rc=$?
set -e

if [[ "$compile_rc" -eq 0 ]]; then
  cat "$LOG" >&2
  fail "capacity witness unexpectedly compiled"
fi
if [[ "$compile_rc" -ge 128 ]]; then
  cat "$LOG" >&2
  fail "capacity witness terminated by signal rc=$compile_rc"
fi
if [[ -e "$OUT" ]]; then
  cat "$LOG" >&2
  fail "capacity rejection left an output artifact: $OUT"
fi
grep -Fq 'too many globals: shared IR module capacity exceeded (max 2048 slots)' "$LOG" || {
  cat "$LOG" >&2
  fail "capacity diagnostic was missing or changed"
}

echo "[madaros-global-capacity] PASS: 2049 globals are rejected before IR slot overflow with a bounded diagnostic"
