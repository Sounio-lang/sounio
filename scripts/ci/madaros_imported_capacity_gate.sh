#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_KEEP:-0}"

fail() {
  echo "[madaros-imported-capacity] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-imported-capacity.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_BIN:-$WORK/madaros}"
MAIN="$WORK/main.sio"
DEP="$WORK/cap_dep.sio"
OUT="$WORK/main.elf"
LOG="$WORK/compile.log"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "rebuilt Madaros is missing or not executable: $MADAROS_ELF"

for i in $(seq 0 1023); do
  printf 'pub fn dep%s() -> i64 { return %s }\n' "$i" "$i" >>"$DEP"
done

printf 'use cap_dep::{dep0}\n' >"$MAIN"
for i in $(seq 0 1023); do
  printf 'fn local%s() -> i64 { return %s }\n' "$i" "$i" >>"$MAIN"
done
printf 'fn main() -> i64 { return dep0() }\n' >>"$MAIN"

set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$MAIN" -o "$OUT" >"$LOG" 2>&1
compile_rc=$?
set -e

if [[ "$compile_rc" -eq 0 ]]; then
  cat "$LOG" >&2
  fail "aggregate capacity witness unexpectedly compiled"
fi
if [[ "$compile_rc" -ge 128 ]]; then
  cat "$LOG" >&2
  fail "aggregate capacity witness terminated by signal rc=$compile_rc"
fi
if [[ -e "$OUT" ]]; then
  cat "$LOG" >&2
  fail "aggregate capacity rejection left an output artifact: $OUT"
fi
grep -Fq 'too many functions: shared IR module capacity exceeded (max 2048 slots)' "$LOG" || {
  cat "$LOG" >&2
  fail "aggregate capacity diagnostic was missing or changed"
}
if grep -Fq 'ir_summary_failed' "$LOG"; then
  cat "$LOG" >&2
  fail "aggregate capacity rejection degraded to ir_summary_failed"
fi

echo "[madaros-imported-capacity] PASS: 2049 aggregate imported slots are rejected before IR summary overflow"
