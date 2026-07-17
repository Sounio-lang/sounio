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
BOUNDARY_SOURCE="$WORK/boundary.sio"
BOUNDARY_OUT="$WORK/boundary.elf"
BOUNDARY_LOG="$WORK/boundary.log"
BOUNDARY_RUN_LOG="$WORK/boundary.run.log"
OVERFLOW_SOURCE="$WORK/overflow.sio"
OVERFLOW_OUT="$WORK/overflow.elf"
OVERFLOW_LOG="$WORK/overflow.log"

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

for i in $(seq 0 2046); do
  printf 'var g%s: i64 = 0\n' "$i" >>"$BOUNDARY_SOURCE"
done
cat >>"$BOUNDARY_SOURCE" <<'SOUNIO'
fn main() -> i32 with IO, Mut, Panic {
  0
}
SOUNIO

set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$BOUNDARY_SOURCE" -o "$BOUNDARY_OUT" >"$BOUNDARY_LOG" 2>&1
boundary_compile_rc=$?
set -e

if [[ "$boundary_compile_rc" -ne 0 ]]; then
  cat "$BOUNDARY_LOG" >&2
  fail "2047-global boundary witness did not compile rc=$boundary_compile_rc"
fi
[[ -e "$BOUNDARY_OUT" ]] || fail "2047-global boundary witness did not produce an output artifact"
chmod +x "$BOUNDARY_OUT"
set +e
"$BOUNDARY_OUT" >"$BOUNDARY_RUN_LOG" 2>&1
boundary_run_rc=$?
set -e
if [[ "$boundary_run_rc" -ne 0 ]]; then
  cat "$BOUNDARY_RUN_LOG" >&2
  fail "2047-global boundary witness did not execute cleanly rc=$boundary_run_rc"
fi

for i in $(seq 0 2047); do
  printf 'var g%s: i64 = 0\n' "$i" >>"$OVERFLOW_SOURCE"
done
cat >>"$OVERFLOW_SOURCE" <<'SOUNIO'
fn main() -> i32 with IO, Mut, Panic {
  0
}
SOUNIO

set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$OVERFLOW_SOURCE" -o "$OVERFLOW_OUT" >"$OVERFLOW_LOG" 2>&1
overflow_compile_rc=$?
set -e

if [[ "$overflow_compile_rc" -eq 0 ]]; then
  cat "$OVERFLOW_LOG" >&2
  fail "2048-global boundary witness unexpectedly compiled"
fi
if [[ "$overflow_compile_rc" -ge 128 ]]; then
  cat "$OVERFLOW_LOG" >&2
  fail "2048-global boundary witness terminated by signal rc=$overflow_compile_rc"
fi
if [[ -e "$OVERFLOW_OUT" ]]; then
  cat "$OVERFLOW_LOG" >&2
  fail "2048-global capacity rejection left an output artifact: $OVERFLOW_OUT"
fi
grep -Fq 'too many globals: shared IR module capacity exceeded (max 2047 slots)' "$OVERFLOW_LOG" || {
  cat "$OVERFLOW_LOG" >&2
  fail "2048-global capacity diagnostic was missing or changed"
}

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
grep -Fq 'too many globals: shared IR module capacity exceeded (max 2047 slots)' "$LOG" || {
  cat "$LOG" >&2
  fail "capacity diagnostic was missing or changed"
}

echo "[madaros-global-capacity] PASS: 2047 globals execute; exact 2048 boundary and canonical 2049 fixture reject before IR slot overflow"
