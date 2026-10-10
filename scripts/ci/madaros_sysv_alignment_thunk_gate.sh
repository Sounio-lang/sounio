#!/usr/bin/env bash
# #2537: SysV callee alignment witness for dynlink thunk.
# Builds lib_align.c (SSE store+load on stack), compiles Sounio test, verifies runtime.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC=tests/run-pass/sysv_alignment_thunk_witness.sio
WORKDIR="${TMPDIR:-/tmp}/sounio-sysv-align-$$"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

SO="$WORKDIR/libkl14d_align.so"
OUT="$WORKDIR/sysv_alignment_thunk.elf"

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "madaros_sysv_alignment_thunk_gate: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || {
  echo "madaros_sysv_alignment_thunk_gate: set MADAROS_RAW_BIN to a Madaros ELF" >&2
  exit 2
}

gcc -shared -fPIC -O0 -msse4.1 -o "$SO" tests/fixtures/kl14d/lib_align.c
echo "MADAROS_SYSV_ALIGNMENT_THUNK_GATE_OK"

if ! MADAROS_RAW_BIN="$RAW" "$SOUC" build "$SRC" -o "$OUT" >"$WORKDIR/compile.out" 2>"$WORKDIR/compile.err"; then
  echo "madaros_sysv_alignment_thunk_gate: FAIL compile" >&2
  cat "$WORKDIR/compile.out" "$WORKDIR/compile.err" >&2
  exit 1
fi

chmod +x "$OUT"
export LD_LIBRARY_PATH="$WORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set +e
got="$("$OUT" 2>"$WORKDIR/run.err")"
rc=$?
set -e
if [[ $rc -ne 0 ]] || ! echo "$got" | grep -q 'sysv_alignment_thunk: PASS'; then
  echo "madaros_sysv_alignment_thunk_gate: FAIL run rc=$rc stdout=[$got]" >&2
  cat "$WORKDIR/run.err" >&2 || true
  exit 1
fi

echo "MADAROS_SYSV_ALIGNMENT_THUNK_GATE_OK"
