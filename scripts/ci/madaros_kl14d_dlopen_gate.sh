#!/usr/bin/env bash
# KL-14d3: Madaros dlopen/dlsym/invoke/dlclose via libdl.so.2 (f(35)==42).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC=tests/run-pass/kl14d_dlopen.sio
WORKDIR="${TMPDIR:-/tmp}/sounio-kl14d-dlopen-$$"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

SO="$WORKDIR/libkl14d_dl.so"
OUT="$WORKDIR/kl14d_dlopen.elf"
SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "madaros_kl14d_dlopen_gate: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || {
  echo "madaros_kl14d_dlopen_gate: set MADAROS_RAW_BIN to a Madaros ELF" >&2
  exit 2
}

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

gcc -shared -fPIC -O0 -o "$SO" tests/fixtures/kl14d/lib_dl.c

if ! MADAROS_RAW_BIN="$RAW" "$SOUC" build "$SRC" -o "$OUT" >"$WORKDIR/compile.out" 2>"$WORKDIR/compile.err"; then
  echo "madaros_kl14d_dlopen_gate: FAIL compile" >&2
  cat "$WORKDIR/compile.out" "$WORKDIR/compile.err" >&2
  exit 1
fi

dyn="$(readelf -d "$OUT" 2>/dev/null || true)"
echo "$dyn" | grep -q 'Shared library: \[libdl\.so\.2\]' || {
  echo "madaros_kl14d_dlopen_gate: FAIL missing DT_NEEDED libdl.so.2" >&2
  echo "$dyn" >&2
  exit 1
}

chmod +x "$OUT"
export LD_LIBRARY_PATH="$WORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set +e
got="$("$OUT" 2>"$WORKDIR/run.err")"
rc=$?
set -e
if [[ $rc -ne 0 ]] || ! echo "$got" | grep -q 'kl14d_dlopen: PASS'; then
  echo "madaros_kl14d_dlopen_gate: FAIL run rc=$rc stdout=[$got]" >&2
  cat "$WORKDIR/run.err" >&2 || true
  readelf -d "$OUT" >&2 || true
  # If SEGV/unresolved, libc may be required as an extra NEEDED — surface that.
  exit 1
fi

echo "MADAROS_KL14D_DLOPEN_GATE_OK"
