#!/usr/bin/env bash
# KL-14b: Madaros must emit a dynamically linked ELF that resolves one
# non-builtin extern (kl14b_add) from a controlled shared library.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC=tests/run-pass/kl14b_dynlink_one_symbol.sio
PROBE_C=tests/fixtures/kl14b/kl14b_probe.c
WORKDIR="${TMPDIR:-/tmp}/sounio-kl14b-$$"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

SO="$WORKDIR/libkl14b_probe.so"
OUT="$WORKDIR/kl14b_dynlink_one_symbol.elf"

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "madaros_kl14b_dynlink_gate: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || {
  echo "madaros_kl14b_dynlink_gate: set MADAROS_RAW_BIN to a Madaros ELF" >&2
  exit 2
}

gcc -shared -fPIC -O0 -o "$SO" "$PROBE_C"

# bin/souc + MADAROS_RAW_BIN is the Witness Gate shape (see KL-4 gate).
if ! MADAROS_RAW_BIN="$RAW" "$SOUC" build "$SRC" -o "$OUT" 2>"$WORKDIR/compile.err"; then
  echo "madaros_kl14b_dynlink_gate: FAIL compile" >&2
  cat "$WORKDIR/compile.err" >&2
  exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
  echo "madaros_kl14b_dynlink_gate: readelf missing" >&2
  exit 2
fi

dyn="$(readelf -d "$OUT" 2>/dev/null || true)"
echo "$dyn" | grep -q 'Shared library: \[libkl14b_probe.so\]' || {
  echo "madaros_kl14b_dynlink_gate: FAIL missing DT_NEEDED libkl14b_probe.so" >&2
  echo "$dyn" >&2
  exit 1
}
readelf -l "$OUT" | grep -q 'INTERP' || {
  echo "madaros_kl14b_dynlink_gate: FAIL missing PT_INTERP" >&2
  exit 1
}
readelf -l "$OUT" | grep -q 'DYNAMIC' || {
  echo "madaros_kl14b_dynlink_gate: FAIL missing PT_DYNAMIC" >&2
  exit 1
}

chmod +x "$OUT"
export LD_LIBRARY_PATH="$WORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
got="$("$OUT" || true)"
echo "$got" | grep -q 'kl14b_dynlink_one_symbol: PASS' || {
  echo "madaros_kl14b_dynlink_gate: FAIL run output: $got" >&2
  exit 1
}

echo "MADAROS_KL14B_DYNLINK_GATE_OK"
