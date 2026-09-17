#!/usr/bin/env bash
# KL-14c: Madaros must resolve N non-builtin externs from one DT_NEEDED .so.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC=tests/run-pass/kl14c_dynlink_n_symbols.sio
PROBE_C=tests/fixtures/kl14c/kl14c_probe.c
WORKDIR="${TMPDIR:-/tmp}/sounio-kl14c-$$"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

SO="$WORKDIR/libkl14c_probe.so"
OUT="$WORKDIR/kl14c_dynlink_n_symbols.elf"

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "madaros_kl14c_dynlink_gate: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || {
  echo "madaros_kl14c_dynlink_gate: set MADAROS_RAW_BIN to a Madaros ELF" >&2
  exit 2
}

gcc -shared -fPIC -O0 -o "$SO" "$PROBE_C"
nm -D "$SO" | grep -q ' T kl14c_add$' || {
  echo "madaros_kl14c_dynlink_gate: FAIL probe missing kl14c_add" >&2
  exit 1
}
nm -D "$SO" | grep -q ' T kl14c_mul$' || {
  echo "madaros_kl14c_dynlink_gate: FAIL probe missing kl14c_mul" >&2
  exit 1
}
nm -D "$SO" | grep -q ' T kl14c_neg$' || {
  echo "madaros_kl14c_dynlink_gate: FAIL probe missing kl14c_neg" >&2
  exit 1
}

if ! MADAROS_RAW_BIN="$RAW" "$SOUC" build "$SRC" -o "$OUT" >"$WORKDIR/compile.out" 2>"$WORKDIR/compile.err"; then
  echo "madaros_kl14c_dynlink_gate: FAIL compile" >&2
  cat "$WORKDIR/compile.out" "$WORKDIR/compile.err" >&2
  exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
  echo "madaros_kl14c_dynlink_gate: readelf missing" >&2
  exit 2
fi

dyn="$(readelf -d "$OUT" 2>/dev/null || true)"
echo "$dyn" | grep -q 'Shared library: \[libkl14c_probe.so\]' || {
  echo "madaros_kl14c_dynlink_gate: FAIL missing DT_NEEDED libkl14c_probe.so" >&2
  echo "$dyn" >&2
  exit 1
}
readelf -l "$OUT" | grep -q 'INTERP' || {
  echo "madaros_kl14c_dynlink_gate: FAIL missing PT_INTERP" >&2
  exit 1
}
readelf -l "$OUT" | grep -q 'DYNAMIC' || {
  echo "madaros_kl14c_dynlink_gate: FAIL missing PT_DYNAMIC" >&2
  exit 1
}

# At least three GLOB_DAT relocs (one per symbol).
rela_n="$(readelf -r --use-dynamic "$OUT" 2>/dev/null | grep -c 'R_X86_64_GLOB_DAT' || true)"
if [[ "${rela_n:-0}" -lt 3 ]]; then
  echo "madaros_kl14c_dynlink_gate: FAIL expected >=3 GLOB_DAT, got ${rela_n:-0}" >&2
  readelf -r --use-dynamic "$OUT" >&2 || true
  exit 1
fi

chmod +x "$OUT"
export LD_LIBRARY_PATH="$WORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set +e
got="$("$OUT" 2>"$WORKDIR/run.err")"
rc=$?
set -e
if [[ $rc -ne 0 ]] || ! echo "$got" | grep -q 'kl14c_dynlink_n_symbols: PASS'; then
  echo "madaros_kl14c_dynlink_gate: FAIL run rc=$rc stdout=[$got]" >&2
  echo "--- stderr ---" >&2
  cat "$WORKDIR/run.err" >&2 || true
  echo "--- readelf -d ---" >&2
  readelf -d "$OUT" >&2 || true
  echo "--- readelf -r --use-dynamic ---" >&2
  readelf -r --use-dynamic "$OUT" >&2 || true
  exit 1
fi

echo "MADAROS_KL14C_DYNLINK_GATE_OK"
