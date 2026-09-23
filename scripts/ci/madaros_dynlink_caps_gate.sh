#!/usr/bin/env bash
# Regression gate: >8 dynlink symbols across >4 DT_NEEDED libraries.
# The old hard caps (max_symbols=8, max_libs=4, dynstr=256) rejected this
# with rc=22/23. The caps are now derived from actual need.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SRC=tests/run-pass/dynlink_caps_exceeded.sio
WORKDIR="${TMPDIR:-/tmp}/sounio-dyncaps-$$"
mkdir -p "$WORKDIR"
trap "rm -rf \"$WORKDIR\"" EXIT

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "gate: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || { echo "gate: set MADAROS_RAW_BIN to a Madaros ELF" >&2; exit 2; }

gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14b_probe.so" tests/fixtures/kl14b/kl14b_probe.c
gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14c_probe.so" tests/fixtures/kl14c/kl14c_probe.c
gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14d_a.so" tests/fixtures/kl14d/lib_a.c
gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14d_b.so" tests/fixtures/kl14d/lib_b.c
Z="$(find /usr/lib /lib -name "libzstd.so.1*" 2>/dev/null | head -1 || true)"
[[ -n "$Z" ]] || { echo "gate: libzstd.so.1 not found" >&2; exit 2; }
cp -Lf "$Z" "$WORKDIR/libzstd.so.1"

OUT="$WORKDIR/caps.elf"
if ! MADAROS_RAW_BIN="$RAW" "$SOUC" build "$SRC" -o "$OUT" >"$WORKDIR/c.out" 2>"$WORKDIR/c.err"; then
  echo "gate: FAIL compile (old caps would give rc=22/23 here)" >&2
  cat "$WORKDIR/c.out" "$WORKDIR/c.err" >&2
  exit 1
fi

needed_n="$(readelf -d "$OUT" 2>/dev/null | grep -c "Shared library:")"
if [[ "${needed_n:-0}" -lt 5 ]]; then
  echo "gate: FAIL expected >=5 DT_NEEDED, got ${needed_n:-0}" >&2
  readelf -d "$OUT" >&2 || true
  exit 1
fi

chmod +x "$OUT"
export LD_LIBRARY_PATH="$WORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
set +e
got="$("$OUT" 2>"$WORKDIR/run.err")"; rc=$?
set -e
if [[ $rc -ne 0 ]] || ! echo "$got" | grep -q "dynlink_caps_exceeded: PASS"; then
  echo "gate: FAIL run rc=$rc stdout=[$got]" >&2
  cat "$WORKDIR/run.err" >&2 || true
  exit 1
fi
echo "MADAROS_DYNLINK_CAPS_GATE_OK"
