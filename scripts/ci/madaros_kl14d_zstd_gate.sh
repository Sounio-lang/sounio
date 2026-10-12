#!/usr/bin/env bash
# KL-14d2: Madaros round-trip against system libzstd.so.1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SRC=tests/run-pass/kl14d_zstd_e2e.sio
WORKDIR="${TMPDIR:-/tmp}/sounio-kl14d-zstd-$$"
mkdir -p "$WORKDIR"
trap 'rm -rf "$WORKDIR"' EXIT

OUT="$WORKDIR/kl14d_zstd_e2e.elf"
SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "madaros_kl14d_zstd_gate: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || {
  echo "madaros_kl14d_zstd_gate: set MADAROS_RAW_BIN to a Madaros ELF" >&2
  exit 2
}

# Prefer this checkout's stdlib so wrappers actually call ZSTD_* (emit DT_NEEDED).
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

# Resolve libzstd on the runner.
if ! ldconfig -p 2>/dev/null | grep -q 'libzstd\.so\.1'; then
  if [[ ! -e /usr/lib/x86_64-linux-gnu/libzstd.so.1 && ! -e /lib/x86_64-linux-gnu/libzstd.so.1 ]]; then
    echo "madaros_kl14d_zstd_gate: libzstd.so.1 not found on runner" >&2
    exit 2
  fi
fi

if ! MADAROS_RAW_BIN="$RAW" "$SOUC" build "$SRC" -o "$OUT" >"$WORKDIR/compile.out" 2>"$WORKDIR/compile.err"; then
  echo "madaros_kl14d_zstd_gate: FAIL compile" >&2
  cat "$WORKDIR/compile.out" "$WORKDIR/compile.err" >&2
  exit 1
fi

dyn="$(readelf -d "$OUT" 2>/dev/null || true)"
echo "$dyn" | grep -q 'Shared library: \[libzstd\.so\.1\]' || {
  echo "madaros_kl14d_zstd_gate: FAIL missing DT_NEEDED libzstd.so.1" >&2
  echo "$dyn" >&2
  exit 1
}

chmod +x "$OUT"
set +e
got="$("$OUT" 2>"$WORKDIR/run.err")"
rc=$?
set -e
if [[ $rc -ne 0 ]] || ! echo "$got" | grep -q 'kl14d_zstd_e2e: PASS'; then
  echo "madaros_kl14d_zstd_gate: FAIL run rc=$rc stdout=[$got]" >&2
  cat "$WORKDIR/run.err" >&2 || true
  readelf -d "$OUT" >&2 || true
  exit 1
fi

echo "MADAROS_KL14D_ZSTD_GATE_OK"
