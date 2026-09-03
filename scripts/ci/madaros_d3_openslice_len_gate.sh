#!/usr/bin/env bash
# Madaros D3 open-slice / array `.len()` closeout.
# Historical: method lowering invented a body-less `len` fn → SIGSEGV at lower;
# println of the result also needed scalar_kind=1 for intrinsic `.len()`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
unset MADAROS_RAW_BIN || true
unset SOUNIO_MADAROS_BIN || true
SOUC="${SOUC:-$ROOT/bin/souc}"

echo "== madaros_d3_openslice_len_gate (local) =="
LOCAL_OUT="$(mktemp -d)"; trap 'rm -rf "$LOCAL_OUT" "$MM_OUT"' EXIT
LOCAL_ELF="$LOCAL_OUT/local.elf"
if ! "$SOUC" compile tests/epistemic_trust/madaros_d3_openslice_len_local.sio -o "$LOCAL_ELF" >"$LOCAL_OUT/compile.log" 2>&1; then
  echo "FAIL: local compile"
  tail -40 "$LOCAL_OUT/compile.log" || true
  exit 1
fi
chmod +x "$LOCAL_ELF"
if ! "$LOCAL_ELF" >"$LOCAL_OUT/run.log" 2>&1; then
  echo "FAIL: local run"
  cat "$LOCAL_OUT/run.log" || true
  exit 1
fi
grep -q 'D3_OPENSLICE_LEN_OK' "$LOCAL_OUT/run.log" || {
  echo "FAIL: missing local sentinel"
  cat "$LOCAL_OUT/run.log" || true
  exit 1
}

echo "== madaros_d3_openslice_len_gate (import) =="
MM_OUT="$(mktemp -d)"
MM_ELF="$MM_OUT/mm.elf"
if ! "$SOUC" compile tests/run-pass/d3_openslice_len/main.sio -o "$MM_ELF" >"$MM_OUT/compile.log" 2>&1; then
  echo "FAIL: import compile"
  tail -40 "$MM_OUT/compile.log" || true
  exit 1
fi
chmod +x "$MM_ELF"
if ! "$MM_ELF" >"$MM_OUT/run.log" 2>&1; then
  echo "FAIL: import run"
  cat "$MM_OUT/run.log" || true
  exit 1
fi
grep -Eq '^4$' "$MM_OUT/run.log" || {
  echo "FAIL: expected println 4 from import witness"
  cat "$MM_OUT/run.log" || true
  exit 1
}

echo "MADAROS_D3_OPENSLICE_LEN_GATE_OK"
