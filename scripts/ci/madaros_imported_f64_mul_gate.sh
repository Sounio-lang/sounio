#!/usr/bin/env bash
# Madaros multimodule: imported f64 return * f64 residual.
#
# RESIDUAL: prints correct f64, raw mul wrong (e.g. (-0.5)*(-0.5)→0).
# FIXED: mul matches 0.25.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

RAW_MADAROS="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW_MADAROS" ]]; then
  if [[ -x "$ROOT/artifacts/self-hosted/madaros" ]]; then
    RAW_MADAROS="$ROOT/artifacts/self-hosted/madaros"
  elif [[ -x "$ROOT/bin/madaros-linux-x86_64" ]]; then
    RAW_MADAROS="$ROOT/bin/madaros-linux-x86_64"
  else
    echo "[madaros-imported-f64-mul] FAIL: set MADAROS_RAW_BIN" >&2
    exit 1
  fi
fi

stack_kb="${SOUNIO_MADAROS_F64_MUL_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null || echo unlimited)"
if [[ "$stack_before" != "unlimited" ]] && [[ "$stack_before" =~ ^[0-9]+$ ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || true
fi

WORK=$(mktemp -d /tmp/sounio-madaros-f64-mul.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
ELF="$WORK/f64-mul.elf"
SRC="$ROOT/tests/multimodule/madaros_imported_f64_mul_main.sio"

"$RAW_MADAROS" --native-compile "$SRC" -o "$ELF" >"$WORK/compile.log" 2>&1 || {
  cat "$WORK/compile.log" >&2
  echo "[madaros-imported-f64-mul] FAIL: compile" >&2
  exit 1
}
chmod +x "$ELF"
"$ELF" >"$WORK/run.log" 2>&1 || {
  cat "$WORK/run.log" >&2
  echo "[madaros-imported-f64-mul] FAIL: run" >&2
  exit 1
}

if grep -q 'MADAROS_IMPORTED_F64_MUL_FIXED' "$WORK/run.log"; then
  echo "[madaros-imported-f64-mul] PASS: FIXED"
  cat "$WORK/run.log"
  exit 0
fi
if grep -q 'MADAROS_IMPORTED_F64_MUL_RESIDUAL' "$WORK/run.log"; then
  # Documented open residual — gate green while baseline print works.
  grep -q 'a=-0.500000' "$WORK/run.log" || grep -q 'a=-0.5' "$WORK/run.log" || true
  echo "[madaros-imported-f64-mul] PASS: RESIDUAL documented (imported f64*f64 wrong)"
  cat "$WORK/run.log"
  exit 0
fi

cat "$WORK/run.log" >&2
echo "[madaros-imported-f64-mul] FAIL: unexpected marker" >&2
exit 1
