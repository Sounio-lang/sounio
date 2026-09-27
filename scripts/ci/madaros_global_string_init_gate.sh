#!/usr/bin/env bash
# Gate: Madaros native-v2 module-level string literal BSS init (Wave15e).
#
# Residual closed: `let S: string = "hi"; println(S)` used to SEGV because the
# global-init side table only carried i64/f64 words, leaving BSS null and omitting
# the literal from rodata. Fix: GLOBAL_STR_INIT_* → BSS_INIT_STRING_MAGIC + LEA.
#
# Requires current-source Madaros (stock prebuilt before Wave15e fails this gate).
# Usage:
#   bash scripts/ci/madaros_global_string_init_gate.sh
#   SOUC=./bin/souc bash scripts/ci/madaros_global_string_init_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
FIXTURE="$ROOT/tests/run-pass/global_string_lit_init.sio"
OUT_DIR="$(mktemp -d /tmp/madaros_global_string_init.XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

echo "== madaros_global_string_init_gate =="
echo "souc=$SOUC"
"$SOUC" --version 2>&1 | head -1 || true

echo "[gate] compile $FIXTURE"
"$SOUC" compile "$FIXTURE" -o "$OUT_DIR/global_string.elf" >"$OUT_DIR/compile.log" 2>&1
if [[ ! -x "$OUT_DIR/global_string.elf" ]]; then
  echo "FAIL: no ELF" >&2
  tail -40 "$OUT_DIR/compile.log" >&2
  exit 1
fi

# Rodata must contain the payloads (pre-fix omitted them entirely).
# `strings` defaults to min length 4 — use -n 2 for short literals like "hi".
if ! strings -n 2 "$OUT_DIR/global_string.elf" | grep -qx 'hi'; then
  echo "FAIL: rodata missing 'hi'" >&2
  strings -n 2 "$OUT_DIR/global_string.elf" | head -20 >&2
  exit 1
fi
if ! strings -n 2 "$OUT_DIR/global_string.elf" | grep -qx 'yo'; then
  echo "FAIL: rodata missing 'yo'" >&2
  exit 1
fi

set +e
out="$("$OUT_DIR/global_string.elf" 2>"$OUT_DIR/run.err")"
rc=$?
set -e
echo "[gate] run_rc=$rc"
echo "[gate] stdout:"
printf '%s\n' "$out"

if [[ $rc -ne 0 ]]; then
  echo "FAIL: runtime rc=$rc (expected 0; pre-fix SEGV=139)" >&2
  cat "$OUT_DIR/run.err" >&2 || true
  exit 1
fi

# Four lines: PREFIX, MUTABLE, get_prefix(), PREFIX ++ "!"
mapfile -t lines <<<"$out"
if [[ ${#lines[@]} -lt 4 ]]; then
  echo "FAIL: expected ≥4 stdout lines, got ${#lines[@]}" >&2
  exit 1
fi
if [[ "${lines[0]}" != "hi" ]]; then
  echo "FAIL: line0 expected 'hi', got '${lines[0]}'" >&2
  exit 1
fi
if [[ "${lines[1]}" != "yo" ]]; then
  echo "FAIL: line1 expected 'yo', got '${lines[1]}'" >&2
  exit 1
fi
if [[ "${lines[2]}" != "hi" ]]; then
  echo "FAIL: line2 expected 'hi', got '${lines[2]}'" >&2
  exit 1
fi
if [[ "${lines[3]}" != "hi!" ]]; then
  echo "FAIL: line3 expected 'hi!', got '${lines[3]}'" >&2
  exit 1
fi

echo "MADAROS_GLOBAL_STRING_INIT_GATE_OK"
