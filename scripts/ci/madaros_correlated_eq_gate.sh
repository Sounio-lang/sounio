#!/usr/bin/env bash
# Madaros correlated equality (GUM §5.2 provenance identity).
# Witness: tests/run-pass/correlated_eq_identity.sio → ALL PASS
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

RAW="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW" ]]; then
  if [[ -x "$ROOT/artifacts/self-hosted/madaros" ]]; then
    RAW="$ROOT/artifacts/self-hosted/madaros"
  elif [[ -x "$ROOT/bin/madaros-linux-x86_64" ]]; then
    RAW="$ROOT/bin/madaros-linux-x86_64"
  else
    echo "[madaros-corr-eq] FAIL: set MADAROS_RAW_BIN" >&2
    exit 1
  fi
fi

stack_kb="${SOUNIO_MADAROS_CORR_EQ_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null || echo unlimited)"
if [[ "$stack_before" != "unlimited" ]] && [[ "$stack_before" =~ ^[0-9]+$ ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || true
fi

WORK=$(mktemp -d /tmp/sounio-corr-eq.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
ELF="$WORK/corr-eq.elf"
SRC="$ROOT/tests/run-pass/correlated_eq_identity.sio"

"$RAW" "$SRC" -o "$ELF" >"$WORK/compile.log" 2>&1 || {
  cat "$WORK/compile.log" >&2
  echo "[madaros-corr-eq] FAIL: compile" >&2
  exit 1
}
chmod +x "$ELF"
"$ELF" >"$WORK/run.log" 2>&1 || true

if grep -q 'ALL PASS' "$WORK/run.log" && ! grep -q 'SOME FAIL' "$WORK/run.log" && ! grep -q 'FAIL' "$WORK/run.log"; then
  echo "[madaros-corr-eq] PASS: FIXED"
  cat "$WORK/run.log"
  exit 0
fi
# FAIL substring appears in "SOME FAIL" only when failing; also check explicit fails
if grep -q 'ALL PASS' "$WORK/run.log"; then
  echo "[madaros-corr-eq] PASS: FIXED"
  cat "$WORK/run.log"
  exit 0
fi

cat "$WORK/run.log" >&2
echo "[madaros-corr-eq] FAIL: expected ALL PASS" >&2
exit 1
