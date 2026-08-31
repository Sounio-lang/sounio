#!/usr/bin/env bash
# Madaros FO cross-function variance: gum_cross_function.sio must PASS with
# var(sum)=5 and var(scaled)=16 (match lean_single).
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
    echo "[madaros-gum-cross] FAIL: set MADAROS_RAW_BIN" >&2
    exit 1
  fi
fi

stack_kb="${SOUNIO_MADAROS_GUM_CROSS_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null || echo unlimited)"
if [[ "$stack_before" != "unlimited" ]] && [[ "$stack_before" =~ ^[0-9]+$ ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || true
fi

WORK=$(mktemp -d /tmp/sounio-gum-cross.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
ELF="$WORK/gum-cross.elf"
SRC="$ROOT/tests/run-pass/gum_cross_function.sio"

"$RAW" "$SRC" -o "$ELF" >"$WORK/compile.log" 2>&1 || {
  cat "$WORK/compile.log" >&2
  echo "[madaros-gum-cross] FAIL: compile" >&2
  exit 1
}
chmod +x "$ELF"
"$ELF" >"$WORK/run.log" 2>&1 || {
  cat "$WORK/run.log" >&2
  echo "[madaros-gum-cross] FAIL: run" >&2
  exit 1
}

if grep -q '^PASS' "$WORK/run.log" \
  && grep -q 'var(sum)=5' "$WORK/run.log" \
  && grep -q 'var(scaled)=16' "$WORK/run.log"; then
  echo "[madaros-gum-cross] PASS: FIXED"
  cat "$WORK/run.log"
  exit 0
fi

cat "$WORK/run.log" >&2
echo "[madaros-gum-cross] FAIL: expected PASS with sum=5 scaled=16" >&2
exit 1
