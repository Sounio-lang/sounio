#!/usr/bin/env bash
# Madaros: i64 field named `value` must not inherit Knowledge.value float typing.
# Witness: tests/run-pass/method_receiver_correct.sio
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
    echo "[madaros-method-receiver] FAIL: set MADAROS_RAW_BIN" >&2
    exit 1
  fi
fi

stack_kb="${SOUNIO_MADAROS_METHOD_RECV_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null || echo unlimited)"
if [[ "$stack_before" != "unlimited" ]] && [[ "$stack_before" =~ ^[0-9]+$ ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || true
fi

WORK=$(mktemp -d /tmp/sounio-method-recv.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
ELF="$WORK/method-recv.elf"
SRC="$ROOT/tests/run-pass/method_receiver_correct.sio"

"$RAW" "$SRC" -o "$ELF" >"$WORK/compile.log" 2>&1 || {
  cat "$WORK/compile.log" >&2
  echo "[madaros-method-receiver] FAIL: compile" >&2
  exit 1
}
chmod +x "$ELF"
"$ELF" >"$WORK/run.log" 2>&1
st=$?
if grep -q 'method_receiver_correct: PASS' "$WORK/run.log" && [[ $st -eq 0 ]]; then
  echo "[madaros-method-receiver] PASS: FIXED"
  cat "$WORK/run.log"
  exit 0
fi
cat "$WORK/run.log" >&2
echo "[madaros-method-receiver] FAIL: expected PASS (exit 0)" >&2
exit 1
