#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
WORK="${SOUNIO_MADAROS_CALL_ARITY_13_DIR:-$(mktemp -d /tmp/sounio-madaros-call-arity-13.XXXXXX)}"
SOURCE="$ROOT_DIR/tests/multimodule/madaros_imported_call_arity_13_main.sio"
ELF="$WORK/call-arity-13.elf"

fail() {
  echo "[madaros-call-arity-13] FAIL: $*" >&2
  exit 1
}

[[ -n "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN must name a current-source Madaros ELF"
[[ -x "$RAW_MADAROS" ]] || fail "Madaros ELF is missing or not executable: $RAW_MADAROS"

# Soft stack for the Madaros *compiler process* (not the witness). After FO GUM
# multi-channel growth, 128 MiB (131072 KiB) is insufficient on GitHub runners
# (SEGV / call-arg scratch overflow during imported lower). Measured 2026-07-26:
# 262144 KiB passes; 131072 fails. Default 512 MiB for headroom.
stack_kb="${SOUNIO_MADAROS_CALL_ARITY_13_STACK_KB:-524288}"
[[ "$stack_kb" =~ ^[1-9][0-9]*$ && ${#stack_kb} -le 9 ]] || fail "invalid stack size: $stack_kb"
stack_before="$(ulimit -S -s 2>/dev/null)" || fail "soft stack limit is unavailable"
[[ "$stack_before" == "unlimited" || "$stack_before" =~ ^[0-9]+$ ]] || fail "invalid soft stack limit: $stack_before"
if [[ "$stack_before" != "unlimited" ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || fail "could not raise soft stack limit to ${stack_kb} KiB"
fi
stack_after="$(ulimit -S -s 2>/dev/null)" || fail "soft stack limit is unavailable after update"
[[ "$stack_after" == "unlimited" || "$stack_after" =~ ^[0-9]+$ ]] || fail "invalid updated soft stack limit: $stack_after"
if [[ "$stack_after" != "unlimited" ]] && ((stack_after < stack_kb)); then
  fail "soft stack limit remained below ${stack_kb} KiB: $stack_after"
fi
echo "[madaros-call-arity-13] stack_kb before=$stack_before after=$stack_after requested=$stack_kb"

trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
"$RAW_MADAROS" --native-compile "$SOURCE" -o "$ELF" >"$WORK/compile.log" 2>&1 || {
  cat "$WORK/compile.log" >&2
  fail "imported 13-argument witness did not compile"
}
[[ -s "$ELF" ]] || fail "compiler did not emit an ELF"
chmod +x "$ELF"

"$ELF" >"$WORK/run.log" 2>&1 || {
  cat "$WORK/run.log" >&2
  fail "imported 13-argument witness did not execute"
}
grep -Fxq 'MADAROS_IMPORTED_CALL_ARITY_13_OK' "$WORK/run.log" || {
  cat "$WORK/run.log" >&2
  fail "runtime marker is missing"
}

echo "[madaros-call-arity-13] PASS: imported calls preserved 7, 8, 9, and 13 ordered arguments"
