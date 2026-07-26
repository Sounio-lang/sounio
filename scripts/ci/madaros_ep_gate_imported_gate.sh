#!/usr/bin/env bash
# Madaros multimodule: i64 field-compare via call-arg boundary (ep_gate residual).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
WORK="${SOUNIO_MADAROS_EP_GATE_DIR:-$(mktemp -d /tmp/sounio-madaros-ep-gate.XXXXXX)}"
SOURCE="$ROOT_DIR/tests/multimodule/madaros_ep_gate_main.sio"
ELF="$WORK/ep-gate.elf"

fail() {
  echo "[madaros-ep-gate] FAIL: $*" >&2
  exit 1
}

[[ -n "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN must name a Madaros ELF"
[[ -x "$RAW_MADAROS" ]] || fail "Madaros ELF missing or not executable: $RAW_MADAROS"

stack_kb="${SOUNIO_MADAROS_EP_GATE_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null || echo unlimited)"
if [[ "$stack_before" != "unlimited" ]] && [[ "$stack_before" =~ ^[0-9]+$ ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || fail "could not raise soft stack to ${stack_kb} KiB"
fi

trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

"$RAW_MADAROS" --native-compile "$SOURCE" -o "$ELF" >"$WORK/compile.log" 2>&1 || {
  cat "$WORK/compile.log" >&2
  fail "ep_gate multimodule probe did not compile"
}
[[ -s "$ELF" ]] || fail "compiler did not emit ELF"
chmod +x "$ELF"

"$ELF" >"$WORK/run.log" 2>&1 || {
  cat "$WORK/run.log" >&2
  fail "ep_gate multimodule probe did not execute"
}
grep -Fxq 'MADAROS_EP_GATE_VIA_ARG_OK' "$WORK/run.log" || {
  cat "$WORK/run.log" >&2
  fail "via_arg gate marker missing"
}

echo "[madaros-ep-gate] PASS: imported i64 field compare via call-arg boundary"
