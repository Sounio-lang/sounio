#!/usr/bin/env bash
# Madaros multimodule residual: i64 struct-field used in if/sub vs return/call-arg.
#
# Current expected Madaros behaviour (residual open):
#   ret/add0 correct; gate_field/gate_let == 0; sub garbage; via_arg/via_ret == 1
# When native field-if is fixed, the gate upgrades to require FIXED marker.
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
    echo "[madaros-field-if-i64] FAIL: set MADAROS_RAW_BIN" >&2
    exit 1
  fi
fi

stack_kb="${SOUNIO_MADAROS_FIELD_IF_STACK_KB:-524288}"
stack_before="$(ulimit -S -s 2>/dev/null || echo unlimited)"
if [[ "$stack_before" != "unlimited" ]] && [[ "$stack_before" =~ ^[0-9]+$ ]] && ((stack_before < stack_kb)); then
  ulimit -S -s "$stack_kb" 2>/dev/null || true
fi

WORK=$(mktemp -d /tmp/sounio-madaros-field-if.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
ELF="$WORK/field-if.elf"
SRC="$ROOT/tests/multimodule/madaros_field_if_i64_main.sio"

"$RAW_MADAROS" --native-compile "$SRC" -o "$ELF" >"$WORK/compile.log" 2>&1 || {
  cat "$WORK/compile.log" >&2
  echo "[madaros-field-if-i64] FAIL: compile" >&2
  exit 1
}
chmod +x "$ELF"
"$ELF" >"$WORK/run.log" 2>&1 || {
  cat "$WORK/run.log" >&2
  echo "[madaros-field-if-i64] FAIL: run" >&2
  exit 1
}

if grep -q 'MADAROS_FIELD_IF_I64_FIXED' "$WORK/run.log"; then
  echo "[madaros-field-if-i64] PASS: FIXED (native field-if closed)"
  exit 0
fi
if grep -q 'MADAROS_FIELD_IF_I64_RESIDUAL' "$WORK/run.log"; then
  # Residual is documented-open: gate still PASSes while workarounds hold.
  # Fail hard only if via_arg baseline regresses.
  grep -q 'via_arg=1' "$WORK/run.log" || {
    cat "$WORK/run.log" >&2
    echo "[madaros-field-if-i64] FAIL: via_arg baseline broken" >&2
    exit 1
  }
  grep -q 'via_ret=1' "$WORK/run.log" || {
    cat "$WORK/run.log" >&2
    echo "[madaros-field-if-i64] FAIL: via_ret baseline broken" >&2
    exit 1
  }
  echo "[madaros-field-if-i64] PASS: RESIDUAL documented (gate_field broken; via_arg holds)"
  cat "$WORK/run.log"
  exit 0
fi

cat "$WORK/run.log" >&2
echo "[madaros-field-if-i64] FAIL: unexpected marker" >&2
exit 1
