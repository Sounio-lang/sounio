#!/usr/bin/env bash
# #2537: SysV callee-alignment witness for the Madaros dynlink stub.
#
# 41fae115e replaced the dynlink stub `call [rip+GOT]; ret` with
# `jmp [rip+GOT]`, so a foreign callee is entered with the SysV alignment
# (return-address slot == 8 mod 16). This gate:
#   1. builds tests/fixtures/kl14d/lib_align.c (explicit entry-alignment check
#      + movaps round trip through an aligned stack buffer);
#   2. compiles and runs tests/run-pass/sysv_alignment_thunk_witness.sio with
#      the shipped stub: it must print PASS and exit 0;
#   3. rebuilds it with SOUNIO_SABOTAGE_DYNLINK_THUNK_CALLRET=1 (the old
#      call+ret stub): it must FAIL. A witness that passes with the stub it
#      guards against measures nothing.
# The OK line is printed only after all three steps succeeded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TAG="madaros_sysv_alignment_thunk_gate"
fail() { echo "$TAG: FAIL $*" >&2; exit 1; }

SRC=tests/run-pass/sysv_alignment_thunk_witness.sio
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-sysv-align.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

SOUC="${SOUC:-$ROOT/bin/souc}"
RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-${MADAROS_BIN:-}}}"
[[ -x "$SOUC" ]] || { echo "$TAG: souc not executable: $SOUC" >&2; exit 2; }
[[ -n "$RAW" && -x "$RAW" ]] || { echo "$TAG: set MADAROS_RAW_BIN to a Madaros ELF" >&2; exit 2; }
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SABOTAGE_DYNLINK_THUNK_CALLRET

gcc -shared -fPIC -O0 -o "$WORKDIR/libkl14d_align.so" tests/fixtures/kl14d/lib_align.c \
  || fail "could not build tests/fixtures/kl14d/lib_align.c"
export LD_LIBRARY_PATH="$WORKDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

build() {  # <out> [env assignments...]
  local out="$1"; shift
  env "$@" MADAROS_RAW_BIN="$RAW" "$SOUC" build "$SRC" -o "$out" >"$out.log" 2>&1 \
    || { cat "$out.log" >&2; fail "compile failed ($*)"; }
  chmod +x "$out"
}
run() {  # <elf> -> sets RUN_RC, RUN_OUT
  RUN_RC=0
  RUN_OUT="$(timeout 60 "$1" 2>"$1.err")" || RUN_RC=$?
}

# ---- shipped stub: must pass
build "$WORKDIR/witness.elf"
run "$WORKDIR/witness.elf"
if [[ $RUN_RC -ne 0 ]] || ! grep -qxF 'sysv_alignment_thunk: PASS' <<<"$RUN_OUT"; then
  cat "$WORKDIR/witness.elf.err" >&2 || true
  fail "shipped dynlink stub: rc=$RUN_RC stdout=[$RUN_OUT]"
fi
echo "$TAG: PASS shipped stub (jmp [rip+GOT]): $RUN_OUT"

# ---- old call+ret stub: must fail
build "$WORKDIR/sabotaged.elf" SOUNIO_SABOTAGE_DYNLINK_THUNK_CALLRET=1
run "$WORKDIR/sabotaged.elf"
if [[ $RUN_RC -eq 0 ]] || grep -qxF 'sysv_alignment_thunk: PASS' <<<"$RUN_OUT"; then
  fail "witness passed with the call+ret stub (rc=$RUN_RC stdout=[$RUN_OUT]): it does not detect a misaligned callee"
fi
echo "$TAG: PASS call+ret stub is caught: rc=$RUN_RC stdout=[$RUN_OUT]"

echo "MADAROS_SYSV_ALIGNMENT_THUNK_GATE_OK"
