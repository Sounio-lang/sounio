#!/usr/bin/env bash
# #1161: `madaros --ir-roundtrip` / `--ir-dump` must either prove a real SOIR
# round trip or refuse with a reason -- never print OK over a lossy encoding.
#
# Before #1161 both modes ran a second writer in compiler/main.sio (version
# byte 2, 40 opcodes, no call arguments / imm_flags / function fields past
# prof_counter_id) and "round-tripped" by byte-copying the buffer and comparing
# the copy with itself, so programs with globals or extern calls printed
# "ir-roundtrip: OK". Both modes now run the real SOIR writer and reader,
# ir/serialize.sio (SOIR v6): a preflight refuses, by name, what v6 does not
# carry, and the round trip decodes the image, re-encodes the decoded module
# and requires the two images to be byte-identical.
#
#   1. tests/fixtures/soir/v2_representable.sio and v6_calls_and_lets.sio (call
#      arguments, integer lets, an extern call -- carried by v6, not by the old
#      v2 writer): rc 0, "ir-roundtrip: OK"; and --ir-dump rc 0.
#   2. tests/fixtures/soir/globals_extern_call.sio (BSS global + extern f64
#      call) and tests/fixtures/soir/int_calls_roundtrip.sio (named return
#      types): nonzero rc, "SOIR v6 cannot represent <reason>", no "OK" -- for
#      both modes.
#   3. Mutation: SOUNIO_SABOTAGE_IR_ROUNDTRIP=1 flips one bit of the image that
#      is decoded (the first function name's first byte); --ir-roundtrip must
#      report that the decoded module re-encodes differently and fail, so the
#      verifier is shown to be able to fail.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
TAG="madaros_ir_roundtrip_gate"
fail() { echo "$TAG: FAIL $*" >&2; exit 1; }

RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-}}"
[[ -n "$RAW" && -x "$RAW" ]] || { echo "$TAG: set MADAROS_RAW_BIN to a Madaros ELF" >&2; exit 2; }
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-ir-roundtrip.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

raw() (  # same stack reservation as bin/madaros
  ulimit -s "${MADAROS_STACK_KB:-524288}" 2>/dev/null || ulimit -s unlimited 2>/dev/null || true
  exec timeout 300 "$RAW" "$@"
)

expect_ok() {  # <mode> <file>
  local rc=0
  raw "$1" "$2" >"$WORK/out" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then tail -n 20 "$WORK/out" >&2; fail "$1 $2 exited $rc, expected 0"; fi
  if [[ "$1" == "--ir-roundtrip" ]]; then
    grep -q "^ir-roundtrip: OK bytes=" "$WORK/out" || { tail -n 20 "$WORK/out" >&2; fail "$1 $2 printed no OK line"; }
  fi
  echo "$TAG: PASS $1 $2: $(grep -m1 '^ir-' "$WORK/out")"
}

expect_refused() {  # <mode> <file>
  local rc=0
  raw "$1" "$2" >"$WORK/out" 2>&1 || rc=$?
  if [[ $rc -eq 0 ]] || grep -q "^ir-roundtrip: OK\|^ir-dump: bytes=" "$WORK/out"; then
    tail -n 20 "$WORK/out" >&2
    fail "$1 $2 exited $rc and/or claimed success; SOIR cannot represent this program"
  fi
  grep -q "SOIR v6 cannot represent" "$WORK/out" || { tail -n 20 "$WORK/out" >&2; fail "$1 $2 refused without naming what SOIR cannot represent"; }
  echo "$TAG: PASS $1 $2 refused (rc=$rc): $(grep -m1 'SOIR v6 cannot represent' "$WORK/out")"
}

expect_ok --ir-roundtrip tests/fixtures/soir/v2_representable.sio
expect_ok --ir-dump tests/fixtures/soir/v2_representable.sio
expect_ok --ir-roundtrip tests/fixtures/soir/v6_calls_and_lets.sio
expect_ok --ir-dump tests/fixtures/soir/v6_calls_and_lets.sio
# Mutation: with one encoded byte flipped the decode must report a mismatch.
rc=0
SOUNIO_SABOTAGE_IR_ROUNDTRIP=1 raw --ir-roundtrip tests/fixtures/soir/v2_representable.sio >"$WORK/out" 2>&1 || rc=$?
if [[ $rc -eq 0 ]] || grep -q "^ir-roundtrip: OK" "$WORK/out" || ! grep -q "the decoded module re-encodes differently" "$WORK/out"; then
  tail -n 20 "$WORK/out" >&2
  fail "sabotaged image (first function name flipped) was not caught by the round trip (rc=$rc)"
fi
echo "$TAG: PASS mutation: flipped name byte caught (rc=$rc): $(grep -m1 're-encodes differently' "$WORK/out")"

expect_refused --ir-roundtrip tests/fixtures/soir/globals_extern_call.sio
expect_refused --ir-dump tests/fixtures/soir/globals_extern_call.sio
expect_refused --ir-roundtrip tests/fixtures/soir/int_calls_roundtrip.sio
echo "MADAROS_IR_ROUNDTRIP_GATE_OK"
