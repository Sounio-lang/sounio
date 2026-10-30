#!/usr/bin/env bash
# `&&` and `||` short-circuit in Madaros-compiled programs (ADR-014).
#
# Builds tests/run-pass/logical_ops_short_circuit.sio with the compiler under
# test and requires SHORT_CIRCUIT_OK. The witness counts calls on each side of
# `&&` / `||` and guards an out-of-range index behind a length check. Madaros
# before ADR-014 evaluated both operands and printed 101010 (three wrong call
# counts); lean_single already short-circuits. The Full Test Suite runs this
# file under lean_single; this gate runs it under the current-source Madaros.
#
# Usage: madaros_short_circuit_gate.sh   (MADAROS_BIN = compiler under test,
#                                         default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-short-circuit.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-short-circuit] FAIL: $*" >&2; exit 1; }
SRC="$ROOT_DIR/tests/run-pass/logical_ops_short_circuit.sio"
( ulimit -s 1572864 2>/dev/null || true; "$MADAROS" build "$SRC" -o "$WORK/w.elf" ) >"$WORK/build.log" 2>&1 \
  || { tail -n 20 "$WORK/build.log" >&2; fail "build failed"; }
chmod +x "$WORK/w.elf"
set +e
out="$("$WORK/w.elf" 2>&1)"
rc=$?
set -e
[[ "$rc" -eq 0 && "$out" == "SHORT_CIRCUIT_OK" ]] || fail "rc=$rc output=$(tr '\n' ' ' <<<"$out") (101010 = both operands evaluated)"
echo "[madaros-short-circuit] PASS: && and || evaluate the right operand only when the left one does not decide"
