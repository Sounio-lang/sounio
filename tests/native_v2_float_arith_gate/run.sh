#!/usr/bin/env bash
# native_v2_float_arith_gate — guards the both-memory-operand float-arithmetic miscompile.
#
# THE BUG: float arithmetic between TWO memory-loaded operands (two f64/f32 params, two
# float array elements) miscompiled to 0. The backend's binop float-detection only sees
# regs flagged by IrLoadFloat; a float operand that originates from a param or an array
# element carries no such flag, so the binop took the INTEGER path and added the raw f64
# bit patterns as integers (-> garbage -> `as i64` = 0). The memory+literal case
# (a + 1.0) worked because the literal forced the float path.
#
# THE FIX (self-hosted/ir/lower.sio + ir/ir.sio + native/codegen_x86_linux.sio): the
# lowerer tracks float-typed locals/params/array-elements (LowerLocalStack.is_float, seeded
# from f32/f64 type annotations) and tags the IrBinOp (imm_flags bit 1) when its operands
# are float; the backend routes a tagged binop to the SSE path. Strictly type-conditioned,
# so integer arithmetic is never mis-routed to SSE (p09-p11 integer controls).
#
# Anti-overclaim (C0/R3/R4): every case asserts FILE-PRESENT + the EXACT exit code of the
# RUN ELF. p08 is the memory+literal control (must stay 4). p09-p11 are integer controls
# (must stay on the integer path with correct integer results). Prints N/N; nonzero on miss.
#
# KNOWN RESIDUAL (DEFER): struct FIELD float arithmetic (p.x + p.y) is NOT fixed here — the
# lowerer tracks no local struct types and the field read site uses a type-free name hash
# (see lower.sio field_idx_from_name_simple), and StructFieldEntry carries no type, so the
# field's f64-ness is not recoverable without a larger type-threading change. Not gated.
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ulimit -s 1048576 2>/dev/null || true

declare -A EXP
while read -r name val; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  EXP[$name]=$val
done < "$DIR/EXPECTED.txt"

pass=0; tot=0
for src in "$DIR"/p[0-9]*.sio; do
  tot=$((tot+1))
  n=$(basename "$src" .sio)
  exp="${EXP[$n]:-}"
  out="/tmp/floatarith_$n.elf"
  rm -f "$out"
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  if [ -f "$out" ]; then
    chmod +x "$out"; "$out" >/dev/null 2>&1; rc=$?
    if [ "$rc" = "$exp" ]; then
      echo "PASS  $n (exit $rc)"; pass=$((pass+1))
    else
      echo "WRONG $n (got exit $rc want $exp)"
    fi
  else
    echo "FAIL  $n (no ELF emitted)"
  fi
done
echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
