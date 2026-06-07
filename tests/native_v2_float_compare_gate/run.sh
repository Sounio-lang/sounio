#!/usr/bin/env bash
# native_v2_float_compare_gate — guards the M1 f32-comparison silent miscompile.
#
# THE BUG (a6739047c): `let x:f32 = 1.5 as f32` lowered the `as f32` cast to
# IrFloatToInt (cvttsd2si) because the cast-type matcher only recognized f64.
# That truncated the float value to an integer (1.5 -> 1), dropped the float flag,
# and the downstream `x > y` ran as an INTEGER compare (cmp/setl) of 1 vs 1 -> 0.
# Witness `1.5 as f32 > 1.0 as f32` ran 0 instead of 12.
#
# THE FIX (self-hosted/ir/lower.sio + native/codegen_x86_linux.sio): `as f32` is
# now treated as a FLOAT cast like `as f64` (the backend stores both as f64), and
# IrIntToFloat passes a float source through unchanged (is_float_reg guard) so the
# value/flag survive into a ucomisd compare.
#
# Anti-overclaim (C0/R4): every positive asserts FILE-PRESENT + the EXACT exit
# code of the RUN ELF. p01 is the M1 witness (12). p05/p06 are int->float
# regression guards (the shared IrIntToFloat path must still convert ints).
# Prints N/N; nonzero exit on any miss.
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
  out="/tmp/floatcmp_$n.elf"
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
