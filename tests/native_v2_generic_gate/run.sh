#!/usr/bin/env bash
# native_v2_generic_gate — guards generic functions (`fn f<T>(...) -> T`).
#
# THE GAP: the parser parsed the `<T>` header and DISCARDED it; the checker then saw `T` as
# an unknown named type, so a call `id(42)` failed (E009 param T vs arg i64) and using the
# result failed (E008 caller-return i64 vs the uninferred T). Generic fns were unusable.
#
# THE FIX (parser + checker only; lowering is type-erased and unchanged):
#   - parser stores `<T>` on FnDef.type_params (parser/items.sio, ast.sio).
#   - the sig records type_param_count/names (FnSig, both the *mut and by-value collectors).
#   - at the call site the type params are INFERRED from the argument types with CONSISTENCY
#     checks, and the return type is substituted with the inferred bindings.
# Type-erasure is sound only for 8-byte GP-class values, so wide-int (i128/u128) and float
# (f32/f64) instantiations are REJECTED rather than miscompiled.
#
# Anti-overclaim / soundness discriminators (mirror the closure gate):
#   p01-p05  positives: must emit ELF and exit with the expected code (p03 exercises erasure
#            across a 2nd type via a string; p05 a concrete+generic param mix).
#   r01      inconsistent T (i64 vs string) MUST reject -- proves REAL binding, not the
#            "unknown type accepts anything" wildcard shortcut.
#   r02      the generic RESULT used ill-typed (id(5):i64 into a string) MUST reject -- proves
#            the return type is substituted to a concrete type, not left unknown.
#   r03/r04  wide-int / float instantiation MUST reject (erasure unsound at non-GP width) --
#            a clean --check failure, never a wrong-answer ELF.
# Rejects assert --check FAILS (rc in 1..127: a real diagnostic, NOT a compiler crash >=128).
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
for src in "$DIR"/[pr][0-9]*.sio; do
  tot=$((tot+1))
  n=$(basename "$src" .sio)
  exp="${EXP[$n]:-}"
  if [ "$exp" = "REJECT" ]; then
    "$BIN" --check "$src" >/dev/null 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "WRONG $n (REJECT expected but --check accepted)"
    elif [ "$rc" -ge 128 ]; then
      echo "WRONG $n (compiler crashed rc=$rc, expected clean reject)"
    else
      echo "PASS  $n (rejected cleanly, rc=$rc)"; pass=$((pass+1))
    fi
  else
    out="/tmp/gengate_$n.elf"; rm -f "$out"
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
  fi
done
echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
