#!/usr/bin/env bash
# native_v2_tuple_signature_absorbed_gate — guards absorbed tuple-signature fixes:
# contextual int array literal binding, assignment RHS parsing, implicit unit
# returns, bool literals in native-v2, and assert builtin exit semantics.
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ulimit -s 1048576 2>/dev/null || true

declare -A EXP
while read -r name val; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  EXP[$name]=$val
done < "$DIR/EXPECTED_ELF.txt"

pass=0
tot=0

for src in "$DIR"/assert*.sio "$DIR"/unit_implicit_exit0.sio; do
  tot=$((tot + 1))
  n=$(basename "$src" .sio)
  exp="${EXP[$n]:-}"
  out="/tmp/tuple_sig_$n.elf"
  rm -f "$out"
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  if [ -f "$out" ]; then
    chmod +x "$out"
    "$out" >/dev/null 2>&1
    rc=$?
    if [ "$rc" = "$exp" ]; then
      echo "PASS  $n (exit $rc)"
      pass=$((pass + 1))
    else
      echo "WRONG $n (got exit $rc want $exp)"
    fi
  else
    echo "FAIL  $n (no ELF emitted)"
  fi
done

for src in "$DIR"/array_repeat_i8_binding.sio "$DIR"/assignment_rhs_*.sio; do
  tot=$((tot + 1))
  n=$(basename "$src" .sio)
  if "$BIN" --check "$src" >/dev/null 2>&1; then
    echo "PASS  $n (check OK)"
    pass=$((pass + 1))
  else
    echo "FAIL  $n (check failed)"
  fi
done

tot=$((tot + 1))
if "$BIN" --check "$DIR/array_repeat_i8_binding_type_mismatch.sio" >/tmp/tuple_sig_mismatch.out 2>&1; then
  echo "FAIL  array_repeat_i8_binding_type_mismatch (unexpected check OK)"
else
  if rg -q "this binding expects a different type" /tmp/tuple_sig_mismatch.out; then
    echo "PASS  array_repeat_i8_binding_type_mismatch (expected mismatch)"
    pass=$((pass + 1))
  else
    echo "FAIL  array_repeat_i8_binding_type_mismatch (wrong diagnostic)"
  fi
fi

echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
