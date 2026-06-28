#!/usr/bin/env bash
# native_v2_global_bss_gate — guards absorbed root2 global BSS lowering fixes.
#
# The historical failure mode was a lowerer/module copy hazard while resolving
# top-level BSS globals during body lowering. Every case asserts FILE-PRESENT
# plus exact ELF exit code after native-v2 compile.
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
ulimit -s 1048576 2>/dev/null || true

declare -A EXP
while read -r name val; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  EXP[$name]=$val
done < "$DIR/EXPECTED.txt"

pass=0
tot=0
for src in "$DIR"/g[0-9]*.sio; do
  tot=$((tot + 1))
  n=$(basename "$src" .sio)
  exp="${EXP[$n]:-}"
  out="/tmp/globalbss_$n.elf"
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
echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
