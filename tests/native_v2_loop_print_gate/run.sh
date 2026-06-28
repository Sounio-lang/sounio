#!/usr/bin/env bash
# native_v2_loop_print_gate — guards two absorbed compiler fixes:
# - range for-in lowering emits working loop IR, including continue/break labels;
# - print/println dispatch routes numeric values to numeric builtins, not print_str.
#
# Anti-overclaim: every case asserts FILE-PRESENT plus exact ELF exit code.
# Print cases also execute the generated ELF; a bad print_str route historically
# dereferenced numeric values as pointers and crashed at runtime.
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
for src in "$DIR"/p[0-9]*.sio; do
  tot=$((tot + 1))
  n=$(basename "$src" .sio)
  exp="${EXP[$n]:-}"
  out="/tmp/loopprint_$n.elf"
  stdout="/tmp/loopprint_$n.out"
  rm -f "$out" "$stdout"
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  if [ -f "$out" ]; then
    chmod +x "$out"
    "$out" >"$stdout" 2>&1
    rc=$?
    if [ "$rc" = "$exp" ]; then
      echo "PASS  $n (exit $rc)"
      pass=$((pass + 1))
    else
      echo "WRONG $n (got exit $rc want $exp)"
      if [ -s "$stdout" ]; then
        sed -n '1,5p' "$stdout"
      fi
    fi
  else
    echo "FAIL  $n (no ELF emitted)"
  fi
done
echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
