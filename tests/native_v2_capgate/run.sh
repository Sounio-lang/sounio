#!/usr/bin/env bash
# Capability gate for modular source→native-ELF (--native-v2-compile).
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
declare -A EXP
while read -r name val; do [[ "$name" == \#* || -z "$name" ]] && continue; EXP[$name]=$val; done < "$DIR/EXPECTED.txt"
pass=0; tot=0
for src in "$DIR"/[0-9]*.sio; do
  tot=$((tot+1)); n=$(basename "$src" .sio); out="/tmp/capgate_$n.elf"; rm -f "$out"
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  if [ -f "$out" ]; then chmod +x "$out"; "$out" >/dev/null 2>&1; rc=$?
    if [ "$rc" = "${EXP[$n]}" ]; then echo "PASS  $n"; pass=$((pass+1)); else echo "WRONG $n (got $rc want ${EXP[$n]})"; fi
  else echo "FAIL  $n (compile)"; fi
done
echo "---- $pass/$tot ----"
