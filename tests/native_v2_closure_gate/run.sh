#!/usr/bin/env bash
# No-capture closure gate for the modular native-v2 backend (--native-v2-compile).
#
# A no-capture lambda lifted to a synthetic top-level fn must run its body with the
# passed argument. Before the fix the param-registration loop stored param_count via
# a deref-store through the box FIELD `(*lo.current_func).param_count = v` which
# silently no-ops (documented c634b38f miscompile) -> param_count stayed 0 -> the
# codegen param spill (rdi->slot) was never emitted -> the closure body read its
# argument from an uninitialized slot, so f(41) returned ~1 instead of 42.
#
#   - p01 single-arg increment        -> exit 42
#   - p02 two-arg add (loop iterates)  -> exit 42 (re-read/re-box per param)
#   - p03 single-arg square            -> exit 49
#   - p04 zero-arg constant            -> exit 99 (loop never enters)
#   - r01 CAPTURING closure: REJECTED (no ELF) - capture is out of scope; it must
#         NOT silently mis-emit (free var would resolve against empty locals).
#
# Positive (p*) witnesses MUST emit an ELF and exit with the expected code.
# Reject  (r*) witnesses MUST NOT emit an ELF (anti-overclaim: file absence, not just rc).
set -u
BIN="${1:?usage: run.sh <souc-binary>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
declare -A EXP
while read -r name val; do
  [[ "$name" == \#* || -z "$name" ]] && continue
  EXP[$name]=$val
done < "$DIR/EXPECTED.txt"

pass=0; tot=0
for src in "$DIR"/[pr][0-9]*.sio; do
  tot=$((tot+1))
  n=$(basename "$src" .sio)
  out="/tmp/closgate_$n.elf"
  rm -f "$out"
  exp="${EXP[$n]:-}"
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  crc=$?
  if [ "$exp" = "REJECT" ]; then
    if [ -f "$out" ]; then
      echo "WRONG $n (REJECT expected but ELF emitted)"
    elif [ "$crc" -ge 128 ]; then
      # A compiler crash (e.g. SIGSEGV rc=139) also produces no ELF -- but it is NOT a
      # clean reject. Fail loudly so the lookup_fn_by_name materialization crash can't
      # masquerade as a pass.
      echo "WRONG $n (compiler crashed rc=$crc, expected clean reject)"
    else
      echo "PASS  $n (rejected cleanly, rc=$crc, no ELF)"; pass=$((pass+1))
    fi
  else
    if [ -f "$out" ]; then
      chmod +x "$out"; "$out" >/dev/null 2>&1; rc=$?
      if [ "$rc" = "$exp" ]; then
        echo "PASS  $n (exit $rc)"; pass=$((pass+1))
      else
        echo "WRONG $n (got $rc want $exp)"
      fi
    else
      echo "FAIL  $n (compile, no ELF)"
    fi
  fi
done
echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
