#!/usr/bin/env bash
# Enum lowering gate for the modular native-v2 backend (--native-v2-compile).
#
# Covers tagged-enum struct-variant construction, match dispatch, and payload binding
# via a tagged-aggregate representation (discriminant at reserved slot 64), plus the
# soundness rejections that keep the representation honest:
#   - p01 struct-variant construct+match -> exit 42
#   - p02 MANDATORY mixed-enum UNIT collision trap: E::A of an enum that ALSO has a
#         struct variant must dispatch correctly (proves unit variants are boxed
#         uniformly, not left bare-int while the struct variant is a pointer). exit 7.
#   - p03 pure all-unit enum match (protects the bare-int path the compiler relies on)
#   - p04 all-unit == (must NOT be false-rejected)  - p05 all-unit `as i64` (discriminant)
#   - p06 struct variant as fn param + 2nd-arm dispatch (call ABI)
#   - p07 struct variant returned from fn then matched (return ABI)
#   - p08 plain struct unaffected (full-path keying)
#   - r01 tuple-variant ctor: REJECTED (no ELF) - at base SIGSEGV'd (R1 closure)
#   - r02/r03 == on a tagged (payload) enum: REJECTED (boxed-pointer compare would miscompile)
#   - r04 `as i64` on a tagged enum: REJECTED (would yield pointer not discriminant)
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
  out="/tmp/enumgate_$n.elf"
  rm -f "$out"
  exp="${EXP[$n]:-}"
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  if [ "$exp" = "REJECT" ]; then
    if [ -f "$out" ]; then
      echo "WRONG $n (REJECT expected but ELF emitted)"
    else
      echo "PASS  $n (rejected, no ELF)"; pass=$((pass+1))
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
