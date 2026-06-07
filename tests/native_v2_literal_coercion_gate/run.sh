#!/usr/bin/env bash
# Numeric-literal expected-type coercion gate.
#
# Proves the release lever (self-hosted/check/{check,compat}.sio): a bare numeric
# LITERAL adopts the expected numeric type from context, WITHOUT opening the
# soundness hole of "any numeric mismatch is accepted".
#
# POSITIVES (p*) — newly-accepted literal programs that compile (--native-v2-compile,
#   which runs the typechecker) and RUN to the correct value (R1: the literal lowers
#   at the COERCED width). p06 is the width-observable witness: an i32 field set to a
#   literal, read back through field-get and returned directly (exit 77) — if the
#   literal were silently kept i64 or truncated, this would not read back 77.
#     p01 literal in return position (fn ->i32 {42})
#     p02 typed let (let x: i32 = 42)
#     p03 struct field initializers (P{x:7,y:35})
#     p04 fn-arg literal (add(40,2))
#     p05 unary-neg-of-literal in return position (fn ->i32 {-42}; exit 214 = -42 & 0xFF)
#     p06 WIDTH-OBSERVABLE i32 field literal read back (exit 77)
#     p07 typed-let f64 literal (float-literal path)
#
# REJECTS (r*) — C0 twins. Each MUST still FAIL --check; a coercion that accepts
#   any of these is a SOUNDNESS HOLE worse than the false-reject this fix removes.
#     r01 non-literal var of wrong type in a typed slot (let y:i64=5; let x:i32=y)
#     r02 returning an i64 VAR from an i32 fn (block-tail ident, not a literal)
#     r03 literal -> bool   (non-numeric target)
#     r04 literal -> string (non-numeric target)
#     r05 out-of-range narrow literal u8 = 256  (R1 value-range guard)
#     r06 out-of-range narrow literal i8 = 128  (R1 value-range guard)
#     r07 STALE-FLAG twin: call with a literal arg returning i64 from an i32 fn
#         (the call node must clear the per-expr literal flag -> still rejects)
#     r08 non-literal arg of wrong width to a typed param (arg coercion must not
#         leak to vars: take(y) where y:i64, param i32)
#
# Anti-overclaim: positives assert FILE-PRESENT + exact exit code; rejects assert
# --check FAILS (rc!=0). Prints N/N; nonzero exit on any miss.
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
    # Must be rejected at type-check time (no over-accept).
    "$BIN" --check "$src" >/dev/null 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "PASS  $n (rejected by --check, rc=$rc)"; pass=$((pass+1))
    else
      echo "WRONG $n (REJECT expected but --check accepted it)"
    fi
  else
    out="/tmp/litcoerce_$n.elf"
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
  fi
done
echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
