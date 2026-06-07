#!/usr/bin/env bash
# native_v2_string_gate — guards string `.len()` and string `+` (concat) lowering.
#
# THE GAP: the native-v2 backend already had working `str_len` / `str_concat` builtins
# (emit_builtin_str_len_into / emit_builtin_str_concat_into, dispatched by name), and the
# checker already typed `s.len()` -> i64 and string `a + b` -> string. But the LOWERER did
# not bridge the surface syntax to those builtins:
#   - `s.len()` was an ExprMethodCall whose method `len` resolved to no user fn -> the
#     fail-closed report_error -> ir_bodies_failed (no ELF).
#   - string `a + b` fell through to the generic integer IrBinOp(OpAdd) -> the two rodata
#     POINTERS were added as integers into a garbage pointer -> SIGSEGV at use (a SILENT
#     miscompile, not a clean reject).
#
# THE FIX (self-hosted/ir/lower.sio only): the lowerer now tracks string-typed locals/params
# (LowerLocalStack.is_string, seeded from a `: string` annotation, a string-literal/string-
# builtin initializer, or a string param) and:
#   - lowers `s.len()` (string receiver) to a synthesized call to the str_len builtin;
#   - lowers `a + b` (either operand string -> both are, per the checker) to str_concat.
# No new IR opcode, no codegen change: the proven builtins are reused via the ordinary call
# ABI (find_or_add_fn_id + ir_lower_prepend_validated_arg + ir_call, name-keyed in codegen).
#
# Anti-overclaim: every p* asserts FILE-PRESENT + the EXACT RUN-ELF exit code (p08 checks
# concat CONTENT, not just length; p09 is the INTEGER `+` control that must stay 7). r01 is
# a REJECT witness: array `.len()` is a DIFFERENT semantics (element count, unimplemented)
# and MUST still fail closed (no ELF) -- proving the str_len redirect is gated on string-ness
# and never silently runs strlen over an array handle. Prints N/N; nonzero on any miss.
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

# Positive witnesses: must emit an ELF and exit with the expected code.
for src in "$DIR"/p[0-9]*.sio; do
  tot=$((tot+1))
  n=$(basename "$src" .sio)
  exp="${EXP[$n]:-}"
  out="/tmp/strgate_$n.elf"
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

# Reject witnesses: array `.len()` MUST NOT emit an ELF (file absence, not just rc).
for src in "$DIR"/r[0-9]*.sio; do
  tot=$((tot+1))
  n=$(basename "$src" .sio)
  out="/tmp/strgate_$n.elf"
  rm -f "$out"
  "$BIN" --native-v2-compile "$src" -o "$out" >/dev/null 2>&1
  if [ -f "$out" ]; then
    echo "WRONG $n (REJECT expected but ELF emitted)"
  else
    echo "PASS  $n (rejected, no ELF)"; pass=$((pass+1))
  fi
done

echo "---- $pass/$tot ----"
[ "$pass" = "$tot" ] || exit 1
