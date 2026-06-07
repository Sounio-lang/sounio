#!/usr/bin/env bash
# native_v2_struct_field_float_gate — guards the struct-FIELD float-arithmetic fix.
#
# THE BUG (residual of the both-memory-operand float-arith fix): float arithmetic between
# two struct FIELD reads (`p.x + p.y`) miscompiled to 0. The binop float-detection sees
# only regs flagged float; a field read carried no float-ness because (a) the lowerer
# tracked no local struct types, (b) StructFieldEntry carried no type, and (c) the summary
# path never populated the struct-layout table (register_struct_fields_ref wrote through a
# double-deref-through-Box store that c634b38f no-ops — only the type-free field-index hash
# kept working). So the field operands took the INTEGER path and added raw f64 bit patterns
# (-> garbage -> `as i64` = 0).
#
# THE FIX (ir/ir.sio + ir/lower.sio): StructFieldEntry gains is_float (seeded from the field
# type at registration); the layout table is populated via the PROVEN mut registration
# (binds `let sl = (*lo).struct_layouts` + copy-out); each struct-typed param/let records its
# struct-layout index (annotation or struct-literal initializer); and a field read `p.x`
# resolves base -> layout -> field is_float to tag the IrBinOp onto the SSE path.
#
# Anti-overclaim (C0/R3/R4): every case asserts FILE-PRESENT + the EXACT exit code of the
# RUN ELF. s06/s07 are INTEGER field controls (must stay on the integer path); s07/s08 mix
# int+float fields in one struct so a per-field (not per-struct) float decision is required.
# Prints N/N; nonzero on miss.
#
# KNOWN RESIDUAL (DEFER, scoped out by design — never mis-tagged, just unfixed -> stays 0):
# non-ident field bases — nested `o.inner.x`, `f().x`, `a[i].x` — and reference/box struct
# params (`&P`/`Box<P>`): the field-float resolver only handles an IDENT base bound to a
# known plain-named struct layout (mirrors the array-index precedent).
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
for src in "$DIR"/s[0-9]*.sio; do
  tot=$((tot+1))
  n=$(basename "$src" .sio)
  exp="${EXP[$n]:-}"
  out="/tmp/structfieldfloat_$n.elf"
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
