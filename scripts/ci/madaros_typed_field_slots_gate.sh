#!/usr/bin/env bash
# Field reads through a tuple's second struct slot and through a deref'd
# Option<Box<T>> resolve against the declared struct, not the untyped vote.
#
# When Madaros cannot name a receiver's struct it picks a field index by a
# majority vote over every struct with a field of that name. In the gen2
# whole-program compile (#2966) the vote chose the wrong struct in
# Parser::parse_let_stmt (`pat.name` read Pattern.kind) and in
# pure_fn_reg_record (`(*node).head.name` read word 6 of a Param). Two shapes
# lost the type:
#   1. `let pat = pair.1` where `pair = p.parse_pattern()` returns
#      (Parser, Pattern): only the FIRST struct slot of a tuple return was typed.
#   2. `var cur = *params` with `params: &Option<Box<ParamList>>`: the deref
#      copy did not inherit the Option<Box<T>> payload layout.
#   3. `let next = pair2.1` where `p.parse_call_arg_expr()` returns
#      (Parser, Box<ExprList>): a Box<T> or Option<Box<T>> tuple slot was not
#      typed at all, so Parser::parse_call_expr_box stored every later call
#      argument's head as its label.
# Each witness surrounds the right struct with decoys that win the vote.
# Madaros from main prints 3, 1055 and 3 (or 40300003); lean_single passes all.
#
# Usage: madaros_typed_field_slots_gate.sh   (MADAROS_BIN = compiler under test,
#                                             default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-typed-slots.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-typed-field-slots] FAIL: $*" >&2; exit 1; }

check() {
  local name="$1" want="$2"
  local src="$ROOT_DIR/tests/run-pass/$name.sio"
  ( ulimit -s 1572864 2>/dev/null || true; "$MADAROS" build "$src" "$WORK/$name.elf" ) >"$WORK/$name.log" 2>&1 \
    || { tail -n 20 "$WORK/$name.log" >&2; fail "$name: build failed"; }
  chmod +x "$WORK/$name.elf"
  set +e
  local out
  out="$("$WORK/$name.elf" 2>&1)"
  local rc=$?
  set -e
  [[ "$rc" -eq 0 && "$out" == "$want" ]] || fail "$name: rc=$rc output=$(tr '\n' ' ' <<<"$out"), expected $want"
}

check tuple_second_struct_slot_typed TUPLE_SLOT_TYPED_OK
check deref_option_box_payload_typed DEREF_OPTION_PAYLOAD_OK
check tuple_box_slot_typed TUPLE_BOX_SLOT_OK
check tuple_slot_forward_method_typed TUPLE_SLOT_FORWARD_OK
echo "[madaros-typed-field-slots] PASS: tuple second struct slot, tuple Box/Option<Box> slot and deref'd Option<Box<T>> read the declared struct's fields"
