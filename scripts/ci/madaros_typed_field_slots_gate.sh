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
#   4. `l as *mut ItemList` over a Box<T> (private_fn_identity's item walk)
#      kept the handle, so `(*node).head.kind` read the handle number as an
#      address. A raw-pointer cast of a managed object now yields the address
#      of its first field (__ptr_of_ref), like heap_alloc memory.
#      `*p` as a whole struct is that object again (base = p - header), and
#      `let p = f(...)` with f returning `*mut T` types p as T (pfi_child).
#   5. `var cur = (*t).head` with head declared `*mut Chunk` was untyped, so
#      `(*cur).next` used field index 1 and collided with entries[1]
#      (fn_sig_table_add). A raw-pointer field now types the local.
#   6. `(*(*c).tab).entries[i]` (the deref of a raw-pointer field) took the
#      generic deref and read word 0 of the table (checker_struct_visible).
#   7. `*p` / `(*p).method()` over a raw struct with a fixed-array field
#      copies the raw payload into an object, and `*p = v` copies back
#      (the check.sio by-value bridges). A raw cast of a managed value whose
#      struct has a fixed-array field is refused: the raw layout keeps the
#      array inline, the object does not.
#   8. `var paths: [string; N]` carries `string` as its struct type; as a store
#      target it was taken for a string and written one byte at a time.
#   9. `&(*p)` / `&!(*p)` over a raw `*mut S` reach `&S` / `&!S` callees as a
#      reference to the raw payload (a `&!` write lands in it); a raw pointer
#      local is never value-copied as an S (`(*h).s = p` stored 1). `&!(*p)`
#      for an S with a fixed-array field is refused.
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
check deref_array_param_elem_typed DEREF_ARRAY_ELEM_OK
check raw_ptr_of_managed_object RAW_PTR_OF_MANAGED_OK
check raw_ptr_field_chain_typed RAW_PTR_FIELD_CHAIN_OK
check raw_ptr_field_deref_typed RAW_PTR_FIELD_DEREF_OK
check raw_struct_whole_copy RAW_STRUCT_WHOLE_COPY_OK
check string_array_store STRING_ARRAY_STORE_OK
check raw_ref_alias RAW_REF_ALIAS_OK

cat >"$WORK/refuse.sio" <<'SIO'
struct Seg { buf: [i8; 16], len: i64 }

fn seg_ptr(b: Box<Seg>) -> *mut Seg with Mut, Panic, Div, Alloc {
    b as *mut Seg
}

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    let p = seg_ptr(Box::new(Seg { buf: [0; 16], len: 3 }))
    println((*p).len)
    return 0
}
SIO
set +e
( ulimit -s 1572864 2>/dev/null || true; "$MADAROS" build "$WORK/refuse.sio" "$WORK/refuse.elf" ) >"$WORK/refuse.log" 2>&1
refuse_rc=$?
set -e
[[ "$refuse_rc" -ne 0 ]] && grep -q "fixed-array field" "$WORK/refuse.log" \
  || fail "raw cast of a managed struct with a fixed-array field: rc=$refuse_rc, expected a refusal naming the fixed-array field"

cat >"$WORK/refuse_ref.sio" <<'SIO'
struct D { arr: [i64; 4], n: i64 }

fn bump(r: &!D) with Mut, Panic, Div { r.n = r.n + 1 }

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    let q = heap_alloc(256) as *mut D
    bump(&!(*q))
    println((*q).n)
    return 0
}
SIO
set +e
( ulimit -s 1572864 2>/dev/null || true; "$MADAROS" build "$WORK/refuse_ref.sio" "$WORK/refuse_ref.elf" ) >"$WORK/refuse_ref.log" 2>&1
refuse_ref_rc=$?
set -e
[[ "$refuse_ref_rc" -ne 0 ]] && grep -q "fixed-array field" "$WORK/refuse_ref.log" \
  || fail "&!(*p) over a raw struct with a fixed-array field: rc=$refuse_ref_rc, expected a refusal"
echo "[madaros-typed-field-slots] PASS: tuple second struct slot, tuple Box/Option<Box> slot, deref'd Option<Box<T>> and raw-pointer casts read the declared struct's fields"
