#!/usr/bin/env bash
# Only functions and globals get IR function slots in the item preseed.
#
# lowerer_preseed_one_item_mut used to be one nested
#   if kind == ItemFn { if !skip { let fd = ..; if fd != 0 {..} if fd == 0 {..} } }
#   else if kind == ItemImpl {..}
# and lean_single compiled it so that a `use` item (and an `algebra` or
# `effect` declaration) reached the `fd == 0` arm without passing either test.
# The lean_single-built gen1 therefore gave each such item a BSS slot, empty-
# named for `use`, while gen2 (built by Madaros) did not. Every later fn id
# shifted by one per item and gen2 != gen3 from the first module on (#2966).
#
# A compiler under test passes when, for each program below, its IR trace
# (SOUNIO_NV2_IR_TRACE) has no function slot named "" and none named after the
# declaration:
#   1. a two-module program (`use mm::c::{c_add}`),
#   2. an `algebra Octonion` declaration,
#   3. an `effect Choice` declaration.
# Each program must also build and print its marker.
#
# Usage: madaros_preseed_item_slots_gate.sh   (MADAROS_BIN = compiler under
#                                              test, default bin/souc)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/souc}"
WORK="$(mktemp -d /tmp/sounio-preseed-slots.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
fail() { echo "[madaros-preseed-item-slots] FAIL: $*" >&2; exit 1; }
ulimit -s unlimited 2>/dev/null || ulimit -s 1048576 2>/dev/null || true

mkdir -p "$WORK/lib/mm"
cat >"$WORK/lib/mm/c.sio" <<'SIO'
pub fn c_add(a: i64, b: i64) -> i64 with Mut, Panic { a + b + 100 }
SIO
cat >"$WORK/use.sio" <<'SIO'
use mm::c::{c_add}
fn main() -> i64 with IO, Mut, Panic, Div {
    print("PRESEED_USE ")
    println(c_add(10, 5))
    0
}
SIO
cat >"$WORK/algebra.sio" <<'SIO'
algebra Octonion over f64 {
    add: commutative, associative
    mul: alternative, non_commutative
    reassociate: fano_selective
}

fn main() -> i32 with IO {
    println("PRESEED_ALGEBRA")
    0
}
SIO
cat >"$WORK/effect.sio" <<'SIO'
effect Choice {
    fn pick() -> bool
}

fn main() -> i32 with IO {
    println("PRESEED_EFFECT")
    0
}
SIO

# check <name> <stdlib dir> <decl name or -> <expected stdout>
check() {
  local name="$1" lib="$2" decl="$3" want="$4"
  local trace="$WORK/$name.trace" elf="$WORK/$name.elf"
  SOUNIO_STDLIB_PATH="$lib" SOUNIO_NV2_IR_TRACE=1 "$MADAROS" build "$WORK/$name.sio" -o "$elf" >"$trace" 2>&1 \
    || fail "$name: build failed: $(tail -3 "$trace")"
  # The absence checks below are vacuous on an empty trace: require the trace
  # to list the program's slots (main plus the runtime helpers).
  local nfn; nfn="$(grep -c '^NV2_IR function fn=' "$trace" || true)"
  [[ -n "$nfn" && "$nfn" -ge 2 ]] || fail "$name: SOUNIO_NV2_IR_TRACE listed $nfn function slots (expected >= 2); cannot check slot names"
  grep -q '^NV2_IR function fn=[0-9]* name=main ' "$trace" || fail "$name: the IR trace has no slot for main"
  if grep -q '^NV2_IR function fn=[0-9]* name= ' "$trace"; then
    fail "$name: an IR function slot has an empty name: $(grep -m1 '^NV2_IR function fn=[0-9]* name= ' "$trace")"
  fi
  if [ "$decl" != "-" ] && grep -q "^NV2_IR function fn=[0-9]* name=$decl " "$trace"; then
    fail "$name: the declaration '$decl' was given an IR function slot"
  fi
  chmod +x "$elf"
  local out; out="$("$elf" 2>&1)" || fail "$name: binary exited non-zero: $out"
  [ "$out" = "$want" ] || fail "$name: printed '$out', expected '$want'"
}

check use "$WORK/lib" - "PRESEED_USE 115"
check algebra "$ROOT_DIR/stdlib" Octonion "PRESEED_ALGEBRA"
check effect "$ROOT_DIR/stdlib" Choice "PRESEED_EFFECT"
echo "[madaros-preseed-item-slots] PASS: use / algebra / effect items get no IR function slot"
