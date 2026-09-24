#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP:-0}"

fail() {
  echo "[madaros-f64-lowering] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_F64_LOWERING_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_F64_LOWERING_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-f64-lowering.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_F64_LOWERING_GATE_BIN:-$WORK/madaros}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_F64_LOWERING_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

SOUNIO_MADAROS_DEREF_F64_GATE_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_DEREF_F64_GATE_DIR="$WORK/deref" \
SOUNIO_MADAROS_DEREF_F64_GATE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_imported_deref_f64_array_gate.sh"

SOUNIO_MADAROS_GLOBAL_F64_GATE_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_GLOBAL_F64_GATE_DIR="$WORK/global" \
SOUNIO_MADAROS_GLOBAL_F64_GATE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_global_f64_scratch_gate.sh"

SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_DIR="$WORK/global-capacity" \
SOUNIO_MADAROS_GLOBAL_CAPACITY_GATE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_global_capacity_gate.sh"

SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_DIR="$WORK/imported-capacity" \
SOUNIO_MADAROS_IMPORTED_CAPACITY_GATE_KEEP=1 \
  bash "$ROOT_DIR/scripts/ci/madaros_imported_capacity_gate.sh"

MADAROS_RAW_BIN="$MADAROS_ELF" \
SOUNIO_MADAROS_CALL_ARITY_13_DIR="$WORK/imported-call-arity-13" \
  bash "$ROOT_DIR/scripts/ci/madaros_imported_call_arity_13_gate.sh"

# f64-array tuple side table capacity (lower.sio LOWER_FN_TUPLE_ARR_CAP).
#
# Past the cap the collector used to stop recording and say nothing, so later
# tuple-returning fns reverted to the integer element path in a program that
# still compiled clean. It is now a hard error, and that error is raised in a
# pre-pass that runs ONE LINE before a lowerer_new() whose lower_hard_error_reset
# would wipe it -- so the only thing that proves it survives is a real overflow
# through the real driver. Two arms, because a rejection alone cannot tell a
# boundary from a blanket failure:
#   CAP     tuple-returning fns   must compile and run, INCLUDING the last slot
#                                 (its name sits at the far end of the name buffer)
#   CAP+1   tuple-returning fns   must be rejected, naming the cap and the fn
# READ THE CAP, DO NOT PIN IT -- same reason madaros_global_capacity_gate.sh
# derives BSS_MAX_GLOBALS: a boundary gate whose boundary is a literal tests the
# literal, not the compiler.
TUPLE_CAP="$(grep -E '^let LOWER_FN_TUPLE_ARR_CAP: i64 = [0-9]+' \
    "$ROOT_DIR/self-hosted/ir/lower.sio" | grep -oE '[0-9]+$' | head -1)"
[[ -n "$TUPLE_CAP" ]] || fail "LOWER_FN_TUPLE_ARR_CAP is no longer declared where this gate looks"
TUPLE_OVER="$((TUPLE_CAP + 1))"

# Compile-boundary ordering. This is a STRUCTURAL check, and deliberately says so:
# the property it protects -- an over-cap source must not poison the NEXT source
# compiled in the same process -- cannot be exercised here, because every
# madaros CLI invocation compiles exactly once. What can be pinned is the one
# ordering that would silently break it. lower_hard_error_reset() re-arms reasons 3/4
# from the table's sticky fault, so the table must be reset FIRST at the
# compile barrier; the other way round, a stale fault re-arms straight into the
# next compile and rejects a perfectly good source.
BARRIER_BODY="$(awk '/^fn module_frontend_global_init_compile_begin\(/{on=1} on{print} on&&/^}/{exit}' \
    "$ROOT_DIR/self-hosted/compiler/module_frontend.sio" | sed 's|//.*$||')"
[[ -n "$BARRIER_BODY" ]] || fail "module_frontend_global_init_compile_begin is no longer where this gate looks"
reset_line="$(grep -n 'lower_fn_tuple_arr_reset()' <<<"$BARRIER_BODY" | head -1 | cut -d: -f1)"
herr_line="$(grep -n 'lower_hard_error_reset()' <<<"$BARRIER_BODY" | head -1 | cut -d: -f1)"
[[ -n "$reset_line" ]] || fail "the compile barrier no longer resets the f64-array tuple table: an over-cap source would poison every later compile in the process"
[[ -n "$herr_line" ]] || fail "the compile barrier no longer calls lower_hard_error_reset(); this ordering check has nothing to order against"
[[ "$reset_line" -lt "$herr_line" ]] || fail "the compile barrier resets the tuple table AFTER lower_hard_error_reset(): the hard-error reset re-arms from the previous compile's stale fault"

TUPLE_DIR="$WORK/tuple-capacity"
mkdir -p "$TUPLE_DIR"

# $1 = number of tuple-returning fns, $2 = output source. Every fn returns an
# f64-array slot, so every one of them needs a side-table entry.
gen_tuple_fns() {
  local n="$1" out="$2" i
  : >"$out"
  for i in $(seq 0 "$((n - 1))"); do
    printf 'fn t%s() -> ([f64; 2], [f64; 2]) with Mut, Panic {\n  var a: [f64; 2] = [0.0; 2]\n  var b: [f64; 2] = [0.0; 2]\n  a[0] = 1.5\n  (a, b)\n}\n' "$i" >>"$out"
  done
  printf 'fn main() -> i32 with IO, Mut, Panic {\n  let (x, y) = t%s()\n  let d: f64 = x[0] * 2.0\n  if d == 3.0 {\n    return 0\n  }\n  1\n}\n' "$((n - 1))" >>"$out"
}

AT_SRC="$TUPLE_DIR/at_cap.sio"
AT_OUT="$TUPLE_DIR/at_cap.elf"
gen_tuple_fns "$TUPLE_CAP" "$AT_SRC"
set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$AT_SRC" -o "$AT_OUT" >"$TUPLE_DIR/at_cap.log" 2>&1
at_rc=$?
set -e
if [[ "$at_rc" -ne 0 ]]; then
  tail -n 40 "$TUPLE_DIR/at_cap.log" >&2
  fail "${TUPLE_CAP}-tuple-fn boundary witness did not compile rc=$at_rc"
fi
[[ -e "$AT_OUT" ]] || fail "${TUPLE_CAP}-tuple-fn boundary witness produced no output artifact"
chmod +x "$AT_OUT"
set +e
"$AT_OUT" >"$TUPLE_DIR/at_cap.run.log" 2>&1
at_run_rc=$?
set -e
# Exit 1 here is the corruption itself: t<CAP-1> is the LAST recorded entry, and
# if its f64 slot were not marked, x[0] * 2.0 would convert the f64 bits as an
# integer and miss 3.0.
if [[ "$at_run_rc" -ne 0 ]]; then
  cat "$TUPLE_DIR/at_cap.run.log" >&2
  fail "${TUPLE_CAP}-tuple-fn witness ran rc=$at_run_rc: the LAST table entry's f64 slot was not marked"
fi

OVER_SRC="$TUPLE_DIR/over_cap.sio"
OVER_OUT="$TUPLE_DIR/over_cap.elf"
gen_tuple_fns "$TUPLE_OVER" "$OVER_SRC"
set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$OVER_SRC" -o "$OVER_OUT" >"$TUPLE_DIR/over_cap.log" 2>&1
over_rc=$?
set -e
if [[ "$over_rc" -eq 0 ]]; then
  tail -n 40 "$TUPLE_DIR/over_cap.log" >&2
  fail "${TUPLE_OVER}-tuple-fn witness compiled clean: the table dropped its metadata silently again"
fi
if [[ "$over_rc" -ge 128 ]]; then
  tail -n 40 "$TUPLE_DIR/over_cap.log" >&2
  fail "${TUPLE_OVER}-tuple-fn witness terminated by signal rc=$over_rc"
fi
if [[ -e "$OVER_OUT" ]]; then
  fail "${TUPLE_OVER}-tuple-fn capacity rejection left an output artifact: $OVER_OUT"
fi
# The cap AND the first fn past it: t<CAP> is the (CAP+1)th, so it is the one
# whose entry did not fit. Naming it proves the sticky fault state carried the
# name through the reset, not just a bare flag.
#
# Copilot follow-up (#2570): lower.sio's reason-3 message was reworded from
# "f64-array tuple-slot metadata" to "scalar/f64-array tuple-slot metadata"
# once LOWER_FN_TUPLE_ARR_SCALAR started sharing this same cap (a scalar-only
# f64 tuple slot can now trigger this fault too, not just an array one) --
# this gate's expected substring was not updated to match, so a real,
# correctly-worded cap rejection failed this gate as if the diagnostic had
# silently changed. Matches the wording actually printed by
# lower_fn_tuple_arr_fault reason 3 today.
grep -Fq "more than ${TUPLE_CAP} functions need scalar/f64-array tuple-slot metadata (their own tuple return, or a returned function's), starting at \`t${TUPLE_CAP}\`" "$TUPLE_DIR/over_cap.log" || {
  tail -n 40 "$TUPLE_DIR/over_cap.log" >&2
  fail "${TUPLE_OVER}-tuple-fn capacity diagnostic was missing or changed"
}

# f64-array tuple SLOT limit (lower.sio LOWER_FN_TUPLE_ARR_MAX_SLOTS). The array mask
# rides in bits 32+ of an i64, so slot 31 would land on the sign bit; slots
# 0..MAX-1 are representable and a `[f64; N]` at MAX or later must be REFUSED, not
# skipped -- it used to be skipped by a bare `bit < 31`, leaving the array to be
# read as integers in a program that compiled clean.
#
# The frontend's tuple LITERAL cap (E008 past 16 elements) does not make this
# unreachable: a wide tuple TYPE compiles clean when no literal produces it. So the
# witness declares the type and gives it a body that needs no literal (a self call)
# behind an `if false`, which is all the collector needs to see. Read the limit,
# do not pin it.
SLOT_MAX="$(grep -E '^let LOWER_FN_TUPLE_ARR_MAX_SLOTS: i64 = [0-9]+' \
    "$ROOT_DIR/self-hosted/ir/lower.sio" | grep -oE '[0-9]+$' | head -1)"
[[ -n "$SLOT_MAX" ]] || fail "LOWER_FN_TUPLE_ARR_MAX_SLOTS is no longer declared where this gate looks"
SLOT_DIR="$WORK/tuple-slot"
mkdir -p "$SLOT_DIR"

# $1 = slot index of the f64 array (the tuple has $1+1 elements), $2 = output source.
gen_wide_tuple() {
  local k="$1" out="$2" ty="" i
  for i in $(seq 1 "$k"); do ty="${ty}i64, "; done
  printf 'fn wide() -> (%s[f64; 2]) {\n  wide()\n}\n\nfn main() -> i32 with IO, Mut, Panic {\n  if false {\n    let t = wide()\n    let x = t.%s\n    let d: f64 = x[0] * 2.0\n  }\n  0\n}\n' "$ty" "$k" >"$out"
}

LAST_OK_SRC="$SLOT_DIR/last_ok.sio"
LAST_OK_OUT="$SLOT_DIR/last_ok.elf"
gen_wide_tuple "$((SLOT_MAX - 1))" "$LAST_OK_SRC"
set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$LAST_OK_SRC" -o "$LAST_OK_OUT" >"$SLOT_DIR/last_ok.log" 2>&1
last_ok_rc=$?
set -e
if [[ "$last_ok_rc" -ne 0 ]]; then
  tail -n 40 "$SLOT_DIR/last_ok.log" >&2
  fail "tuple with its f64 array at slot $((SLOT_MAX - 1)), the LAST representable one, was rejected rc=$last_ok_rc"
fi

TOO_FAR_SRC="$SLOT_DIR/too_far.sio"
TOO_FAR_OUT="$SLOT_DIR/too_far.elf"
gen_wide_tuple "$SLOT_MAX" "$TOO_FAR_SRC"
set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$TOO_FAR_SRC" -o "$TOO_FAR_OUT" >"$SLOT_DIR/too_far.log" 2>&1
too_far_rc=$?
set -e
if [[ "$too_far_rc" -eq 0 ]]; then
  tail -n 40 "$SLOT_DIR/too_far.log" >&2
  fail "tuple with an f64 array at slot ${SLOT_MAX} compiled clean: the slot is being skipped silently again"
fi
if [[ "$too_far_rc" -ge 128 ]]; then
  tail -n 40 "$SLOT_DIR/too_far.log" >&2
  fail "slot-${SLOT_MAX} witness terminated by signal rc=$too_far_rc"
fi
if [[ -e "$TOO_FAR_OUT" ]]; then
  fail "slot-${SLOT_MAX} rejection left an output artifact: $TOO_FAR_OUT"
fi
grep -Fq "function \`wide\`'s f64-array tuple metadata (its own return, or a returned function's) needs slot ${SLOT_MAX} or later" "$SLOT_DIR/too_far.log" || {
  tail -n 40 "$SLOT_DIR/too_far.log" >&2
  fail "slot-${SLOT_MAX} diagnostic was missing or changed"
}


# f64-array tuple table coverage for IMPL METHODS (#2570 review follow-up).
# lower_fn_tuple_f64_arrays_collect only ever walked top-level ItemFn entries;
# an impl method returning a tuple with an [f64; N] slot is lowered as a
# mangled `Type_method` free function (lower_impl_methods_ref) but the
# prepass never reached it, so the table had no entry under that mangled
# name -- every method-result read of the array slot took the integer path.
# expr_is_tuple_slot_f64_array_ref and expr_result_tuple_float_mask_ref (the
# let-binding side) also only recognized an ExprCall base, not
# ExprMethodCall. Two shapes, matching the direct-projection witness above:
#   A  `recv.pair().0[0]`            direct projection off the method call
#   B  `let t = recv.pair(); t.0[0]` in-place index through a local bound
#                                     from the method call's result
IMPLM_DIR="$WORK/tuple-impl-method"
mkdir -p "$IMPLM_DIR"
cat > "$IMPLM_DIR/main.sio" <<'SOUNIO'
struct Pairer {
    scale: f64,
}

impl Pairer {
    fn pair(self) -> ([f64; 2], i64) with Mut, Panic {
        var a: [f64; 2] = [0.0; 2]
        a[0] = self.scale
        (a, 9)
    }
}

fn main() -> i32 with IO, Mut, Panic {
    var bad: i32 = 0
    let recv = Pairer { scale: 1.5 }
    let da: f64 = recv.pair().0[0] * 2.0
    if da != 3.0 { bad = bad + 1 }
    let t = recv.pair()
    let db: f64 = t.0[0] * 2.0
    if db != 3.0 { bad = bad + 2 }
    if bad == 0 {
        println("TUPLE_F64_ARRAY_IMPL_METHOD_OK")
        return 0
    }
    bad
}
SOUNIO
IMPLM_OUT="$IMPLM_DIR/main.elf"
set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$IMPLM_DIR/main.sio" -o "$IMPLM_OUT" >"$IMPLM_DIR/compile.log" 2>&1
implm_compile_rc=$?
set -e
if [[ "$implm_compile_rc" -ne 0 ]]; then
  tail -n 40 "$IMPLM_DIR/compile.log" >&2
  fail "impl-method f64-array-tuple witness did not compile rc=$implm_compile_rc"
fi
[[ -e "$IMPLM_OUT" ]] || fail "impl-method f64-array-tuple witness produced no output artifact"
chmod +x "$IMPLM_OUT"
set +e
IMPLM_RUN_OUT="$("$IMPLM_OUT" 2>&1)"
implm_run_rc=$?
set -e
if [[ "$implm_run_rc" -ne 0 ]] || [[ "$IMPLM_RUN_OUT" != "TUPLE_F64_ARRAY_IMPL_METHOD_OK" ]]; then
  fail "impl-method f64-array-tuple witness ran rc=$implm_run_rc out=[$IMPLM_RUN_OUT]: a method returning an f64-array tuple was read as integer bits"
fi

# LOWER_FN_ARR_CHAIN table capacity (lower_fn_tuple_arr_fault reason 5).
#
# Review follow-up (#2570): past LOWER_FN_ARR_CHAIN_COUNT's cap (shared with
# LOWER_FN_TUPLE_ARR_CAP), lower_fn_arr_chain_insert fails closed via reason 5
# -- the array-chain sibling of the tuple table's reason-3 cap fault above,
# added by this PR but never exercised by any fixture: a regression that
# dropped LOWER_FN_ARR_CHAIN entries, or lost its sticky hard error, would
# stay green. Same two-arm shape as the tuple-table capacity witness above.
#   CAP     one-layer array-chain fns   must compile and run, INCLUDING the
#                                       last slot (the boundary entry)
#   CAP+1   one-layer array-chain fns   must be rejected, naming the cap and
#                                       the fn
CHAIN_DIR="$WORK/arr-chain-capacity"
mkdir -p "$CHAIN_DIR"

# $1 = number of one-layer array-chain fns (`cK() -> fn() -> [f64; 2]`, each
# returning `innerK`), $2 = output source.
gen_chain_fns() {
  local n="$1" out="$2" i
  : >"$out"
  for i in $(seq 0 "$((n - 1))"); do
    printf 'fn inner%s() -> [f64; 2] {\n  var a: [f64; 2] = [0.0; 2]\n  a[0] = 1.5\n  a\n}\nfn c%s() -> fn() -> [f64; 2] {\n  inner%s\n}\n' "$i" "$i" "$i" >>"$out"
  done
  printf 'fn main() -> i32 with IO, Mut, Panic {\n  let f = c%s()\n  let a = f()\n  let d: f64 = a[0] * 2.0\n  if d == 3.0 {\n    return 0\n  }\n  1\n}\n' "$((n - 1))" >>"$out"
}

CHAIN_AT_SRC="$CHAIN_DIR/at_cap.sio"
CHAIN_AT_OUT="$CHAIN_DIR/at_cap.elf"
gen_chain_fns "$TUPLE_CAP" "$CHAIN_AT_SRC"
set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$CHAIN_AT_SRC" -o "$CHAIN_AT_OUT" >"$CHAIN_DIR/at_cap.log" 2>&1
chain_at_rc=$?
set -e
if [[ "$chain_at_rc" -ne 0 ]]; then
  tail -n 40 "$CHAIN_DIR/at_cap.log" >&2
  fail "${TUPLE_CAP}-array-chain-fn boundary witness did not compile rc=$chain_at_rc"
fi
[[ -e "$CHAIN_AT_OUT" ]] || fail "${TUPLE_CAP}-array-chain-fn boundary witness produced no output artifact"
chmod +x "$CHAIN_AT_OUT"
set +e
"$CHAIN_AT_OUT" >"$CHAIN_DIR/at_cap.run.log" 2>&1
chain_at_run_rc=$?
set -e
# Exit 1 here means c<CAP-1>, the LAST recorded array-chain entry, was not
# classified: `let f = c()` never got fn_ptr_ret_array set, so `f()`'s
# result read its f64 bits as an integer.
if [[ "$chain_at_run_rc" -ne 0 ]]; then
  cat "$CHAIN_DIR/at_cap.run.log" >&2
  fail "${TUPLE_CAP}-array-chain-fn witness ran rc=$chain_at_run_rc: the LAST array-chain entry was not classified"
fi

CHAIN_OVER_SRC="$CHAIN_DIR/over_cap.sio"
CHAIN_OVER_OUT="$CHAIN_DIR/over_cap.elf"
gen_chain_fns "$TUPLE_OVER" "$CHAIN_OVER_SRC"
set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$CHAIN_OVER_SRC" -o "$CHAIN_OVER_OUT" >"$CHAIN_DIR/over_cap.log" 2>&1
chain_over_rc=$?
set -e
if [[ "$chain_over_rc" -eq 0 ]]; then
  tail -n 40 "$CHAIN_DIR/over_cap.log" >&2
  fail "${TUPLE_OVER}-array-chain-fn witness compiled clean: the array-chain table dropped its metadata silently again"
fi
if [[ "$chain_over_rc" -ge 128 ]]; then
  tail -n 40 "$CHAIN_DIR/over_cap.log" >&2
  fail "${TUPLE_OVER}-array-chain-fn witness terminated by signal rc=$chain_over_rc"
fi
if [[ -e "$CHAIN_OVER_OUT" ]]; then
  fail "${TUPLE_OVER}-array-chain-fn capacity rejection left an output artifact: $CHAIN_OVER_OUT"
fi
# c<CAP> is the (CAP+1)th array-chain fn, so it is the one whose entry did
# not fit. Naming it proves the sticky fault state carried the name through
# the reset, not just a bare flag -- same rationale as reason 3's check above.
grep -Fq "more than ${TUPLE_CAP} functions return an array-returning function (\`fn() -> [f64; N]\`), starting at \`c${TUPLE_CAP}\`" "$CHAIN_DIR/over_cap.log" || {
  tail -n 40 "$CHAIN_DIR/over_cap.log" >&2
  fail "${TUPLE_OVER}-array-chain-fn capacity diagnostic was missing or changed"
}

# Array-chain depth refusal (lower_fn_tuple_arr_fault reason 6): a function
# whose return type is a TypeFn chain MORE than one layer deep before
# reaching `[f64; N]` cannot be represented by LOWER_FN_ARR_CHAIN's
# exactly-one-layer shape and must be refused outright, independent of the
# cap above. Also added by this PR with no fixture. The prepass that raises
# this walks every top-level fn's declared return type unconditionally (the
# same one gen_chain_fns above exercises), so merely DECLARING the two-layer
# function is enough to trip it -- main's body does not need to call it.
CHAIN_DEPTH_DIR="$WORK/arr-chain-depth"
mkdir -p "$CHAIN_DEPTH_DIR"
cat > "$CHAIN_DEPTH_DIR/main.sio" <<'SOUNIO'
fn deep_inner() -> [f64; 2] {
    var a: [f64; 2] = [0.0; 2]
    a[0] = 1.5
    a
}

fn deep_mid() -> fn() -> [f64; 2] {
    deep_inner
}

fn deep_outer() -> fn() -> fn() -> [f64; 2] {
    deep_mid
}

fn main() -> i32 with IO, Mut, Panic {
    0
}
SOUNIO
CHAIN_DEPTH_OUT="$CHAIN_DEPTH_DIR/main.elf"
set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" compile "$CHAIN_DEPTH_DIR/main.sio" -o "$CHAIN_DEPTH_OUT" >"$CHAIN_DEPTH_DIR/compile.log" 2>&1
chain_depth_rc=$?
set -e
if [[ "$chain_depth_rc" -eq 0 ]]; then
  tail -n 40 "$CHAIN_DEPTH_DIR/compile.log" >&2
  fail "two-layer array-chain witness compiled clean: a chain deeper than one call is being silently misclassified again"
fi
if [[ "$chain_depth_rc" -ge 128 ]]; then
  tail -n 40 "$CHAIN_DEPTH_DIR/compile.log" >&2
  fail "two-layer array-chain witness terminated by signal rc=$chain_depth_rc"
fi
if [[ -e "$CHAIN_DEPTH_OUT" ]]; then
  fail "two-layer array-chain rejection left an output artifact: $CHAIN_DEPTH_OUT"
fi
grep -Fq "function \`deep_outer\` returns a function chain more than one call deep before" "$CHAIN_DEPTH_DIR/compile.log" || {
  tail -n 40 "$CHAIN_DEPTH_DIR/compile.log" >&2
  fail "two-layer array-chain diagnostic was missing or changed"
}

echo "[madaros-f64-lowering] PASS: one shared Madaros ELF passed dereference, global f64, direct capacity, imported capacity, imported wide-call, f64-array tuple table capacity (${TUPLE_CAP} ok, ${TUPLE_OVER} rejected), slot limit (slot $((SLOT_MAX - 1)) ok, slot ${SLOT_MAX} rejected), impl-method tuple coverage, array-chain table capacity (${TUPLE_CAP} ok, ${TUPLE_OVER} rejected), and array-chain depth refusal"
