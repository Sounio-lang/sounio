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
# whose entry did not fit. Naming it proves the sticky overflow state carried the
# name through the reset, not just a bare flag.
grep -Fq "more than ${TUPLE_CAP} functions return a tuple with an \`[f64; N]\` slot, starting at \`t${TUPLE_CAP}\`" "$TUPLE_DIR/over_cap.log" || {
  tail -n 40 "$TUPLE_DIR/over_cap.log" >&2
  fail "${TUPLE_OVER}-tuple-fn capacity diagnostic was missing or changed"
}

echo "[madaros-f64-lowering] PASS: one shared Madaros ELF passed dereference, global f64, direct capacity, imported capacity, imported wide-call, and f64-array tuple table capacity (${TUPLE_CAP} ok, ${TUPLE_OVER} rejected) gates"
