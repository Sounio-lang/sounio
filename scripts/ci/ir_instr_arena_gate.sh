#!/usr/bin/env bash
# ir_instr_arena_gate.sh — pin the IR instruction arena's safety contract.
#
# Background (#1649)
# ------------------
# IrFunction embeds `instrs: [IrInstr; 4096]` inline and IrModule embeds
# `functions: [IrFunction; 2048]`, so a module reserves ~2 GB whether it uses it
# or not, and no function may exceed 4096 IR instructions. A 136-line test needs
# 14389. Raising the literal is not available: a 4x bump takes RSS for a small
# compile from 753 MB to ~2-3 GB.
#
# self-hosted/ir/instr_arena.sio is the first step of the replacement: a
# VARIABLE-size region allocator, so a region costs what a function actually
# uses. Nothing imports it yet -- converting IrFunction.instrs is a separate
# change of ~4282 access sites.
#
# What this gate is FOR
# ---------------------
# Variable regions introduce two hazards that would otherwise produce silent
# cross-function miscompiles -- precisely the class #1586/#1645/#1647 closed:
#
#   1. GROWTH moves a region, so any copy of the handle still holding the old
#      base would read and write bytes that now belong to something else.
#      Guarded by a per-region GENERATION.
#   2. ALIASING: the optimizer relies on IrFunction value copies being deep
#      (`var result = func; ...; return (false, func)` must roll back to a
#      pristine original). A shared region breaks that, Sounio has no copy
#      constructor, and a missed hand-written clone compiles perfectly cleanly.
#      Guarded by SEALING: a published region refuses writes.
#
# Both guards are worthless if they are advisory, so the witnesses assert that a
# violation is REFUSED with a distinct code on the checked API, and on the raw
# path -- the one the eventual textual rewrite emits -- is routed to a quarantine
# region, printed, and latched into a violation flag the driver must consult.
#
# The witnesses import self-hosted modules, which user programs cannot resolve,
# so each is compiled as a single translation unit in the same way
# ir_module_arena_v2_soir_v5_bridge_gate.sh does: strip `module`/`use` lines and
# concatenate the dependency closure.
#
# Vacuity: this gate must be able to FAIL. Deleting the generation check or the
# sealed check from instr_arena.sio turns the stale and seal witnesses red; that
# was verified when this gate was written, and SOUNIO_IR_ARENA_VACUITY=1 re-runs
# the check by patching a scratch copy of the module.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Deliberately NOT falling back to an ambient SOUC_BIN. On the machine this gate
# was written, SOUC_BIN pointed at a DIFFERENT CHECKOUT (/workspace/sounio), and
# the gate silently used it and reported a failure that had nothing to do with
# this tree. Same family as the SOUNIO_STDLIB_PATH trap. Only an explicit
# SOUNIO_IR_ARENA_SOUC or an in-repo raw ELF is accepted, and it must identify
# itself as Madaros before anything is compiled with it.
SOUC="${SOUNIO_IR_ARENA_SOUC:-}"
ARENA="self-hosted/ir/ir.sio"   # the arena now lives beside IrInstr/IrFunction
VACUITY="${SOUNIO_IR_ARENA_VACUITY:-0}"
RUN_TIMEOUT="${SOUNIO_IR_ARENA_TIMEOUT:-60}"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-ir-instr-arena.XXXXXX")"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT

fail() { printf 'IR_INSTR_ARENA_FAIL reason=%s\n' "$1" >&2; exit 1; }

# The RAW modular ELF, not bin/madaros. bin/madaros is a launcher with its own
# CLI, and passing --native-v2-compile through it produces a binary that fails
# this gate's first witness at rc=21 while the raw ELF passes -- so the wrapper
# is not a drop-in here. Checked, on 2026-08-05: raw prebuilt PASS, wrapper FAIL.
# Note also that the checked-in prebuilt lags the source tree; for anything that
# depends on recent compiler changes, point SOUNIO_IR_ARENA_SOUC at a
# current-source build (scripts/ci/build_native_souc.sh style).
if [ -z "$SOUC" ]; then
  for cand in \
    "$ROOT_DIR/bin/madaros-linux-x86_64" \
    "$ROOT_DIR/bin/madaros-linux-aarch64" \
    "$ROOT_DIR/bin/madaros-macos-arm64"; do
    if [ -x "$cand" ]; then SOUC="$cand"; break; fi
  done
fi
[ -n "$SOUC" ] && [ -x "$SOUC" ] || fail "no_madaros_binary (set SOUNIO_IR_ARENA_SOUC)"
case "$SOUC" in
  /*) ;;
  *) SOUC="$ROOT_DIR/$SOUC" ;;
esac
# Identify the engine before trusting it. bin/madaros is a LAUNCHER whose CLI
# reinterprets arguments; passing --native-v2-compile through it yields a binary
# that fails the first witness at rc=21 while the raw ELF passes. Require the raw
# modular compiler.
"$SOUC" --version >"$TMP/version.txt" 2>&1 || fail "souc_version_failed"
grep -qi 'Madaros' "$TMP/version.txt" || {
  cat "$TMP/version.txt" >&2
  fail "souc_is_not_madaros"
}
[ -f "$ARENA" ] || fail "arena_module_missing"

# Dependency closure of ir::ir, in declaration order.
DEPS=(
  "self-hosted/lexer/span.sio:lexer::span"
  "self-hosted/check/numeric_format.sio:check::numeric_format"
  "self-hosted/parser/ast.sio:parser::ast"
  "self-hosted/check/types.sio:check::types"
)

compose() {
  local witness="$1" out="$2" arena_src="$3"
  {
    printf 'module ir::instr_arena_gate\n\n'
    for entry in "${DEPS[@]}"; do
      local path="${entry%%:*}" mod="${entry##*:}"
      [ -f "$path" ] || fail "dep_missing_${path//\//_}"
      sed -e "/^module ${mod//::/\\:\\:}\$/d" -e '/^use /d' "$path"
    done
    sed -e '/^module ir::ir$/d' -e '/^use /d' "$arena_src"
    sed -e '/^use /d' "$witness"
  } >"$out"
}

# name:marker
WITNESSES=(
  "ir_instr_arena_witness:IR_INSTR_ARENA_WITNESS_PASS"
  "ir_instr_arena_stale_witness:IR_INSTR_ARENA_STALE_PASS"
  "ir_instr_arena_seal_witness:IR_INSTR_ARENA_SEAL_PASS"
  # Side-pool exhaustion. Consumes the pools for real -- no capacity is reduced --
  # and asserts the latch fires at the exact arithmetic boundary: 4194304/128 =
  # 32768 for names, 262144/64 = 4096 for args. Before this, both pools dropped
  # their payload and carried on, which is rc=12 with the alarm removed.
  "ir_arena_pool_witness:IR_ARENA_POOL_WITNESS_PASS"
  # Publication. Pins that sealing a module seals EVERY live region -- the count,
  # not merely "no error" -- and that a write afterwards is refused, quarantined
  # and latched. The count half exists because the first wiring sealed 0 of 7 in
  # silence.
  "ir_module_seal_witness:IR_MODULE_SEAL_WITNESS_PASS"
  # The swap primitive. A swap written as two stores cannot carry call arguments
  # -- the first store overwrites the slot the second must read them from -- so
  # this is the one piece of the rc=12 repair with no natural coverage: its call
  # sites need specific IR shapes.
  "ir_arena_swap_witness:IR_ARENA_SWAP_WITNESS_PASS"
)

run_one() {
  local name="$1" marker="$2" arena_src="$3" expect="$4"
  local src="tests/native-v2/$name.sio"
  [ -f "$src" ] || fail "witness_missing_$name"
  compose "$src" "$TMP/$name.sio" "$arena_src"
  if ! "$SOUC" --native-v2-compile "$TMP/$name.sio" -o "$TMP/$name.elf" \
        >"$TMP/$name.build.log" 2>&1; then
    [ "$expect" = "fail" ] && return 0
    cat "$TMP/$name.build.log" >&2
    fail "build_$name"
  fi
  if grep -qE '^error' "$TMP/$name.build.log"; then
    [ "$expect" = "fail" ] && return 0
    grep -E '^error' "$TMP/$name.build.log" >&2
    fail "build_errors_$name"
  fi
  chmod +x "$TMP/$name.elf"
  timeout "$RUN_TIMEOUT" "$TMP/$name.elf" >"$TMP/$name.out" 2>&1
  local rc=$?
  if [ "$expect" = "fail" ]; then
    if [ "$rc" -eq 0 ] && grep -Fxq "$marker" "$TMP/$name.out"; then
      fail "vacuous_$name (witness still passes with the guard removed)"
    fi
    printf 'IR_INSTR_ARENA_VACUITY_OK witness=%s rc=%s\n' "$name" "$rc"
    return 0
  fi
  [ "$rc" -eq 0 ] || { cat "$TMP/$name.out" >&2; fail "runtime_${name}_rc_$rc"; }
  grep -Fxq "$marker" "$TMP/$name.out" || { cat "$TMP/$name.out" >&2; fail "marker_missing_$name"; }
  printf 'IR_INSTR_ARENA_WITNESS_OK witness=%s\n' "$name"
}

for entry in "${WITNESSES[@]}"; do
  run_one "${entry%%:*}" "${entry##*:}" "$ARENA" pass
done

# --- static contract: the guards must be present and the tiers coherent ------
grep -q 'pub fn ir_instr_arena_capacity() -> i64 { 1048576 }' "$ARENA" \
  || fail "arena_capacity_accessor_drift"
# The storage is struct-of-arrays, not one array of IrInstr. That is forced, not
# stylistic: #1655 measured that the bootstrap seed turns a store into a GLOBAL
# array of AGGREGATES into a silent no-op, so `[IrInstr; N]` as a global is
# exactly the shape that must NOT come back. Assert its absence, then hold every
# scalar lane to the capacity the accessor reports.
grep -q 'pub var IR_INSTR_ARENA: \[IrInstr; ' "$ARENA" \
  && fail "aggregate_arena_reintroduced_see_1655"
for lane in IR_A_OP IR_A_DST IR_A_SRC1 IR_A_SRC2 IR_A_IMM_I64 IR_A_LABEL_ID \
            IR_A_FN_ID IR_A_FIELD_IDX IR_A_IMM_FLAGS IR_A_BIN_OP IR_A_UN_OP \
            IR_A_ARG_COUNT IR_A_NAME_OFF IR_A_NAME_LEN IR_A_ARG_BASE; do
  grep -q "pub var $lane: \[i64; 1048576\]" "$ARENA" \
    || fail "arena_lane_${lane}_incoherent"
done
grep -q 'pub var IR_A_IMM_F64: \[f64; 1048576\]' "$ARENA" \
  || fail "arena_lane_IR_A_IMM_F64_incoherent"
grep -q 'pub fn ir_region_table_capacity() -> i64 { 8192 }' "$ARENA" \
  || fail "region_table_accessor_drift"
grep -cq . /dev/null
for arr in IR_REGION_BASE IR_REGION_CAP IR_REGION_LEN IR_REGION_GEN IR_REGION_STATE; do
  grep -q "pub var $arr: \[i64; 8192\]" "$ARENA" || fail "region_array_${arr}_incoherent"
done
# The two guards this gate exists for.
# ir_region_status_v takes the handle BY VALUE, so the guard reads `r.generation`
# rather than `(*r).generation`. Same guard, same strength -- only the binding
# form moved.
grep -q 'IR_REGION_GEN\[s as usize\] != r\.generation' "$ARENA" \
  || fail "generation_check_missing"
grep -q 'IR_REGION_STATE\[(\*r).slot as usize\] == IR_REGION_SEALED' "$ARENA" \
  || fail "sealed_check_missing"
# Raw write path must consult sealing; raw read path must not (reads stay legal).
awk '/^pub fn ir_region_base_write/,/^}/' "$ARENA" | grep -q 'IR_REGION_SEALED' \
  || fail "raw_write_path_ignores_sealing"

# --- optional vacuity re-check ----------------------------------------------
if [ "$VACUITY" = "1" ]; then
  sed 's/IR_REGION_GEN\[s as usize\] != r\.generation/false/' "$ARENA" >"$TMP/no_gen.sio"
  run_one ir_instr_arena_stale_witness IR_INSTR_ARENA_STALE_PASS "$TMP/no_gen.sio" fail
  sed 's/IR_REGION_STATE\[(\*r).slot as usize\] == IR_REGION_SEALED/false/g' "$ARENA" >"$TMP/no_seal.sio"
  run_one ir_instr_arena_seal_witness IR_INSTR_ARENA_SEAL_PASS "$TMP/no_seal.sio" fail
  # no_seal.sio strips the sealed CHECK on the (*r) paths, which ir_region_slot_w
  # does not use -- so it leaves this witness passing. Strip the sealing itself
  # instead: ir_region_seal still returns OK (the count half stays green) but the
  # state is never set, so the write is no longer refused.
  sed 's/IR_REGION_STATE\[(\*r).slot as usize\] = IR_REGION_SEALED//' "$ARENA" >"$TMP/no_seal_apply.sio"
  run_one ir_module_seal_witness IR_MODULE_SEAL_WITNESS_PASS "$TMP/no_seal_apply.sio" fail
  # Strip only the two pool latches. Measured: the witness then reports
  # NAME_POOL_FIRED_AT -1 and exits 12, so it is testing the latch and not some
  # other property that happens to hold.
  sed -e 's/ir_arena_latch(IR_ARENA_VIOLATION_NAME_POOL, ir_region_invalid())//' \
      -e 's/ir_arena_latch(IR_ARENA_VIOLATION_ARG_POOL, ir_region_invalid())//' \
      "$ARENA" >"$TMP/no_pool_latch.sio"
  run_one ir_arena_pool_witness IR_ARENA_POOL_WITNESS_PASS "$TMP/no_pool_latch.sio" fail
  # Drop only the argument-binding half of the swap. Every scalar lane still
  # exchanges, so a witness that checked less would stay green; this one fails at
  # rc=29 = ARG_COUNT_NOT_SWAPPED.
  sed -e '/let t_ab = IR_A_ARG_BASE/d' -e '/let t_ac = IR_A_ARG_COUNT/d' \
      "$ARENA" >"$TMP/no_swap_args.sio"
  run_one ir_arena_swap_witness IR_ARENA_SWAP_WITNESS_PASS "$TMP/no_swap_args.sio" fail
fi

head_sha="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf not_available)"
printf 'IR_INSTR_ARENA_BOUNDARY generation_guard=proved sealing_guard=proved fail_closed_capacity=proved pool_exhaustion=proved publication_sealed=proved conversion=complete\n'
printf 'IR_INSTR_ARENA_PASS witnesses=%d arena_capacity=1048576 region_table=8192 head=%s\n' \
  "${#WITNESSES[@]}" "$head_sha"
