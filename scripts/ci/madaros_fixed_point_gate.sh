#!/usr/bin/env bash
# scripts/ci/madaros_fixed_point_gate.sh
#
# Does Madaros compile Madaros? Today: no. This gate is how we find out where
# it stops, in a way that names the stage.
#
# WHAT IS AND IS NOT TRUE TODAY. lean_single.sio reaches a byte-identical
# self-compiled fixed point and has for months. Madaros never has.
# scripts/ci/build_modular_madaros.sh performs exactly two compilations —
# committed ELF -> lean_single.sio -> seed, then seed -> main.sio -> Madaros.
# There is no second application and no gen1/gen2 pair to compare, so the
# property has never been measured, let alone held. CLAUDE.md:89 says so
# outright: "Do not describe Madaros itself as fixed-point-verified."
#
# THE LADDER. Each rung is named, and the gate reports the first one that
# fails rather than a bare non-zero exit:
#
#   check    gen1 typechecks self-hosted/compiler/main.sio
#   gen2     gen1 compiles main.sio to an ELF
#   run      gen2 answers --version as Madaros
#   gen3     gen2 compiles main.sio to an ELF
#   fixpoint gen2 and gen3 have identical executable payloads
#
# `gen1 == gen2` is NOT the property and must not be asserted. gen1 is built by
# a lean_single-derived seed, and the two backends need not agree on codegen.
# `gen2 == gen3` is the property: the first output Madaros produced about itself,
# reproduced by that output.
#
# usage:  MADAROS_BIN=/path/to/gen1.elf scripts/ci/madaros_fixed_point_gate.sh
#
# SOUNIO_MADAROS_FP_EXPECT=<rung> records the rung the tree is known to reach.
# The gate is green when it reaches exactly that rung and RED both when it falls
# short and when it goes further — a ratchet, so ground gained cannot be lost
# silently and ground gained is not absorbed silently either.
#
# The default is `check`. It was `none` when this gate was written: measured
# 2026-08-04 against origin/main 40116b661d, gen1 reported 3635 errors on
# main.sio and never reached lowering. As of 2026-08-05 `madaros check
# self-hosted/compiler/main.sio` exits 0 with zero diagnostics — the first time
# Madaros has typechecked its own entry point.
#
# The historical wall was named and loud, which is the point of the whole line:
#
#     imported_compile: typecheck ok
#     imported_compile: lower_done
#     IR lowering failed during merge: too many functions:
#         shared IR module capacity exceeded (max 8191 slots)
#
# The cap has since moved. This gate reads IR_MAX_FUNCS from the source instead
# of carrying another numeric copy: a stale 2048 comparison falsely classified
# a 13107-function merge as truncated when the tree's cap was 16384.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

. "$ROOT_DIR/scripts/lib/gate_assert.sh"
. "$ROOT_DIR/scripts/lib/souc_invoke.sh"
gate_name "madaros_fixed_point"

SRC="${SOUNIO_MADAROS_FP_SRC:-self-hosted/compiler/main.sio}"
EXPECT="${SOUNIO_MADAROS_FP_EXPECT:-gen2}"
MIN_INTO_ACC_DONE="${SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE:-40}"
MADAROS="${MADAROS_BIN:-}"
IR_MAX_FUNCS="$(sed -nE 's/^pub let IR_MAX_FUNCS: i64 = ([0-9]+).*$/\1/p' self-hosted/ir/ir.sio | head -1)"

RUNGS=(none check gen2 run gen3 fixpoint)

rung_index() {
  local want="$1" i=0
  for r in "${RUNGS[@]}"; do
    [[ "$r" == "$want" ]] && { printf '%s' "$i"; return 0; }
    i=$((i + 1))
  done
  printf '%s' "-1"
}

[[ "$(rung_index "$EXPECT")" -ge 0 ]] \
  || gate_fail "SOUNIO_MADAROS_FP_EXPECT=$EXPECT is not a rung; expected one of: ${RUNGS[*]}"
[[ "$MIN_INTO_ACC_DONE" =~ ^[0-9]+$ ]] \
  || gate_fail "SOUNIO_MADAROS_FP_MIN_INTO_ACC_DONE=$MIN_INTO_ACC_DONE is not a non-negative integer"
[[ "$IR_MAX_FUNCS" =~ ^[0-9]+$ ]] \
  || gate_fail "could not read IR_MAX_FUNCS from self-hosted/ir/ir.sio"

if [[ -z "$MADAROS" ]]; then
  echo "MADAROS_FIXED_POINT_SKIP: set MADAROS_BIN to a raw Madaros ELF (gen1)" >&2
  exit 0
fi

require_executable "$MADAROS"
if head -c2 "$MADAROS" 2>/dev/null | grep -q '#!'; then
  gate_fail "$MADAROS is a wrapper script, not a raw ELF — a wrapper resolves to whatever compiler happens to be installed, which makes this verdict unattributable to the tree under test"
fi

# The one guard build_modular_madaros.sh cannot make for itself: that script
# invokes its seed as `<seed> <src> <out>`, which is lean_single's argv. Handing
# it a Madaros produces a.out and exit 0. See scripts/lib/souc_invoke.sh.
KIND="$(souc_banner "$MADAROS")"
[[ "$KIND" == "madaros" ]] \
  || gate_fail "MADAROS_BIN identifies as '$KIND', not madaros. This gate compiles with the Madaros argv (\`build <src> <out>\`); giving that argv to lean_single, or lean_single's argv to Madaros, silently compiles the wrong thing to the wrong place."

require_file "$SRC"

WORK="${SOUNIO_MADAROS_FP_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/madaros-fp.XXXXXX")}"
mkdir -p "$WORK"

# module_frontend.sio:334 prefers $SOUNIO_STDLIB_PATH over the tree's own
# stdlib/. An inherited value compiles PARTS OF ANOTHER CHECKOUT into gen2,
# which would make a gen2/gen3 comparison say nothing about this tree.
if [[ -d "$ROOT_DIR/stdlib" ]]; then
  export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
fi
GEN2="$WORK/madaros.gen2"
GEN3="$WORK/madaros.gen3"

REACHED="none"
FAIL_DETAIL=""

reached() { REACHED="$1"; }

echo "MADAROS_FIXED_POINT_V1"
echo "gen1   $MADAROS"
echo "src    $SRC"
echo "work   $WORK"
echo "expect $EXPECT"
echo "min_into_acc_done $MIN_INTO_ACC_DONE"
echo "ir_max_functions $IR_MAX_FUNCS"
echo

# ── rung: check ───────────────────────────────────────────────────────────────
# `build` runs the checker first and refuses to lower a program with errors, so
# a failing check is the honest name for where this stops today.
echo "[rung check] gen1 typechecks $SRC"
( ulimit -s 524288 2>/dev/null || true; "$MADAROS" check "$SRC" ) >"$WORK/check.log" 2>&1
CHECK_RC=$?
ERRORS="$(grep -oE 'error\[E[0-9]+\]' "$WORK/check.log" | wc -l | tr -d ' ')"
echo "           rc=$CHECK_RC errors=$ERRORS"
if [[ "$ERRORS" -gt 0 ]]; then
  echo "           by code:"
  grep -oE 'error\[E[0-9]+\]' "$WORK/check.log" | sort | uniq -c | sort -rn | head -8 | sed 's/^/             /'
fi

if [[ "$CHECK_RC" -ne 0 || "$ERRORS" -gt 0 ]]; then
  FAIL_DETAIL="gen1 cannot typecheck the compiler's own entry point: $ERRORS errors, rc=$CHECK_RC. See $WORK/check.log"
else
  reached check

  # ── rung: gen2 ──────────────────────────────────────────────────────────────
  echo "[rung gen2] gen1 compiles $SRC"
  souc_compile "$MADAROS" "$SRC" "$GEN2" >"$WORK/gen2.log" 2>&1
  GEN2_RC=$?
  MERGED="$(grep -oE 'Merged IR: *[0-9]+' "$WORK/gen2.log" | grep -oE '[0-9]+' | tail -1)"
  INTO_ACC_DONE="$(grep -oE 'into_acc_done[[:space:]]+[0-9]+' "$WORK/gen2.log" | grep -oE '[0-9]+' | tail -1)"
  INTO_ACC_DONE="${INTO_ACC_DONE:-0}"
  FIRST_GEN2_FAILURE="$(grep -m1 -E 'println-poison|IR lowering failed|ir_[a-z_]+_failed|error\[E[0-9]+\]|Error: native code buffer overflow|Error: native relocation table overflow|Failed to write native binary|multimodule native thin-link compilation failed' "$WORK/gen2.log" || true)"
  GEN2_FAILURE_CONTEXT="$(grep -E 'first flagged in preseed stage|unresolved identifiers|lowering-error record|lowering errors:|raised at lower\.sio lines:|cause:' "$WORK/gen2.log" | head -8 || true)"
  echo "           rc=$GEN2_RC merged_ir_functions=${MERGED:-<none>}"
  echo "           into_acc_done=$INTO_ACC_DONE minimum=$MIN_INTO_ACC_DONE"
  if [[ -n "$FIRST_GEN2_FAILURE" ]]; then
    echo "           first_failure=$FIRST_GEN2_FAILURE"
  fi
  if [[ -n "$GEN2_FAILURE_CONTEXT" ]]; then
    echo "           failure_context:"
    printf '%s\n' "$GEN2_FAILURE_CONTEXT" | sed 's/^/             /'
  fi
  if [[ -n "$MERGED" ]] && [[ "$MERGED" -ge "$IR_MAX_FUNCS" ]]; then
    gate_fail "merged IR reached IR_MAX_FUNCS ($MERGED >= $IR_MAX_FUNCS). ir_merge_modules_into may have stopped copying at the cap, so a progress/rung verdict would be attributable to a potentially truncated module. See scripts/ci/madaros_ir_capacity_probe.sh"
  fi
  if [[ "$GEN2_RC" -ne 0 || ! -s "$GEN2" ]]; then
    FAIL_DETAIL="gen1 typechecked $SRC but produced no ELF (rc=$GEN2_RC). See $WORK/gen2.log"
  else
    chmod +x "$GEN2"
    reached gen2

    # ── rung: run ─────────────────────────────────────────────────────────────
    echo "[rung run] gen2 identifies itself"
    GEN2_KIND="$(souc_banner "$GEN2")"
    echo "           banner=$GEN2_KIND"
    if [[ "$GEN2_KIND" != "madaros" ]]; then
      FAIL_DETAIL="gen2 is an ELF but does not run as Madaros (banner=$GEN2_KIND) — the payload is wrong, not merely different"
    else
      reached run

      # ── rung: gen3 ──────────────────────────────────────────────────────────
      echo "[rung gen3] gen2 compiles $SRC"
      souc_compile "$GEN2" "$SRC" "$GEN3" >"$WORK/gen3.log" 2>&1
      GEN3_RC=$?
      echo "           rc=$GEN3_RC"
      if [[ "$GEN3_RC" -ne 0 || ! -s "$GEN3" ]]; then
        FAIL_DETAIL="gen2 runs but cannot compile the source it was built from (rc=$GEN3_RC). See $WORK/gen3.log"
      else
        chmod +x "$GEN3"
        reached gen3

        # ── rung: fixpoint ────────────────────────────────────────────────────
        echo "[rung fixpoint] gen2 vs gen3 executable payloads"
        if bash "$ROOT_DIR/scripts/lib/compare_executable_payloads.sh" "$GEN2" "$GEN3" >"$WORK/cmp.log" 2>&1; then
          echo "           identical"
          reached fixpoint
        else
          FAIL_DETAIL="gen2 and gen3 differ — Madaros compiles itself but not to a fixed point. $(head -3 "$WORK/cmp.log")"
        fi
      fi
    fi
  fi
fi

echo
REACHED_IDX="$(rung_index "$REACHED")"
EXPECT_IDX="$(rung_index "$EXPECT")"
echo "reached  $REACHED"
echo "expected $EXPECT"

if [[ "$REACHED_IDX" -lt "$EXPECT_IDX" ]]; then
  gate_fail "stopped at rung '$REACHED' but this tree is recorded as reaching '$EXPECT'.
$FAIL_DETAIL"
fi

if [[ "${INTO_ACC_DONE:-0}" -lt "$MIN_INTO_ACC_DONE" ]]; then
  gate_fail "self-build progress regressed: into_acc_done=${INTO_ACC_DONE:-0}, required >=$MIN_INTO_ACC_DONE.
First failure: ${FIRST_GEN2_FAILURE:-not found}. See $WORK/gen2.log"
fi

if [[ "$REACHED_IDX" -gt "$EXPECT_IDX" ]]; then
  gate_fail "reached rung '$REACHED', further than the recorded '$EXPECT'. This is PROGRESS, and it is red on purpose: raise SOUNIO_MADAROS_FP_EXPECT (and the default in this file) to '$REACHED' so the ground gained cannot be lost silently."
fi

if [[ "$REACHED" == "fixpoint" ]]; then
  gate_pass "Madaros compiles Madaros to a fixed point: gen2 == gen3"
else
  gate_pass "reached rung '$REACHED' as recorded; the next wall is '${RUNGS[$((REACHED_IDX + 1))]}' — $FAIL_DETAIL"
fi
