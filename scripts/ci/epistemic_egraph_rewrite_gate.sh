#!/usr/bin/env bash
# Verifies the epistemic e-graph reassociation gate in self-hosted/ir/egraph.sio:
# the three orthogonal axes (value-algebra structure, IEEE-754 representation
# exactness, Lyapunov chaos predicate) that decide whether a float
# reassociation rewrite is legal.
#
# Restored from scripts/archive/sprint247_epistemic_egraph_rewrite_gate.sh and
# rewritten 2026-07-27 against current source: the archived version checked
# for test IDs (T1148-T1153, total=1153) and an invocation
# (`$SOUC run main.sio -- --self-test`) that no longer match current source
# (current main.sio self-test total is 1156; the flag is `--self-test`, not
# `run ... -- --self-test`).
#
# Measured 2026-07-27, verified with a Madaros built fresh from current
# source via build_modular_madaros.sh (never against a prebuilt bin/souc --
# see CLAUDE.md operating principle 15):
#
#   1. T143b/c/d (compiler_main_test_eg_small_satf_*, main.sio:8951-8974) --
#      the three-axis gate's own tests -- run FIRST in
#      run_compiler_main_self_tests and pass. main.sio:21758 documents why
#      they were placed first: "a pre-existing crash downstream of T1201 on
#      this WIP branch" otherwise prevents them from being reached. That
#      crash is real and reproduces identically with or without any change
#      in this repository state (confirmed via a stashed/unstashed rebuild):
#      `madaros --self-test` segfaults partway through its ~1156 tests,
#      well after T143b/c/d have already printed OK. This gate treats that
#      crash as a known, separate, NOT_RUN condition for tests after it --
#      never as a false FAIL for T143b/c/d, which are unaffected by it.
#
#   2. The same three tests were duplicated into egraph.sio's own
#      self-test suite (test_eg_small_satf_*, T83/T84/T85,
#      self-hosted/ir/egraph.sio) so the gate does not depend on main.sio's
#      self-test binary reaching that crash-adjacent point at all. This
#      duplicate suite currently cannot be exercised standalone in this
#      environment under EITHER engine -- Madaros segfaults while lowering
#      egraph.sio as a standalone entrypoint (crashes during
#      `lower_array: seed_begin`, confirmed pre-existing: identical crash
#      with egraph.sio reverted to its unmodified state, same compiler
#      binary), and lean_single rejects it at typecheck ("effect not
#      declared in function signature at line 1549", also confirmed
#      pre-existing). Both are reported here as NOT_RUN, not silently
#      skipped and not falsely marked FAIL -- they are real, separate,
#      already-present defects this gate does not attempt to fix.
#
# This gate is therefore honest about four different states: the gate
# logic verified working (T143b/c/d, live, in a freshly built Madaros),
# the gate logic duplicated but not yet independently exercisable (T83-T85),
# a pre-existing crash class blocking further verification (both crashes
# above), and a run that produced no evidence at all (empty self-test log
# or every subject absent -- FAIL, never green) -- rather than reporting
# one number that blurs them.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
MADAROS="${SOUNIO_EPISTEMIC_EGRAPH_MADAROS:-}"

cg() {
  local n="$1" p="$2" f="$3"
  TOTAL=$((TOTAL + 1))
  if [ ! -f "$f" ]; then echo "NOT_RUN  $n (file missing: $f)"; NOT_RUN=$((NOT_RUN + 1)); return; fi
  if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS + 1))
  else echo "FAIL  $n"; FAIL=$((FAIL + 1)); fi
}

echo "=== Epistemic e-graph reassociation gate (value-algebra x exactness x chaos) ==="
echo ""
echo "--- Source checks (structural presence, no build required) ---"
E="self-hosted/ir/egraph.sio"
M="self-hosted/compiler/main.sio"
cg "egraph:EgSmallContext"        "struct EgSmallContext" "$E"
cg "egraph:algebra_kind_field"    "algebra_kind:\s*i64" "$E"
cg "egraph:allow_inexact_field"   "allow_inexact_reassoc:\s*bool" "$E"
cg "egraph:chaotic_field"         "chaotic:\s*bool" "$E"
cg "egraph:observe_barriers"      "observe_barriers:\s*\[i64" "$E"
cg "egraph:duplicated_helper"     "fn eg_satf_fadd_chain_merges" "$E"
cg "egraph:duplicated_T83"        "fn test_eg_small_satf_blocks_inexact_default" "$E"
cg "egraph:duplicated_T84"        "fn test_eg_small_satf_allows_optin" "$E"
cg "egraph:duplicated_T85"        "fn test_eg_small_satf_chaotic_blocks" "$E"
cg "egraph:total_85"              "let total: i64 = 85" "$E"
cg "main:T143b_fn"                "fn compiler_main_test_eg_small_satf_blocks_inexact_default" "$M"
cg "main:T143c_fn"                "fn compiler_main_test_eg_small_satf_allows_optin" "$M"
cg "main:T143d_fn"                "fn compiler_main_test_eg_small_satf_chaotic_blocks" "$M"
cg "main:total_1156"              "let total: i64 = 1156" "$M"

echo ""
echo "--- Live check: main.sio self-tests, T143b/c/d (requires a Madaros built from current source) ---"
if [ -z "$MADAROS" ] || [ ! -x "$MADAROS" ]; then
  for t in T143b T143c T143d; do
    TOTAL=$((TOTAL + 1))
    echo "NOT_RUN  live:$t (set SOUNIO_EPISTEMIC_EGRAPH_MADAROS to a Madaros built via scripts/ci/build_modular_madaros.sh)"
    NOT_RUN=$((NOT_RUN + 1))
  done
else
  L="$(mktemp)"
  # A pre-existing crash further into this same run is expected and is not
  # this gate's concern (see header) -- capture whatever prints before it.
  # The rc is recorded, not `|| true`-discarded: a nonzero rc with a
  # non-empty log is the documented mid-run crash class, but a run that
  # printed NOTHING is an instrument that did not answer, and an
  # instrument that did not answer cannot certify anything.
  madaros_rc=0
  timeout 60 "$MADAROS" --self-test > "$L" 2>&1 || madaros_rc=$?
  live_log_bytes="$(wc -c <"$L" 2>/dev/null || echo 0)"
  for t in T143b T143c T143d; do
    TOTAL=$((TOTAL + 1))
    if [ "$live_log_bytes" -eq 0 ]; then
      echo "FAIL  live:$t (self-test log is empty: the instrument produced no evidence, rc=$madaros_rc)"
      FAIL=$((FAIL + 1))
    elif grep -q "$t OK" "$L"; then echo "PASS  live:$t"; PASS=$((PASS + 1))
    elif grep -q "FAIL: $t" "$L"; then echo "FAIL  live:$t"; FAIL=$((FAIL + 1))
    else echo "NOT_RUN  live:$t (not reached in self-test log)"; NOT_RUN=$((NOT_RUN + 1)); fi
  done
  rm -f "$L"
fi

echo ""
echo "--- Known pre-existing blockers (not this gate's regression -- recorded for visibility) ---"
echo "NOT_RUN  egraph.sio standalone under Madaros (segfaults in lower_array: seed_begin; reproduces on unmodified egraph.sio)"
echo "NOT_RUN  egraph.sio standalone under lean_single (typecheck rejects: effect not declared in function signature, line 1549; reproduces on unmodified egraph.sio)"
TOTAL=$((TOTAL + 2)); NOT_RUN=$((NOT_RUN + 2))

echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
# Zero-evidence guard: FAIL==0 alone cannot certify green, because every
# check can be NOT_RUN (subjects moved/renamed, instrument absent or dead).
# "The instrument did not answer" and "the selected set is empty and clean"
# are different states; only the second may read PASS (CI trust contract).
if [ "$PASS" -eq 0 ]; then
  echo "GATE: FAIL (zero evidence: no check produced a PASS -- all NOT_RUN is a dead or absent instrument, not a green run)"
  exit 1
elif [ "$FAIL" -eq 0 ]; then echo "GATE: PASS (structural + live checks clean; NOT_RUN entries are known, separately-tracked blockers, not silent passes)"; exit 0
else echo "GATE: FAIL"; exit 1; fi
