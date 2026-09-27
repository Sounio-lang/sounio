#!/usr/bin/env bash
# scripts/ci/madaros_dce_reachability_gate.sh
#
# The cross-module dead-code pass must not delete live code, and must still
# delete dead code. Both arms, because either one alone is passable by a pass
# that does nothing.
#
# WHAT THIS CAUGHT. spec_dce_unreachable_item_fns marked reachable names in a
# 512-entry list scanned linearly, and spec_dce_hash_insert returned false —
# silently — when the list was full. spec_dce_filter_items then DELETED every
# top-level fn whose name it could not find. So past 512 distinct reachable
# names the pass began removing functions that were called.
#
# Measured 2026-08-04 against a Madaros built from main, on the 600-link chain
# in tests/multimodule/dce_reach:
#
#     spec_dce: item_fns 602 -> 512 marks=512
#     exit 8, expected 99, compiler rc=0, no diagnostic
#
# Ninety live functions gone and a running binary with the wrong answer. It
# compounded: "did the mark set change" was `count > before`, so a full table
# reported no change and the fixed-point loop exited SATISFIED.
#
# usage:  MADAROS_BIN=/path/to/madaros scripts/ci/madaros_dce_reachability_gate.sh

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "madaros_dce_reachability"

MADAROS="${MADAROS_BIN:-${SOUNIO_MADAROS_DCE_BIN:-}}"
if [[ -z "$MADAROS" ]]; then
  echo "MADAROS_DCE_REACHABILITY_SKIP: set MADAROS_BIN to a raw Madaros ELF" >&2
  exit 0
fi
require_executable "$MADAROS"
if head -c2 "$MADAROS" 2>/dev/null | grep -q '#!'; then
  gate_fail "$MADAROS is a wrapper script, not a raw ELF"
fi

CASES="$ROOT_DIR/tests/multimodule/dce_reach"
require_file "$CASES/dce_chain_main.sio"
require_file "$CASES/dce_prune_main.sio"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/madaros-dce-reach.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# run <name> <entry> <expected exit> <expected before> <expected after>
run_case() {
  local name="$1" entry="$2" want_exit="$3" want_before="$4" want_after="$5"
  local log="$WORK/$name.log" elf="$WORK/$name.elf"

  ( cd "$CASES" && ulimit -s 524288 2>/dev/null || true
    SOUNIO_MM_SPEC_TRACE=1 "$MADAROS" build "$entry" "$elf" ) >"$log" 2>&1
  local build_rc=$?

  [[ "$build_rc" -eq 0 ]] || gate_fail "$name: compile failed rc=$build_rc
$(tail -n 20 "$log")"
  [[ -s "$elf" ]] || gate_fail "$name: compile reported success and emitted no ELF"
  chmod +x "$elf"

  # The trace line is the instrument. If it is absent the pass did not run at
  # all, and a green result below would mean nothing — that is exactly how a
  # DCE bug hides.
  local trace
  trace="$(grep -oE 'spec_dce: item_fns [0-9]+ -> [0-9]+ marks=[0-9]+' "$log" | tail -1)"
  require_nonempty "$trace" "$name: no spec_dce trace — the pass never ran, so this case measures nothing"

  local before after
  before="$(printf '%s' "$trace" | awk '{print $3}')"
  after="$(printf '%s' "$trace" | awk '{print $5}')"

  [[ "$before" == "$want_before" ]] \
    || gate_fail "$name: the pass saw $before top-level fns, expected $want_before — the fixture changed, so the numbers below are about a different program"
  [[ "$after" == "$want_after" ]] \
    || gate_fail "$name: $trace
expected '$want_after' surviving. $(if [[ "$after" -lt "$want_after" ]]; then echo 'FEWER survived than are reachable: the pass is deleting live code.'; else echo 'MORE survived than should: the pass stopped pruning.'; fi)"

  "$elf" >"$WORK/$name.out" 2>&1
  local run_rc=$?
  [[ "$run_rc" -eq "$want_exit" ]] \
    || gate_fail "$name: the binary exited $run_rc, expected $want_exit. The trace said the right functions survived, so this is the answer being wrong for some other reason.
$(cat "$WORK/$name.out")"

  echo "  $name  $trace  exit=$run_rc"
}

echo "MADAROS_DCE_REACHABILITY_V1"
echo "madaros $MADAROS"

# ARM 1 — must not delete live code. 600 links, all reachable.
run_case chain dce_chain_main.sio 99 602 602

# ARM 2 — must still delete dead code. 300 unreachable, one live.
run_case prune dce_prune_main.sio 7 303 3

# ARM 3 — the capacity boundary, which is a different limit and fails in a
# different way. 6000 live functions: over IR_MAX_FUNCS at both 2048 and 4096,
# under the current 8192. Two backend guards read the literal 2048 while the
# arrays they guard were sized from the constant, so from exactly 2046 functions
# the produced binary exited 0 and printed nothing. This arm is deliberately in the same gate
# as the DCE arms because both are the same failure shape — a table that stops
# accepting entries without saying so — and because a compiler that passes the
# first two and drops symbols past 2046 is not a working compiler.
CAP="$ROOT_DIR/tests/multimodule/ir_capacity"
if [[ -f "$CAP/ir_capacity_main.sio" ]]; then
  CAP_ELF="$WORK/ir_capacity.elf"
  ( cd "$CAP" && ulimit -s 524288 2>/dev/null || true
    "$MADAROS" build ir_capacity_main.sio "$CAP_ELF" ) >"$WORK/capacity.log" 2>&1
  cap_rc=$?
  [[ "$cap_rc" -eq 0 && -s "$CAP_ELF" ]] \
    || gate_fail "capacity: 6000 functions failed to compile (rc=$cap_rc)
$(tail -n 12 "$WORK/capacity.log")"
  merged="$(grep -oE 'Merged IR: *[0-9]+' "$WORK/capacity.log" | grep -oE '[0-9]+' | tail -1)"
  require_nonempty "$merged" "capacity: no 'Merged IR:' line — cannot tell how many functions were kept"
  [[ "$merged" -ge 6000 ]] \
    || gate_fail "capacity: only $merged functions reached the merged IrModule, expected at least 6000 — something truncated silently"
  chmod +x "$CAP_ELF"
  "$CAP_ELF" >"$WORK/capacity.out" 2>&1
  cap_exit=$?
  [[ "$cap_exit" -eq 228 ]] \
    || gate_fail "capacity: the 6000-function binary exited $cap_exit, expected 228. It compiled and merged $merged functions, so this is a symbol or code-offset table dropping entries past its own bound — the exact failure the literal 2048 guards in native/elf.sio and native/codegen_x86_linux.sio used to produce."
  echo "  capacity  merged=$merged  exit=$cap_exit"
fi

gate_pass "keeps 602 live functions, drops 300 of 303 dead, and carries 6000 functions through to a correct binary"
