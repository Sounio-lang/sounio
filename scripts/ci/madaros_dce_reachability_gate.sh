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

gate_pass "the reachability pass keeps all 602 live functions and still drops 300 of 303"
