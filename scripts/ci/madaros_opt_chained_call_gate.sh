#!/usr/bin/env bash
# The optimiser must not change what a program prints.
#
# WHY THIS EXISTS
#
# `-O` is advertised in the compiler's own --help, so users are invited to it,
# and until this gate nothing in CI compiled anything with it. Measured
# 2026-08-31 over tests/run-pass: 632 of 1724 buildable programs behaved
# differently under -O, 16 of them silently -- same exit status, different
# output. See docs/audit/OPTIMIZER_DIVERGENCE_2026-08-31.md.
#
# The root cause was one line. ocp_mfi_dse decided a store was dead by scanning
# an instruction's SRC1/SRC2 fields, but a call reads its arguments from
# instr.call_args, a Box list that peel does not walk -- a limitation the code
# states in a comment on ocp_mfi_dce_once and reasons is safe because calls are
# never removed. That reasoning covers removing the CALL. It does not cover
# removing the DEF whose only consumer is a call argument, which is what
# happened: `x = f(x, k)` chains lost every store but the last, so each call
# received the pre-chain value of x.
#
# WHAT THIS MEASURES
#
# One fixture, built twice by the same compiler, differing only in -O. If the
# outputs differ, the optimiser changed the meaning of the program -- which is
# wrong by definition and needs no adjudication between engines.
#
# The fixture must be MULTI-MODULE: the defect needs a call whose arguments
# cross a module boundary. A local function call is inlined into the same
# region and the peel sees the reads.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_RAW_BIN:-${MADAROS_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}}"
CASES="$ROOT_DIR/tests/multimodule"

# entry|expected-token. One line per defect this gate has caught.
FIXTURES="opt_chained_call_accumulator_main.sio|OPT_CHAINED_CALL_ACCUMULATOR_OK
opt_cse_branch_dominance_main.sio|OPT_CSE_BRANCH_DOMINANCE_OK
opt_dedup_imm_stale_reg_main.sio|OPT_DEDUP_IMM_STALE_REG_OK
opt_copy_prop_swap_temp_main.sio|OPT_COPY_PROP_SWAP_TEMP_OK
opt_wide_limb_span_main.sio|OPT_WIDE_LIMB_SPAN_OK"

fail() { echo "MADAROS_OPT_CHAINED_CALL_FAIL: $*" >&2; exit 1; }

[[ -x "$MADAROS" ]] || { echo "madaros-opt-chained-call: no raw Madaros at $MADAROS -- skipping"; exit 0; }


run_one() {  # run_one <label> <flag...>
  local label="$1"; shift
  local elf out rc
  elf="$(mktemp /tmp/opt-chained-XXXXXX.elf)"
  # Build from inside the fixture directory, as the sibling multimodule gates do:
  # sibling-module resolution is relative to the entry file, and the compiler
  # needs the raised stack. Building by path from the repo root works locally and
  # failed in CI -- the first version of this gate did exactly that.
  if ! ( cd "$CASES" && ulimit -s 524288 2>/dev/null || true
         "$MADAROS" "$@" build "$ENTRY" "$elf" ) >/dev/null 2>&1; then
    rm -f "$elf"; fail "$label: the fixture did not build"
  fi
  chmod +x "$elf"
  out="$("$elf" 2>/dev/null || true)"
  rm -f "$elf"
  printf '%s' "$out"
}

checked=0
while IFS='|' read -r ENTRY WANT; do
  [[ -n "$ENTRY" ]] || continue
  [[ -f "$CASES/$ENTRY" ]] || fail "fixture missing: $CASES/$ENTRY"

  plain="$(run_one "$ENTRY without -O")"
  opt="$(run_one "$ENTRY with -O" -O)"

  # Control: the fixture must PASS without -O. If it does not, the fixture is
  # broken and a match between the two columns would prove nothing.
  [[ "$plain" == *"$WANT"* ]] || fail "$ENTRY does not pass without -O (got: ${plain:-<empty>}) -- comparing the two builds would be vacuous"

  [[ "$opt" == *"$WANT"* ]] || fail "$ENTRY with -O printed: ${opt:-<empty>}
  Both defects this gate guards live in self-hosted/ir/opt_cleanup.sio and share a
  shape: a peel carrying state across a boundary it cannot see past.
    ocp_mfi_dse must treat a CALL as a barrier -- its argument reads live in
      instr.call_args and the SRC1/SRC2 scan cannot see them.
    ocp_mfi_cse must drop its table at a BASIC-BLOCK boundary -- it has no
      dominance information, so a match from another block is not substitutable.
    ocp_mfi_dedup_imm must invalidate a register on any WRITE to it -- a
      recorded immediate goes stale the moment something else assigns there."

  [[ "$plain" == "$opt" ]] || fail "$ENTRY: the two builds printed different text
  without -O: $plain
  with    -O: $opt"

  checked=$((checked + 1))
done <<< "$FIXTURES"

echo "madaros-opt-chained-call: -O preserved the output of $checked fixture(s) (control: each passes without -O too)."
