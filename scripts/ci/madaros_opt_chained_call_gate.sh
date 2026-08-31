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
ENTRY="opt_chained_call_accumulator_main.sio"
WANT="OPT_CHAINED_CALL_ACCUMULATOR_OK"

fail() { echo "MADAROS_OPT_CHAINED_CALL_FAIL: $*" >&2; exit 1; }

[[ -x "$MADAROS" ]] || { echo "madaros-opt-chained-call: no raw Madaros at $MADAROS -- skipping"; exit 0; }
[[ -f "$CASES/$ENTRY" ]] || fail "fixture missing: $CASES/$ENTRY"

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

plain="$(run_one "without -O")"
opt="$(run_one "with -O" -O)"

# Control: the fixture must PASS without -O. If it does not, the fixture is
# broken and a match between the two columns would prove nothing.
[[ "$plain" == *"$WANT"* ]] || fail "the fixture does not pass without -O (got: ${plain:-<empty>}) -- comparing the two builds would be vacuous"

[[ "$opt" == *"$WANT"* ]] || fail "with -O the fixture printed: ${opt:-<empty>}
  A chained assignment lost its accumulator. See ocp_mfi_dse in
  self-hosted/ir/opt_cleanup.sio: a call must be a barrier there, because its
  argument reads live in instr.call_args and the SRC1/SRC2 scan cannot see them."

[[ "$plain" == "$opt" ]] || fail "the two builds printed different text
  without -O: $plain
  with    -O: $opt"

echo "madaros-opt-chained-call: -O preserved the program's output (control: the fixture passes without -O too)."
