#!/usr/bin/env bash
# #1680: `madaros --self-test` must run to its summary line, every self-test must pass
# (rc 0 and an N/N summary), and the pinned self-tests below are named for the record.
#
# Before #1680 the run died with SIGSEGV right after T110: const_prop's
# cp_empty_instrs() returns `[ir_nop(); 1024]`, and lean_single (which compiles
# Madaros) returned the wrong base for a struct repeat literal in expression
# position (fixed in #2847), so cp_make_test_func dereferenced a null element.
#
# Each pinned self-test below failed before the change named next to it.
#   T69  nc_min_frame_size keeps the frame sized for high vregs (#2937).
#   T57  runtime context reports tracing_gc = false (d9dab3b11), as the test now expects.
#   T61  contract JSON is written under /tmp, so it no longer needs ./artifacts/omega.
#   T70  the legacy vreg fixture has a body, so it reaches the NC_MAX_VREGS refusal.
#   T84  prologue checked before the epilogue reuses the global code buffer.
#   T85, T156  contract GC flags expected false; JSON numbers present (#2948).
#   T93, T94  LICM fixtures allocate their own IR region.
#   T187 runtime context size 248 (998e230b7).
#   T227, T229  write_file returns the byte count, not 0.
#   T452-T455  x & ~x / x | ~x fold again: the const-fold blocks run in one scan.
#   T24  layout_reorder_functions permutes by cycles (the function table is shared).
#   T98-T100  only unsigned x / 2^n, x % 2^n are strength-reduced.
#   T318, T322  LICM inserts the hoisted load before the loop label.
#   T373, T374  Block AH runs (it was nested inside the OpMul branch).
#   T685  signed (x*3)/4 keeps its division.
#   T692-T694, T704-T706, T735  Blocks CG/CI patch the constant feeding the inner op.
#   T1048  ~x ^ ~y -> x ^ y records the rewritten XOR operands (Block EK chain).
#
# The suite writes its scratch files under /tmp and removes them, so the run no
# longer depends on, or rewrites, the directory it starts in. It still runs from
# a scratch directory so that a regression in that respect cannot dirty a checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="madaros_self_test_gate"
fail() { echo "$TAG: FAIL $*" >&2; exit 1; }

RAW="${MADAROS_RAW_BIN:-${SOUNIO_MADAROS_BIN:-}}"
[[ -n "$RAW" && -x "$RAW" ]] || { echo "$TAG: set MADAROS_RAW_BIN to a Madaros ELF" >&2; exit 2; }
RAW="$(cd "$(dirname "$RAW")" && pwd)/$(basename "$RAW")"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-self-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/tree"

rc=0
(
  cd "$WORK/tree"
  ulimit -s "${MADAROS_STACK_KB:-524288}" 2>/dev/null || ulimit -s unlimited 2>/dev/null || true
  exec timeout 600 "$RAW" --self-test
) >"$WORK/out" 2>&1 || rc=$?

# rc 1 = some self-test failed; anything else (139 SIGSEGV, 124 timeout) is a
# crash or hang, which is what #1680 was.
if [[ $rc -ne 0 && $rc -ne 1 ]]; then
  tail -n 20 "$WORK/out" >&2
  fail "--self-test died with rc=$rc before finishing (last test: $(grep -E '^  (T[0-9a-z]+ OK|FAIL)' "$WORK/out" | tail -1))"
fi
summary="$(grep -E '^compiler/main tests: [0-9]+/[0-9]+ passed' "$WORK/out" || true)"
[[ -n "$summary" ]] || { tail -n 20 "$WORK/out" >&2; fail "--self-test printed no summary line (rc=$rc)"; }
echo "$TAG: run finished: $summary (rc=$rc)"
[[ -z "$(find "$WORK/tree" -type f | head -1)" ]] || fail "--self-test wrote into its working directory"

for t in T69 T57 T61 T70 T84 T85 T93 T94 T156 T187 T227 T229 T452 T453 T454 T455 T24 T98 T99 T100 T318 T322 T373 T374 T685 T692 T693 T694 T704 T705 T706 T735 T1048; do
  if ! grep -qE "^  $t OK" "$WORK/out"; then
    grep -E "^  (FAIL: )?$t( |$)" "$WORK/out" >&2 || true
    fail "pinned self-test $t did not pass"
  fi
done
echo "$TAG: PASS 33 pinned self-tests"
if grep -qE '^  FAIL' "$WORK/out"; then
  grep -E '^  FAIL' "$WORK/out" >&2
  fail "self-tests failed"
fi
[[ $rc -eq 0 ]] || fail "--self-test exited rc=$rc without a FAIL line"
read -r got want < <(sed -E 's#^compiler/main tests: ([0-9]+)/([0-9]+) passed.*#\1 \2#' <<<"$summary")
[[ "$got" == "$want" && "$want" -gt 0 ]] || fail "summary is not N/N: $summary"
echo "$TAG: PASS all $want self-tests"
echo "MADAROS_SELF_TEST_GATE_OK"
