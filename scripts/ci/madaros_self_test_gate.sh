#!/usr/bin/env bash
# #1680: `madaros --self-test` must run to its summary line, and the self-tests pinned
# below must pass. The remaining failures are listed in the #1680 PR stack.
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

for t in T69 T57 T61 T70 T84 T85 T93 T94 T156 T187 T227 T229; do
  if ! grep -qE "^  $t OK" "$WORK/out"; then
    grep -E "^  (FAIL: )?$t( |$)" "$WORK/out" >&2 || true
    fail "pinned self-test $t did not pass"
  fi
done
echo "$TAG: PASS 12 pinned self-tests"
echo "MADAROS_SELF_TEST_GATE_OK"
