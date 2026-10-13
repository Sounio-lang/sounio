#!/usr/bin/env bash
# #1680: `madaros --self-test` must run to its summary line, and the self-tests
# pinned below must pass.
#
# Before #1680 the run died with SIGSEGV right after T110: const_prop's
# cp_empty_instrs() returns `[ir_nop(); 1024]`, and lean_single (which compiles
# Madaros) returned the wrong base for a struct repeat literal in expression
# position (fixed in #2847), so cp_make_test_func dereferenced a null element.
# On main (deede5c87) no test past T110 runs.
#
# Pinned tests (each was failing on main and is fixed by the change named):
#   T69  nc_min_frame_size: a stack-only vreg below NC_MAX_VREGS no longer
#        shrinks the frame sized for vregs above the table (native/lower_ir.sio).
#
# The remaining self-test failures are listed in the #1680 PR; most are
# expectations written before later optimizer passes changed the canonical
# output (e.g. T221 now gets `x << 3` where it expects `x * 8`). They are
# reported, not hidden: this gate does not accept a pass count, it requires the
# run to finish and the pinned tests to pass.
#
# The run writes artifacts/omega/*.selftest.bin relative to its working
# directory, so it runs from a scratch directory seeded with a copy of them,
# never in the checkout.
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
mkdir -p "$WORK/tree/artifacts"
cp -r "$ROOT/artifacts/omega" "$WORK/tree/artifacts/" 2>/dev/null || true

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
  fail "--self-test died with rc=$rc before finishing (last test: $(grep -E '^  (T[0-9]+ OK|FAIL)' "$WORK/out" | tail -1))"
fi
summary="$(grep -E '^compiler/main tests: [0-9]+/[0-9]+ passed' "$WORK/out" || true)"
[[ -n "$summary" ]] || { tail -n 20 "$WORK/out" >&2; fail "--self-test printed no summary line (rc=$rc)"; }
echo "$TAG: run finished: $summary (rc=$rc)"

for t in T69; do
  if ! grep -qE "^  $t OK" "$WORK/out"; then
    grep -E "^  (FAIL: )?$t( |$)" "$WORK/out" >&2 || true
    fail "pinned self-test $t did not pass"
  fi
  echo "$TAG: PASS pinned $t"
done
echo "MADAROS_SELF_TEST_GATE_OK"
