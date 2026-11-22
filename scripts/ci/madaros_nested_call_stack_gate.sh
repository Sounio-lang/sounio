#!/usr/bin/env bash
# madaros_nested_call_stack_gate.sh -- pin the SIO-378 checker frame diet.
#
# WHAT THIS ASSERTS
#
# checker_check_call_expr_inplace carried a ~1.3MB static frame: per-call-site
# SRET and aggregate-materialize slots accumulated in NEXT_SLOT without reuse,
# so every nesting level paid ~1.3MB and any nesting past depth 4-5 SIGSEGVd
# the checker at an 8MB stack. The function is now split into special-forms,
# resolve-callee, direct-sig and tyfn helpers behind a thin shell (~0.15MB
# per level). This gate runs the depth-6 regression test
# (tests/run-pass/nested_call_depth6_sio378.sio, builtin AND user-fn nesting)
# through madaros check with RLIMIT_STACK=8192: pre-fix it dies with SIGSEGV,
# post-fix it passes with wide headroom (depth 12 fits in 4MB, depth 20 in 8MB).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MADAROS="${SOUNIO_MADAROS_STACK_BIN:-$ROOT/artifacts/self-hosted/madaros}"
if [[ ! -x "$MADAROS" ]]; then
    echo "FAIL: no Madaros at $MADAROS -- build it from current sources or set SOUNIO_MADAROS_STACK_BIN" >&2
    exit 2
fi
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/nested-call-stack-gate.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
( ulimit -s 8192; "$MADAROS" check "$ROOT/tests/run-pass/nested_call_depth6_sio378.sio" -o "$TMP/out.mad" )
echo SIO_378_STACK_GATE_OK
