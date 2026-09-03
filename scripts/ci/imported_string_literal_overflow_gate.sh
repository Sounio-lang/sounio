#!/usr/bin/env bash
# Regression gate: a >=256-byte string-literal call argument inside a `pub fn`
# reached only via `use` (an imported-module function, FN_EFFECTS bit 2048)
# must fail the build, not silently succeed with the literal truncated to
# empty.
#
# Root cause: tc_error_hard() -> tc_mark_failed() early-returns without
# setting TYPECHECK_FAILED for imported-module functions (by design, so a
# cosmetic type error in unused imported code doesn't block the whole build --
# see tc_mark_failed's own comment). compile_primary()'s string-literal-length
# guard used to rely solely on that path, so this specific error class -- which
# is data corruption (the literal is truncated), not a cosmetic type error --
# silently fell through the same escape hatch: an "error:" line printed, but
# compilation proceeded and shipped a working binary containing the wrong
# string. Fixed by setting TYPECHECK_FAILED directly in this one guard, same
# idiom as tc_unbalanced_braces (#1634).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURE_DIR="$ROOT_DIR/tests/compiler/imported_string_literal_overflow"
KEEP_WORK="${SOUNIO_IMPORTED_STRLIT_OVERFLOW_GATE_KEEP:-0}"

fail() {
  echo "[imported-strlit-overflow] FAIL: $*" >&2
  exit 1
}

WORK="$(mktemp -d /tmp/sounio-imported-strlit-overflow.XXXXXX)"
if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

COMPILER="${SOUNIO_IMPORTED_STRLIT_OVERFLOW_GATE_BIN:-$WORK/lean_single}"
if [[ -z "${SOUNIO_IMPORTED_STRLIT_OVERFLOW_GATE_BIN:-}" ]]; then
  SEED="${SOUNIO_IMPORTED_STRLIT_OVERFLOW_GATE_SEED:-$ROOT_DIR/bin/souc-lean-single-x86_64}"
  [[ -x "$SEED" ]] || fail "lean_single seed is missing or not executable: $SEED"
  if ! "$ROOT_DIR/scripts/dev/souc-build-lock.sh" \
      "$SEED" "$ROOT_DIR/self-hosted/compiler/lean_single.sio" "$COMPILER" \
      >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source lean_single build failed"
  fi
  chmod +x "$COMPILER"
fi
[[ -x "$COMPILER" ]] || fail "lean_single candidate is missing or not executable: $COMPILER"

ELF="$WORK/out.elf"
LOG="$WORK/compile.log"

set +e
"$COMPILER" "$FIXTURE_DIR/main.sio" "$ELF" --target x86_64-linux >"$LOG" 2>&1
RC=$?
set -e

[[ "$RC" -ne 0 ]] || fail "compile succeeded (rc=0) for an oversized imported-module string literal -- expected the build to fail"
grep -Fq 'error: string literal exceeds maximum of 256 bytes' "$LOG" || {
  cat "$LOG" >&2
  fail "expected diagnostic missing from compile output"
}
grep -Fxq 'typecheck: failed' "$LOG" || {
  cat "$LOG" >&2
  fail "typecheck: failed line missing -- TYPECHECK_FAILED was not set"
}
[[ ! -s "$ELF" ]] || fail "no-binary-on-failure violated: $ELF was written despite the compile failing"

echo "[imported-strlit-overflow] PASS: oversized imported-module string literal is rejected, no binary written"
