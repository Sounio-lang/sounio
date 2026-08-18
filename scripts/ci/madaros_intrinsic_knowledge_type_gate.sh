#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIXTURES="$ROOT_DIR/scripts/ci/fixtures/madaros_intrinsic_knowledge_type"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "[intrinsic-knowledge-type] FAIL: $*" >&2
  exit 1
}

RAW="${MADAROS_RAW_BIN:-}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-current-source"
  bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" \
    >"$WORK/build.log" 2>&1 || {
      cat "$WORK/build.log" >&2
      fail "current-source Madaros build failed"
    }
fi
[[ -x "$RAW" ]] || fail "Madaros ELF is not executable: $RAW"

run_madaros() {
  MADAROS_RAW_BIN="$RAW" MADAROS_STACK_KB=524288 \
    "$ROOT_DIR/bin/madaros" "$@"
}

run_madaros check "$FIXTURES/positive.sio" >"$WORK/positive.check" 2>&1 || {
  cat "$WORK/positive.check" >&2
  fail "positive control no longer typechecks"
}
run_madaros run "$FIXTURES/positive.sio" >"$WORK/positive.run" 2>&1 || {
  cat "$WORK/positive.run" >&2
  fail "positive control did not execute"
}
grep -q '^INTRINSIC_KNOWLEDGE_VALUE_OK$' "$WORK/positive.run" || {
  cat "$WORK/positive.run" >&2
  fail "Knowledge<T> payload remained wrong at runtime"
}

set +e
run_madaros check "$FIXTURES/reject_nonknowledge.sio" >"$WORK/reject.check" 2>&1
reject_rc=$?
set -e
[[ "$reject_rc" -ne 0 ]] || fail "non-Knowledge acknowledge argument was accepted"
grep -q 'error\[E009\]' "$WORK/reject.check" || {
  cat "$WORK/reject.check" >&2
  fail "non-Knowledge control was rejected for an unexpected reason"
}

echo "status=pass"
echo "metrics {total=3,passed=3,failed=0,not_run=0}"
echo "[intrinsic-knowledge-type] PASS: real Knowledge<T> signature preserves payload and rejects scalar impostors"
