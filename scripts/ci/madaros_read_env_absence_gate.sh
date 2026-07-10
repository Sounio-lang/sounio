#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${MADAROS_READ_ENV_SOUC_BIN:-$ROOT_DIR/bin/souc}"
CASE_TIMEOUT="${MADAROS_READ_ENV_TIMEOUT:-150}"
SOURCE="${MADAROS_READ_ENV_SOURCE:-tests/stdlib/eisa/test_eisa_evm_v1.sio}"
LOG="$(mktemp "${TMPDIR:-/tmp}/madaros-read-env-absence.XXXXXX.log")"

cleanup() {
  rm -f "$LOG"
}
trap cleanup EXIT

fail() {
  echo "[madaros-read-env-absence] FAIL: $*" >&2
  exit 1
}

[[ -x "$SOUC_BIN" ]] || fail "$SOUC_BIN is not executable"
[[ -s "$SOURCE" ]] || fail "$SOURCE is missing or empty"

echo "[madaros-read-env-absence] source=$SOURCE"
if ! scripts/dev/souc-build-lock.sh timeout "$CASE_TIMEOUT" \
  env -u SOUNIO_MODULE_FRONTEND_LOWER_TRACE \
    -u SOUNIO_DUMP_MERGED_CALLS \
    SOUNIO_SOUC_ENGINE=lean_single \
    SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
    bash scripts/lib/run_selfhost_fresh.sh \
      "$SOUC_BIN" self-hosted/compiler/lean.sio --check-merged "$SOURCE" \
      >"$LOG" 2>&1; then
  cat "$LOG" >&2
  fail "source-fresh merged checker failed or timed out with trace env absent"
fi

cat "$LOG"
grep -Fxq 'souc-lean check-merged: begin' "$LOG" \
  || fail "merged import path did not start"
grep -Fxq 'souc-lean check-merged: verdict-ready' "$LOG" \
  || fail "merged checker produced no runtime verdict"
grep -Fxq 'souc-lean check-merged: ok' "$LOG" \
  || fail "merged checker did not accept the EISA witness"

echo "[madaros-read-env-absence] PASS: missing env values use the canonical empty-string path"
