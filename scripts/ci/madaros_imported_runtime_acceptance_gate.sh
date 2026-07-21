#!/usr/bin/env bash
# #901: imported nominal layout identity must survive lowering and fail closed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_BIN:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
KEEP_WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_KEEP:-0}"
WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_DIR:-$(mktemp -d /tmp/sounio-madaros-imported-runtime-acceptance.XXXXXX)}"
PASS_SOURCE="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_nested_field_chain_main.sio"
MISS_SOURCE="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_known_layout_miss_main.sio"

fail() {
  echo "[madaros-imported-runtime-acceptance] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros wrapper is missing or not executable: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN must name an explicit current-source Madaros ELF"
[[ -x "$RAW_MADAROS" ]] || fail "explicit current-source Madaros is missing or not executable: $RAW_MADAROS"
[[ -f "$PASS_SOURCE" ]] || fail "positive witness is missing: $PASS_SOURCE"
[[ -f "$MISS_SOURCE" ]] || fail "negative witness is missing: $MISS_SOURCE"

mkdir -p "$WORK/pass" "$WORK/miss"
if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

set +e
(
  cd "$WORK/pass"
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" run "$PASS_SOURCE"
) >"$WORK/pass/run.log" 2>&1
pass_rc=$?
set -e
[[ "$pass_rc" -eq 0 ]] || {
  cat "$WORK/pass/run.log" >&2
  fail "typed nested-field witness exited rc=$pass_rc"
}
grep -Fxq 'ISSUE_901_NESTED_FIELD_CHAIN_OK' "$WORK/pass/run.log" || {
  cat "$WORK/pass/run.log" >&2
  fail "typed nested-field witness lost its exact marker"
}

set +e
(
  cd "$WORK/miss"
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" run "$MISS_SOURCE"
) >"$WORK/miss/run.log" 2>&1
miss_rc=$?
set -e
[[ "$miss_rc" -ne 0 ]] || {
  cat "$WORK/miss/run.log" >&2
  fail "known-layout miss unexpectedly compiled or ran"
}
[[ ! -e "$WORK/miss/a.out" ]] || {
  cat "$WORK/miss/run.log" >&2
  fail "known-layout miss emitted a native artifact"
}
if grep -Fxq 'ISSUE_901_NESTED_FIELD_CHAIN_OK' "$WORK/miss/run.log"; then
  cat "$WORK/miss/run.log" >&2
  fail "known-layout miss reached the positive runtime marker"
fi

echo "[madaros-imported-runtime-acceptance] PASS: nested fields retain nominal layouts and known-layout misses fail closed"
