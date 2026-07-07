#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
unset SOUNIO_SOUC_ENGINE
unset SOUNIO_SOUC_BIN
CASE_TIMEOUT="${EISA_MADAROS_IMPORT_TIMEOUT:-300}"
SOUC_BIN="${EISA_MADAROS_IMPORT_SOUC_BIN:-./bin/souc}"

fail() {
  echo "[eisa-madaros-import] FAIL: $*" >&2
  exit 1
}

run_case() {
  local name="$1"
  shift
  echo "[eisa-madaros-import] case=$name"
  timeout "$CASE_TIMEOUT" "$@"
}

git ls-files --error-unmatch stdlib/eisa/core.sio >/dev/null 2>&1 \
  || fail "stdlib/eisa/core.sio is not tracked"
[[ -s stdlib/eisa/core.sio ]] || fail "stdlib/eisa/core.sio is missing or empty"
grep -qx '!/stdlib/eisa/core[.]sio' .gitignore \
  || fail ".gitignore must unignore stdlib/eisa/core.sio"
[[ -x "$SOUC_BIN" ]] || fail "$SOUC_BIN is not executable; build or restore the compiler wrapper first"

info_out="$("$SOUC_BIN" info)"
printf '%s\n' "$info_out"
grep -q 'wrapper:.*bin/madaros' <<<"$info_out" \
  || fail "bin/souc default engine is not Madaros"
grep -Fxq 'identity:     Madares v0.80.0 -- the Sounio self-hosted compiler' <<<"$info_out" \
  || fail "bin/souc default identity changed from the M3a validated Madaros build"

run_case check_core "$SOUC_BIN" check tests/stdlib/eisa/test_eisa_core.sio
run_case run_core "$SOUC_BIN" run tests/stdlib/eisa/test_eisa_core.sio
run_case check_evm_v2 "$SOUC_BIN" check tests/stdlib/eisa/test_eisa_evm_v2.sio
run_case check_v1e_showcase "$SOUC_BIN" check tests/stdlib/eisa/test_eisa_v1e_showcase.sio

echo "[eisa-madaros-import] PASS: EISA core import is tracked and Madaros default-lane check boundary is live"
