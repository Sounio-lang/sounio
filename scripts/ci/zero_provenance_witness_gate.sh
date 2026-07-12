#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="$ROOT_DIR/tests/known_failures/zero_provenance_native_v2_probe.sio"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zero-provenance.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[zero-provenance] FAIL: $*" >&2
  exit 1
}

[[ -f "$SOURCE" ]] || fail "missing witness: $SOURCE"

"$ROOT_DIR/bin/souc" check "$SOURCE" >"$TMP_DIR/check.log" 2>&1 || {
  cat "$TMP_DIR/check.log" >&2
  fail "Madaros check failed"
}

output="$(SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$SOURCE" 2>&1)" || {
  printf '%s\n' "$output" >&2
  fail "lean_single execution failed"
}

printf '%s\n' "$output" | grep -Fq 'ZERO_PROVENANCE PASS' || {
  printf '%s\n' "$output" >&2
  fail "missing pass marker"
}

echo '[zero-provenance] PASS: five surface-zero paths remain distinguishable'
