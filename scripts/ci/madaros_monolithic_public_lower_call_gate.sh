#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_MONO_PUBLIC_LOWER_GATE_KEEP:-0}"
BIN="${SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN:-}"

fail() {
  echo "[madaros-mono-public-lower-gate] FAIL: $*" >&2
  exit 1
}

[[ -n "$BIN" ]] || fail "SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN is required; generic/prebuilt resolution is forbidden"
[[ -x "$BIN" ]] || fail "explicit Madaros ELF is missing or not executable: $BIN"

if [[ -n "${SOUNIO_MADAROS_MONO_PUBLIC_LOWER_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_MONO_PUBLIC_LOWER_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-mono-public-lower-gate.XXXXXX)"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

set +e
env -u SOUNIO_MADAROS_BIN -u MADAROS_RAW_BIN \
  SOUNIO_MADAROS_MONO_PUBLIC_LOWER_BIN="$BIN" \
  SOUNIO_MADAROS_MONO_PUBLIC_LOWER_MATRIX_DIR="$WORK/matrix" \
  bash "$ROOT_DIR/scripts/ci/madaros_monolithic_public_lower_call_matrix.sh" \
  >"$WORK/matrix-driver.log" 2>&1
matrix_rc=$?
set -e

cat "$WORK/matrix-driver.log"
if [[ -f "$WORK/matrix/metadata.tsv" ]]; then
  cat "$WORK/matrix/metadata.tsv"
fi
if [[ -f "$WORK/matrix/receipt.tsv" ]]; then
  cat "$WORK/matrix/receipt.tsv"
fi

if [[ "$matrix_rc" != "0" ]]; then
  fail "public lower call matrix exited rc=$matrix_rc"
fi

COMBINED_LOG="$WORK/matrix/bss_typed_adds.log"
grep -Fxq \
  'monolithic_public_lower_call: bss_globals=3 bss_bytes=144 f64_adds=1 i64_adds=1' \
  "$COMBINED_LOG" || fail "exact BSS/f64/i64 diagnostic missing"
grep -Fxq 'MONOLITHIC_PUBLIC_LOWER_CALL PASS' "$COMBINED_LOG" \
  || fail "exact public lower PASS marker missing"

echo "[madaros-mono-public-lower-gate] PASS: public lower returns across the matrix and preserves combined BSS/add typing"
