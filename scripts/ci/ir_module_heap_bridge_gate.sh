#!/usr/bin/env bash

set -euo pipefail

RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECTED_SHA256="${IR_MODULE_HEAP_BRIDGE_EXPECT_SHA256:-}"
EXPECTED_BANNER='Madaros v0.80.0 -- the Sounio self-hosted compiler'

fail() {
  echo "IR_MODULE_HEAP_BRIDGE_FAIL reason=$1" >&2
  exit 1
}

[[ -n "$RAW_MADAROS" ]] || fail missing_explicit_madaros
[[ "$RAW_MADAROS" == /* ]] || fail madaros_must_be_absolute
[[ -f "$RAW_MADAROS" ]] || fail madaros_not_regular_file
[[ -x "$RAW_MADAROS" ]] || fail madaros_not_executable
magic="$(head -c4 "$RAW_MADAROS" | od -An -tx1 | tr -d '[:space:]')"
[[ "$magic" == 7f454c46 ]] || fail madaros_not_raw_elf

compiler_sha256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"
if [[ -n "$EXPECTED_SHA256" && "$compiler_sha256" != "$EXPECTED_SHA256" ]]; then
  fail madaros_sha256_mismatch
fi

set +e
output="$(
  timeout 300 "$RAW_MADAROS" --ir-heap-bridge-self-test 2>&1
)"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  printf '%s\n' "$output" >&2
  fail "internal_self_test_rc_$rc"
fi
grep -Fxq "$EXPECTED_BANNER" <<<"$output" || {
  printf '%s\n' "$output" >&2
  fail version_banner_missing
}
grep -Fxq 'IR_MODULE_HEAP_BRIDGE_PASS' <<<"$output" || {
  printf '%s\n' "$output" >&2
  fail pass_marker_missing
}
if grep -Fq 'IR_MODULE_HEAP_BRIDGE_FAIL' <<<"$output"; then
  printf '%s\n' "$output" >&2
  fail contradictory_fail_marker
fi
if grep -Fq 'fallback' <<<"$output"; then
  printf '%s\n' "$output" >&2
  fail fallback_observed
fi
echo "IR_MODULE_HEAP_BRIDGE_PASS compiler=$RAW_MADAROS sha256=$compiler_sha256"
