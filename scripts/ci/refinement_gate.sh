#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc
REQUIRE_ZERO_SKIPS="${SOUNIO_REFINEMENT_GATE_REQUIRE_ZERO_SKIPS:-0}"

echo "[refinement-gate] running refinement-focused SIO suite"
output="$(bash "$ROOT_DIR/scripts/run_sio_test_suite.sh" refinement --verbose)"
printf '%s\n' "$output"

pass_count="$(printf '%s\n' "$output" | awk '/^  Pass:/ {print $2}' | tail -n1)"
fail_count="$(printf '%s\n' "$output" | awk '/^  Fail:/ {print $2}' | tail -n1)"
skip_count="$(printf '%s\n' "$output" | awk '/^  Skip:/ {print $2}' | tail -n1)"

if [[ -z "$pass_count" || -z "$fail_count" || -z "$skip_count" ]]; then
  echo "[refinement-gate] unable to parse suite summary" >&2
  exit 2
fi

if [[ "$fail_count" != "0" ]]; then
  echo "[refinement-gate] FAIL: refinement suite reported failures=$fail_count" >&2
  exit 1
fi

if [[ "$skip_count" != "0" ]]; then
  if [[ "$REQUIRE_ZERO_SKIPS" == "1" ]]; then
    echo "[refinement-gate] FAIL: refinement suite reported skips=$skip_count" >&2
    exit 1
  fi
  echo "[refinement-gate] WARN: refinement suite reported skips=$skip_count (allowed)" >&2
fi

echo "[refinement-gate] PASS: pass=$pass_count fail=$fail_count skip=$skip_count"
echo "REFINEMENT_STRICT_GATE_PASS"
