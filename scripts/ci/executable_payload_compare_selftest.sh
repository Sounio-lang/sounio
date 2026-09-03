#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARE="$ROOT_DIR/scripts/lib/compare_executable_payloads.sh"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sounio-payload-selftest.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

printf 'same payload\n' >"$WORK_DIR/first"
cp "$WORK_DIR/first" "$WORK_DIR/second"
printf 'different payload\n' >"$WORK_DIR/different"

"$COMPARE" "$WORK_DIR/first" "$WORK_DIR/second"
if "$COMPARE" "$WORK_DIR/first" "$WORK_DIR/different"; then
  echo "error: payload comparator accepted different files" >&2
  exit 1
fi

echo "EXECUTABLE_PAYLOAD_COMPARE_SELFTEST_PASS"
