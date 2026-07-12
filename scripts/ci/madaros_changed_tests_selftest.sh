#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GATE="$ROOT_DIR/scripts/ci/madaros_changed_tests_gate.sh"

bash -n "$GATE"

selected="$($GATE --select-only \
  tests/run-pass/array_repeat_i8_binding.sio \
  tests/run-pass/generic_struct_return.sio \
  docs/internal/coordination/CI_WATCH_CONTRACT.md)"

grep -Fxq 'tests/run-pass/array_repeat_i8_binding.sio' <<< "$selected"
if grep -Fq 'generic_struct_return.sio' <<< "$selected"; then
  echo 'madaros-changed-tests: selected an unannotated test' >&2
  exit 1
fi
if grep -Fq 'CI_WATCH_CONTRACT.md' <<< "$selected"; then
  echo 'madaros-changed-tests: selected a path outside tests/run-pass' >&2
  exit 1
fi

skip="$($GATE -- docs/internal/coordination/CI_WATCH_CONTRACT.md)"
grep -Fq 'MADAROS_CHANGED_TESTS_SKIP' <<< "$skip"

echo 'MADAROS_CHANGED_TESTS_SELFTEST_PASS'
