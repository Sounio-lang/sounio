#!/usr/bin/env bash
# A named diagnostic must be counted without changing the failed rung/verdict.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${1:-$(mktemp -d "${TMPDIR:-/tmp}/fixed-point-diagnostic.XXXXXX")}"
mkdir -p "$WORK"
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror \
  "$ROOT/scripts/ci/fixtures/named_error_compiler.cpp" -o "$WORK/compiler"
set +e
SOUNIO_MADAROS_CACHE='' MADAROS_BIN="$WORK/compiler" \
  SOUNIO_MADAROS_FP_DIR="$WORK/rung" bash "$ROOT/scripts/ci/madaros_fixed_point_gate.sh" > "$WORK/gate.log" 2>&1
status=$?
set -e
test "$status" = 1
grep -Fq 'rc=1 errors=1' "$WORK/gate.log"
grep -Fq "stopped at rung 'none'" "$WORK/gate.log"
grep -Fq 'error[E-USE-MOD-CAP]' "$WORK/gate.log"
echo 'PASS: named checker diagnostic is counted; failing check remains a failing rung'
