#!/usr/bin/env bash
# Combined gate: autodiff::epistemic_dual, collections::vec, collections::stack.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
run() {
  echo "== $2 =="
  $SOUC check "$1" >/dev/null 2>&1 || { echo "FAIL check $1"; fail=1; }
  if $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; then
    "$OUT/x.elf" | grep -q "$3" || { echo "FAIL run $2"; fail=1; }
  else echo "FAIL compile $2"; fail=1; fi
}
run stdlib/autodiff/epistemic_dual.sio  tests/stdlib/autodiff/test_edual_stdlib.sio        EDUAL_STDLIB_OK
run stdlib/collections/vec.sio          tests/stdlib/collections/test_vec_stdlib.sio        VEC_STDLIB_OK
run stdlib/collections/stack.sio        tests/stdlib/collections/test_stack_stdlib.sio      STACK_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_BATCH3_GATE_OK"
exit $fail
