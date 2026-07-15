#!/usr/bin/env bash
# Combined gate for the grouped run-proof batch: special::beta, math::hyperbolic, math::rational.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
run() { # module_check_path  driver_path  sentinel
  echo "== $2 =="
  $SOUC check "$1" >/dev/null 2>&1 || { echo "FAIL check $1"; fail=1; }
  if $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; then
    "$OUT/x.elf" | grep -q "$3" || { echo "FAIL run $2"; fail=1; }
  else echo "FAIL compile $2"; fail=1; fi
}
run stdlib/special/beta.sio       tests/stdlib/special/test_beta_stdlib.sio        BETA_STDLIB_OK
run stdlib/math/hyperbolic.sio    tests/stdlib/math/test_hyperbolic_stdlib.sio     HYPERBOLIC_STDLIB_OK
run stdlib/math/rational.sio      tests/stdlib/math/test_rational_stdlib.sio       RATIONAL_STDLIB_OK
[ $fail -eq 0 ] && echo "MATH_VERTICALS_BATCH_GATE_OK"
exit $fail
