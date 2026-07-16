#!/usr/bin/env bash
# Deepen-batch 8: extend coverage of already-shipped verticals with untested public API.
# Multi-module drivers -> lean_single engine.
#   core::result             — IntResult/FloatResult monad (ok/err, unwrap, unwrap_or, err)
#   algebra::jordan          — exceptional Jordan algebra J3(O): trace, det, Jordan product, square
#   algebra::cayley_dickson  — CD tower: commutativity/associativity by level (168 = |PSL(2,7)|)
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
run() { # module driver sentinel
  echo "== $2 =="
  $SOUC check "$1" >/dev/null 2>&1 || { echo "FAIL check $1"; fail=1; }
  if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; then
    chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "$3" || { echo "FAIL run $2"; fail=1; }
  else echo "FAIL compile $2"; fail=1; fi
}
run stdlib/core/result.sio             tests/stdlib/core/test_result_deep_stdlib.sio            RESULT_DEEP_STDLIB_OK
run stdlib/algebra/jordan.sio          tests/stdlib/algebra/test_jordan_deep_stdlib.sio         JORDAN_DEEP_STDLIB_OK
run stdlib/algebra/cayley_dickson.sio  tests/stdlib/algebra/test_cayley_dickson_deep_stdlib.sio CAYLEY_DICKSON_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN8_GATE_OK"
exit $fail
