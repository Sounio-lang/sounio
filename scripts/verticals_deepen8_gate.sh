#!/usr/bin/env bash
# Deepen-batch 8: extend coverage of already-shipped verticals with untested public API.
# Multi-module drivers under default Madaros.
#   core::result             — IntResult/FloatResult monad (ok/err, unwrap, unwrap_or, err)
#   algebra::jordan          — exceptional Jordan algebra J3(O): trace, det, Jordan product, square
#   algebra::cayley_dickson  — CD tower: commutativity/associativity by level (168 = |PSL(2,7)|)
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
engine_info="$($SOUC info 2>&1)"
if ! grep -qF 'Madaros v' <<<"$engine_info"; then
  echo "FAIL: verticals deepen8 gate requires default Madaros" >&2
  printf '%s\n' "$engine_info" >&2
  exit 1
fi
run() { # module driver sentinel
  echo "== $2 =="
  $SOUC check "$1" >/dev/null 2>&1 || { echo "FAIL check $1"; fail=1; }
  if $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; then
    chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "$3" || { echo "FAIL run $2"; fail=1; }
  else echo "FAIL compile $2"; fail=1; fi
}
run stdlib/core/result.sio             tests/stdlib/core/test_result_deep_stdlib.sio            RESULT_DEEP_STDLIB_OK
run stdlib/algebra/jordan.sio          tests/stdlib/algebra/test_jordan_deep_stdlib.sio         JORDAN_DEEP_STDLIB_OK
run stdlib/algebra/cayley_dickson.sio  tests/stdlib/algebra/test_cayley_dickson_deep_stdlib.sio CAYLEY_DICKSON_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN8_GATE_OK"
exit $fail
