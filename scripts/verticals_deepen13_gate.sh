#!/usr/bin/env bash
# Deepen-batch 13: import witnesses for already-shipped numerical APIs whose
# older tests copied the algorithm instead of calling stdlib.
# Multi-module drivers -> lean_single engine.
#   polynomial::lib         — Horner eval, derivative, quadratic roots
#   roots::lib id dispatch  — bisection/newton/secant on f_id=0, convergence width
#   interpolation::spline   — epistemic value lane stays separate from uncertainty
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
run stdlib/polynomial/lib.sio              tests/stdlib/polynomial/test_polynomial_deep_stdlib.sio POLY_DEEP_STDLIB_OK
run stdlib/roots/lib.sio                   tests/stdlib/roots/test_roots_id_deep_stdlib.sio       ROOTS_ID_DEEP_STDLIB_OK
run stdlib/interpolation/spline.sio        tests/stdlib/interpolation/test_espline_deep_stdlib.sio ESPLINE_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN13_GATE_OK"
exit $fail
