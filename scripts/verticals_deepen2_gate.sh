#!/usr/bin/env bash
# Deepen-batch 2: extend coverage of already-shipped verticals with untested public API.
# Large struct-return / multi-module drivers -> lean_single engine.
#   linalg::matnm — LU + QR reconstruction, QR orthogonality, add/sub/scale
#   units::lib     — quantity_sub (quadrature), reverse conversions, derived-dimension algebra
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
run stdlib/linalg/matnm.sio  tests/stdlib/linalg/test_matnm_deep_stdlib.sio  MATNM_DEEP_STDLIB_OK
run stdlib/units/lib.sio      tests/stdlib/units/test_units_deep_stdlib.sio  UNITS_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN2_GATE_OK"
exit $fail
