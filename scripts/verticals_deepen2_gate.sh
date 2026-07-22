#!/usr/bin/env bash
# Deepen-batch 2: extend coverage of already-shipped verticals with untested public API.
# Large struct-return / multi-module drivers under default Madaros.
#   linalg::matnm — LU + QR reconstruction, QR orthogonality, add/sub/scale
#   units::lib     — quantity_sub (quadrature), reverse conversions, derived-dimension algebra
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
engine_info="$($SOUC info 2>&1)"
if ! grep -qF 'Madaros v' <<<"$engine_info"; then
  echo "FAIL: verticals deepen2 gate requires default Madaros" >&2
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
run stdlib/linalg/matnm.sio  tests/stdlib/linalg/test_matnm_deep_stdlib.sio  MATNM_DEEP_STDLIB_OK
run stdlib/units/lib.sio      tests/stdlib/units/test_units_deep_stdlib.sio  UNITS_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN2_GATE_OK"
exit $fail
