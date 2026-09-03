#!/usr/bin/env bash
# Deepen-batch 7: extend coverage of already-shipped verticals with untested public API.
# Multi-module drivers -> lean_single engine.
#   algebra::sedenion — the zero-divisor hallmark z*w=0 (|z|,|w|>0), norm/dot/add/scale
#   algebra::clifford — geometric algebra: dim 2^n, signature squares, reversion, norm
#   cmp::lib          — comparison / min-max / clamp / sign / approx_eq utilities
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
run stdlib/algebra/sedenion.sio  tests/stdlib/algebra/test_sedenion_deep_stdlib.sio  SEDENION_DEEP_STDLIB_OK
run stdlib/algebra/clifford.sio   tests/stdlib/algebra/test_clifford_deep_stdlib.sio  CLIFFORD_DEEP_STDLIB_OK
run stdlib/cmp/lib.sio            tests/stdlib/cmp/test_cmp_stdlib.sio                CMP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN7_GATE_OK"
exit $fail
