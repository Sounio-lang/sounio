#!/usr/bin/env bash
# Combined gate: encoding::hex, math::dd64, math::combinatorics_perm.
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
run stdlib/encoding/hex.sio               tests/stdlib/encoding/test_hex_stdlib.sio                   HEX_STDLIB_OK
run stdlib/math/dd64.sio                  tests/stdlib/math/test_dd64_stdlib.sio                      DD64_STDLIB_OK
run stdlib/math/combinatorics_perm.sio    tests/stdlib/math/test_combinatorics_perm_stdlib.sio        COMBINATORICS_PERM_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_BATCH2_GATE_OK"
exit $fail
