#!/usr/bin/env bash
# Deepen-batch 6: extend coverage of already-shipped verticals with untested public API.
# Multi-module drivers -> lean_single engine.
#   epistemic::covariance — correlation-aware GUM propagation u_y^2 = J Sigma J^T, det/trace/scale/PD
#   algebra::octonion     — normed-division-algebra identities (unit norms, e_i^2=-1, |ab|=|a||b|, anticommute)
#   crypto::sha256        — hashes vs published FIPS 180-4 / NIST test vectors ("abc", empty)
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
run stdlib/epistemic/covariance.sio  tests/stdlib/epistemic/test_covariance_deep_stdlib.sio  COVARIANCE_DEEP_STDLIB_OK
run stdlib/algebra/octonion.sio       tests/stdlib/algebra/test_octonion_deep_stdlib.sio     OCTONION_DEEP_STDLIB_OK
run stdlib/crypto/sha256.sio          tests/stdlib/crypto/test_sha256_deep_stdlib.sio        SHA256_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN6_GATE_OK"
exit $fail
