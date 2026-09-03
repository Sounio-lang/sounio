#!/usr/bin/env bash
# Deepen-batch 1: extend coverage of already-shipped science verticals with the public API
# left untested by their original run-proofs. All multi-module -> lean_single engine.
#   prob::distributions  — gamma/exponential/uniform/poisson/dirichlet + non-standard normal
#   special::erf         — erfinv + erfc tail
#   special::gamma       — half-integer + large-argument reference points
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
run stdlib/prob/distributions.sio  tests/stdlib/prob/test_prob_deep_stdlib.sio      PROB_DEEP_STDLIB_OK
run stdlib/special/erf.sio          tests/stdlib/special/test_erf_deep_stdlib.sio    ERF_DEEP_STDLIB_OK
run stdlib/special/gamma.sio        tests/stdlib/special/test_gamma_deep_stdlib.sio  GAMMA_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN1_GATE_OK"
exit $fail
