#!/usr/bin/env bash
# Deepen-batch 3: extend coverage of already-shipped verticals with untested public API.
# Multi-module drivers -> lean_single engine.
#   math::dd64      — error-free transforms (two_sum/two_prod), div precision, scalar/neg/abs/cmp
#   special::beta   — lbeta + regularized-incomplete-beta anchors (uniform CDF, arcsine law)
#   signal::filter  — IIR1 step responses, IIR2 notch/bandpass selectivity, FIR moving average
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
run stdlib/math/dd64.sio       tests/stdlib/math/test_dd64_deep_stdlib.sio     DD64_DEEP_STDLIB_OK
run stdlib/special/beta.sio     tests/stdlib/special/test_beta_deep_stdlib.sio  BETA_DEEP_STDLIB_OK
run stdlib/signal/filter.sio    tests/stdlib/signal/test_filter_deep_stdlib.sio FILTER_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN3_GATE_OK"
exit $fail
