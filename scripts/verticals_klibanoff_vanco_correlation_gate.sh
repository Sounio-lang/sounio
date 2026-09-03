#!/usr/bin/env bash
# Verticals: epistemic::klibanoff (smooth-ambiguity CE sandwich), clinical::vancomycin_pbpk (PK),
# epistemic::correlation (GUM covariance tracking). Multi-module drivers -> lean_single engine.
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
run stdlib/epistemic/klibanoff.sio        tests/stdlib/epistemic/test_klibanoff_stdlib.sio       KLIBANOFF_STDLIB_OK
run stdlib/clinical/vancomycin_pbpk.sio    tests/stdlib/clinical/test_vancomycin_pbpk_stdlib.sio VANCOMYCIN_PBPK_STDLIB_OK
run stdlib/epistemic/correlation.sio       tests/stdlib/epistemic/test_correlation_stdlib.sio    CORRELATION_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_KLIBANOFF_VANCO_CORRELATION_GATE_OK"
exit $fail
