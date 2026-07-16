#!/usr/bin/env bash
# Combined gate: cybernetic::coupling, signal::fractal, cybernetic::observer (all default Madaros).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
run() { # module driver sentinel engine
  echo "== $2 =="
  $SOUC check "$1" >/dev/null 2>&1 || { echo "FAIL check $1"; fail=1; }
  if [ "${4:-}" = "lean_single" ]; then
    if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; then
      chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "$3" || { echo "FAIL run $2"; fail=1; }
    else echo "FAIL compile $2"; fail=1; fi
  else
    if $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; then
      "$OUT/x.elf" | grep -q "$3" || { echo "FAIL run $2"; fail=1; }
    else echo "FAIL compile $2"; fail=1; fi
  fi
}
run stdlib/cybernetic/coupling.sio  tests/stdlib/cybernetic/test_coupling_stdlib.sio  COUPLING_STDLIB_OK
run stdlib/signal/fractal.sio       tests/stdlib/signal/test_fractal_stdlib.sio       FRACTAL_STDLIB_OK
run stdlib/cybernetic/observer.sio  tests/stdlib/cybernetic/test_observer_stdlib.sio  OBSERVER_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_BATCH8_GATE_OK"
exit $fail
