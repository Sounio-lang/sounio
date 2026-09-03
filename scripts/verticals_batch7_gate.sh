#!/usr/bin/env bash
# Combined gate: epistemic::covariance, cybernetic::variety (default); audio::pure::types (lean_single).
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
run stdlib/epistemic/covariance.sio    tests/stdlib/epistemic/test_covariance_stdlib.sio    COVARIANCE_STDLIB_OK
run stdlib/cybernetic/variety.sio      tests/stdlib/cybernetic/test_variety_stdlib.sio      VARIETY_STDLIB_OK
run stdlib/audio/pure/types.sio        tests/stdlib/audio/test_audio_stdlib.sio             AUDIO_STDLIB_OK   lean_single
[ $fail -eq 0 ] && echo "VERTICALS_BATCH7_GATE_OK"
exit $fail
