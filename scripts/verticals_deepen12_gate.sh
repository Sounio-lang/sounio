#!/usr/bin/env bash
# Deepen-batch 12: extend coverage of already-shipped deterministic utility APIs.
# Multi-module drivers -> lean_single engine.
#   path::lib          - join/parent/filename/stem/extension/from_bytes edge cases
#   iter::range        - stepped ranges, reset, empty ranges, 64-item collect cap
#   random::pcg64_core - bit helpers, 128-bit carry/mul, deterministic PCG stepping
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
run() { # module driver sentinel [softcheck]
  echo "== $2 =="
  if [ "${4:-}" = "softcheck" ]; then
    $SOUC check "$1" >/dev/null 2>&1 || echo "NOTE: standalone check quirk on $1 (driver proves the API)"
  else
    $SOUC check "$1" >/dev/null 2>&1 || { echo "FAIL check $1"; fail=1; }
  fi
  if SOUNIO_SOUC_ENGINE=lean_single $SOUC compile "$2" -o "$OUT/x.elf" >/dev/null 2>&1; then
    chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "$3" || { echo "FAIL run $2"; fail=1; }
  else echo "FAIL compile $2"; fail=1; fi
}
run stdlib/path/lib.sio             tests/stdlib/path/test_path_deep_stdlib.sio        PATH_DEEP_STDLIB_OK
run stdlib/iter/range.sio           tests/stdlib/iter/test_range_deep_stdlib.sio       RANGE_DEEP_STDLIB_OK
run stdlib/random/pcg64_core.sio    tests/stdlib/random/test_pcg64_core_deep_stdlib.sio PCG64_CORE_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN12_GATE_OK"
exit $fail
