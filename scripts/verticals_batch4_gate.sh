#!/usr/bin/env bash
# Combined gate: collections::hashmap, core::result (default Madaros); math::qd128 (lean_single, imports dd64).
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
run stdlib/collections/hashmap.sio  tests/stdlib/collections/test_hashmap_stdlib.sio  HASHMAP_STDLIB_OK
run stdlib/core/result.sio          tests/stdlib/core/test_result_stdlib.sio          RESULT_STDLIB_OK
run stdlib/math/qd128.sio           tests/stdlib/math/test_qd128_stdlib.sio           QD128_STDLIB_OK   lean_single
[ $fail -eq 0 ] && echo "VERTICALS_BATCH4_GATE_OK"
exit $fail
