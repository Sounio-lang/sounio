#!/usr/bin/env bash
# Combined gate: viz::colormap, geo::pure::types, queue::pure::types.
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
run stdlib/viz/colormap.sio        tests/stdlib/viz/test_colormap_stdlib.sio      COLORMAP_STDLIB_OK
run stdlib/geo/pure/types.sio      tests/stdlib/geo/test_geo_types_stdlib.sio     GEO_TYPES_STDLIB_OK
run stdlib/queue/pure/types.sio    tests/stdlib/queue/test_queue_stdlib.sio       QUEUE_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_BATCH5_GATE_OK"
exit $fail
