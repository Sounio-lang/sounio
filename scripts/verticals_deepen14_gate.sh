#!/usr/bin/env bash
# Deepen-batch 14: import witnesses for surfaces whose older tests are stubs
# or inlined copies.
# Multi-module drivers -> lean_single engine.
#   interpolation::lib          — lerp, inverse_lerp, remap, smoothstep, bilerp
#   integrate::epistemic_ode    — decay value lane stays separate from GUM width
#   optimize::nelder_mead       — nm_quadratic minimum of (x-3)^2+(y-2)^2
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
run stdlib/interpolation/lib.sio         tests/stdlib/interpolation/test_lerp_deep_stdlib.sio          LERP_DEEP_STDLIB_OK
run stdlib/integrate/epistemic_ode.sio   tests/stdlib/integrate/test_epistemic_decay_deep_stdlib.sio   DECAY_DEEP_STDLIB_OK
run stdlib/optimize/nelder_mead.sio      tests/stdlib/optimize/test_nm_quadratic_deep_stdlib.sio       NM_QUADRATIC_DEEP_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_DEEPEN14_GATE_OK"
exit $fail
