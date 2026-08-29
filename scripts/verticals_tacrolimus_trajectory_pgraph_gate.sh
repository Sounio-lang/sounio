#!/usr/bin/env bash
# Verticals: clinical::tacrolimus_oral_safety (PK), dialogue::trajectory (conversation store),
# epistemic::perturbation_graph (octonion variance DAG). Multi-module drivers -> lean_single engine.
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
run stdlib/clinical/tacrolimus_oral_safety.sio  tests/stdlib/clinical/test_tacrolimus_stdlib.sio            TACROLIMUS_STDLIB_OK
run stdlib/dialogue/trajectory.sio               tests/stdlib/dialogue/test_trajectory_stdlib.sio           TRAJECTORY_STDLIB_OK
run stdlib/epistemic/perturbation_graph.sio      tests/stdlib/epistemic/test_perturbation_graph_stdlib.sio  PERTURBATION_GRAPH_STDLIB_OK
[ $fail -eq 0 ] && echo "VERTICALS_TACROLIMUS_TRAJECTORY_PGRAPH_GATE_OK"
exit $fail
