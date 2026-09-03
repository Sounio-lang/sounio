#!/usr/bin/env bash
# Fail-honest classifier: imported epistemic::propagate::monte_carlo with a
# named fn(f64)->f64 compiles but produces an invalid Monte Carlo result under
# stock Madaros. A green result must replace this gate with a promotion gate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
PROBE="tests/known_failures/madaros_propagate_monte_carlo_fnptr_probe.sio"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "== madaros_propagate_monte_carlo_fnptr_failclosed_gate =="

unset SOUNIO_SOUC_ENGINE || true
set +e
"$SOUC" run "$PROBE" >"$OUT/madaros.log" 2>&1
MRC=$?
set -e
if [[ "$MRC" -eq 0 ]]; then
    echo "FAIL: Madaros unexpectedly passed; replace this with a green promotion gate"
    cat "$OUT/madaros.log" || true
    exit 1
fi
grep -Fq 'MONTE_CARLO_FNPTR FAIL' "$OUT/madaros.log" || {
    echo "FAIL: missing deterministic wrong-result sentinel"
    cat "$OUT/madaros.log" || true
    exit 1
}
if grep -Fq 'Segmentation fault' "$OUT/madaros.log"; then
    echo "FAIL: segfault is not the classified wrong-result mode"
    cat "$OUT/madaros.log" || true
    exit 1
fi

export SOUNIO_SOUC_ENGINE=lean_single
set +e
"$SOUC" run "$PROBE" >"$OUT/lean.log" 2>&1
LRC=$?
set -e
unset SOUNIO_SOUC_ENGINE || true
[[ "$LRC" -eq 0 ]] || {
    echo "FAIL: lean_single oracle rc=$LRC"
    cat "$OUT/lean.log" || true
    exit 1
}
grep -Fq 'MONTE_CARLO_FNPTR PASS' "$OUT/lean.log" || {
    echo "FAIL: lean_single missing MONTE_CARLO_FNPTR PASS"
    cat "$OUT/lean.log" || true
    exit 1
}

echo "MADAROS_PROPAGATE_MONTE_CARLO_FNPTR_FAILCLOSED_GATE_OK"
