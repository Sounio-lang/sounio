#!/usr/bin/env bash
# Promotion gate: imported epistemic::propagate::monte_carlo with a named
# fn(f64)->f64 matches lean_single on mean/variance bands under default Madaros.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
PROBE="tests/epistemic_trust/madaros_propagate_monte_carlo_fnptr_probe.sio"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "== madaros_propagate_monte_carlo_fnptr_gate =="

unset SOUNIO_SOUC_ENGINE || true
set +e
"$SOUC" run "$PROBE" >"$OUT/madaros.log" 2>&1
MRC=$?
set -e
[[ "$MRC" -eq 0 ]] || {
    echo "FAIL: Madaros rc=$MRC"
    cat "$OUT/madaros.log" || true
    exit 1
}
grep -Fq 'MONTE_CARLO_FNPTR PASS' "$OUT/madaros.log" || {
    echo "FAIL: Madaros missing MONTE_CARLO_FNPTR PASS"
    cat "$OUT/madaros.log" || true
    exit 1
}

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

echo "MADAROS_PROPAGATE_MONTE_CARLO_FNPTR_GATE_OK"
