#!/usr/bin/env bash
# Madaros promotion gate — generic epistemic::propagate::monte_carlo(x, f, n) with a
# named fn(f64)->f64 under default Madaros (imported multi-module native path).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
PROBE="tests/known_failures/madaros_propagate_monte_carlo_fnptr_probe.sio"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "== madaros_propagate_monte_carlo_fnptr_gate =="

engine_line="$("$SOUC" --version 2>&1 | head -1 || true)"
if echo "$engine_line" | grep -qi lean_single; then
    echo "FAIL: gate must run under default Madaros, not lean_single ($engine_line)"
    exit 1
fi
echo "engine: $engine_line"

unset SOUNIO_SOUC_ENGINE || true
set +e
"$SOUC" run "$PROBE" >"$OUT/madaros.log" 2>&1
MRC=$?
set -e
if [[ "$MRC" -ne 0 ]]; then
    echo "FAIL: Madaros rc=$MRC"
    cat "$OUT/madaros.log" || true
    exit 1
fi
grep -Fq 'MONTE_CARLO_FNPTR PASS' "$OUT/madaros.log" || {
    echo "FAIL: missing MONTE_CARLO_FNPTR PASS"
    cat "$OUT/madaros.log" || true
    exit 1
}
if grep -Fq 'MONTE_CARLO_FNPTR FAIL' "$OUT/madaros.log"; then
    echo "FAIL: wrong-result sentinel still present"
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

echo "MADAROS_PROPAGATE_MONTE_CARLO_FNPTR_GATE_OK"
