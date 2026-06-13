#!/usr/bin/env bash
# scripts/gates/erdos90_exact_gate.sh — Erdős [90] exact-integer search gate
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
SOUC="${SOUC:-$ROOT/bin/souc}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

compile() {
    local src="$1" out="$2"
    "$SOUC" "$src" "$out" >/dev/null
}

must_pass_run() {
    local src="$1" expect="$2"
    local out="$WORK/$(basename "$src" .sio).elf"
    compile "$src" "$out"
    chmod +x "$out"
    local log="$WORK/$(basename "$src" .sio).run.log"
    "$out" >"$log" 2>&1
    if ! grep -q "$expect" "$log"; then
        echo "[erdos90-exact] FAIL: $src missing stdout: $expect" >&2
        cat "$log" >&2
        exit 1
    fi
}

must_pass_run tests/run-pass/erdos90_eisenstein_subset_smoke.sio "erdos90_eisenstein_subset_smoke: PASS"
must_pass_run tests/run-pass/erdos90_mixed_pool_smoke.sio "erdos90_mixed_pool_smoke: PASS"

compile stdlib/research/erdos90_repcount_bridge.sio "$WORK/repcount_bridge.elf"
chmod +x "$WORK/repcount_bridge.elf"
"$WORK/repcount_bridge.elf" >"$WORK/repcount_bridge.log" 2>&1
if ! grep -q "r2(15)=0 OK" "$WORK/repcount_bridge.log"; then
    echo "[erdos90-exact] FAIL: repcount bridge" >&2
    cat "$WORK/repcount_bridge.log" >&2
    exit 1
fi

compile examples/erdos/erdos90_repcount_engine.sio "$WORK/repcount_engine.elf"
chmod +x "$WORK/repcount_engine.elf"
"$WORK/repcount_engine.elf" >/dev/null

compile stdlib/research/erdos90_mixed_pool.sio "$WORK/mixed_pool.elf"
chmod +x "$WORK/mixed_pool.elf"
"$WORK/mixed_pool.elf" >"$WORK/mixed_pool.log" 2>&1
if ! grep -q "honest: cross-tag" "$WORK/mixed_pool.log"; then
    echo "[erdos90-exact] FAIL: mixed pool sweep" >&2
    tail -20 "$WORK/mixed_pool.log" >&2
    exit 1
fi

compile stdlib/research/erdos90_eisenstein_subset.sio "$WORK/eisenstein_subset.elf"
chmod +x "$WORK/eisenstein_subset.elf"
"$WORK/eisenstein_subset.elf" >"$WORK/eisenstein_subset.log" 2>&1
if ! grep -q "harb-ok" "$WORK/eisenstein_subset.log"; then
    echo "[erdos90-exact] FAIL: Eisenstein subset sweep" >&2
    tail -30 "$WORK/eisenstein_subset.log" >&2
    exit 1
fi

echo "[erdos90-exact] PASS"