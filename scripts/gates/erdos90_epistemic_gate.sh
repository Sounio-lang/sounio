#!/usr/bin/env bash
# scripts/gates/erdos90_epistemic_gate.sh — Erdős [90] epistemic capability gate
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

must_fail() {
    local src="$1" pattern="$2"
    local out="$WORK/$(basename "$src" .sio).elf"
    local log="$WORK/$(basename "$src" .sio).log"
    if "$SOUC" "$src" "$out" >"$log" 2>&1; then
        echo "[erdos90-epistemic] FAIL: expected compile failure for $src" >&2
        exit 1
    fi
    if ! grep -q "$pattern" "$log"; then
        echo "[erdos90-epistemic] FAIL: $src missing pattern: $pattern" >&2
        tail -20 "$log" >&2
        exit 1
    fi
}

must_pass_run() {
    local src="$1" expect="$2"
    local out="$WORK/$(basename "$src" .sio).elf"
    compile "$src" "$out"
    chmod +x "$out"
    local log="$WORK/$(basename "$src" .sio).run.log"
    "$out" >"$log" 2>&1
    if ! grep -q "$expect" "$log"; then
        echo "[erdos90-epistemic] FAIL: $src missing stdout: $expect" >&2
        cat "$log" >&2
        exit 1
    fi
}

must_pass_run tests/run-pass/erdos90_sigma_d_reconcile.sio "erdos90_sigma_d_reconcile: PASS"
must_pass_run tests/run-pass/erdos90_epistemic_gate_honest.sio "ERDOS90_EPISTEMIC_GATE_PASS"
must_pass_run tests/run-pass/erdos90_epistemic_exact_u7_u9.sio "erdos90_epistemic_exact_u7_u9: PASS"
must_pass_run tests/run-pass/erdos90_epistemic_tau_table.sio "erdos90_epistemic_tau_table: PASS"
must_fail tests/compile-fail/erdos90_epistemic_gate_overclaim.sio "EpistemicComplete violation"
must_fail tests/compile-fail/erdos90_unwrap_unsafe.sio "E170"
compile stdlib/research/erdos90_epistemic.sio "$WORK/erdos90_epistemic.elf"
chmod +x "$WORK/erdos90_epistemic.elf"
"$WORK/erdos90_epistemic.elf" >/dev/null
compile stdlib/research/erdos90_search_epistemic.sio "$WORK/erdos90_search_epistemic.elf"
chmod +x "$WORK/erdos90_search_epistemic.elf"
"$WORK/erdos90_search_epistemic.elf" >/dev/null

echo "[erdos90-epistemic] PASS"