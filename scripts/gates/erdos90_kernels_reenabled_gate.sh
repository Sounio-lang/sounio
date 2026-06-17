#!/usr/bin/env bash
# scripts/gates/erdos90_kernels_reenabled_gate.sh
#
# Gate verifying that the Erdős #90 CPU search kernels compile and run with the
# lean_single self-hosted compiler after the module-syntax cleanup.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${SOUNIO_SOUC_BIN:-$ROOT/bin/souc-lean-single-x86_64}"
WORK="${WORK:-$(mktemp -d)}"
trap 'rm -rf "$WORK"' EXIT INT TERM

[[ -x "$SOUC" ]] || { echo "error: SOUC is not executable: $SOUC" >&2; exit 2; }

echo "[erdos90-kernels-reenabled] compiler=$SOUC"

compile_run() {
    local src="$1"
    local elf="$2"
    local label="$3"
    echo "[erdos90-kernels-reenabled] build $label"
    "$SOUC" "$src" "$elf" >/dev/null 2>&1
    chmod +x "$elf"
    "$elf" || true
}

compile_run "$ROOT/stdlib/research/erdos90_search.sio" "$WORK/erdos90_search.elf" "search"
# Use a small rr band for the gate so it finishes quickly; full sweeps are run separately.
mkdir -p "$WORK/src"
cp "$ROOT/stdlib/research/erdos90_optimize.sio" "$WORK/src/erdos90_optimize_gate.sio"
sed -i 's/run_sweep(50, 10000)/run_sweep(50, 1000)/' "$WORK/src/erdos90_optimize_gate.sio"
compile_run "$WORK/src/erdos90_optimize_gate.sio" "$WORK/erdos90_optimize.elf" "optimize"
# subset compiles but its default 300k-swap runs are too slow for a light gate;
# we only verify it builds.
"$SOUC" "$ROOT/stdlib/research/erdos90_subset.sio" "$WORK/erdos90_subset.elf" >/dev/null 2>&1
chmod +x "$WORK/erdos90_subset.elf"
echo "[erdos90-kernels-reenabled] build subset (run skipped in gate)"
compile_run "$ROOT/stdlib/research/erdos90_kaxi_hc_smoke.sio" "$WORK/erdos90_kaxi_hc_smoke.elf" "kaxi_hc_smoke"

# Spot checks on known-good outputs.
"$WORK/erdos90_search.elf" > "$WORK/search.log" 2>&1 || true
rg -q 'grid configs beating triangular harb\(n\): 5' "$WORK/search.log"

"$WORK/erdos90_optimize.elf" > "$WORK/optimize.log" 2>&1 || true
rg -q 'BEAT' "$WORK/optimize.log"

"$WORK/erdos90_kaxi_hc_smoke.elf" > "$WORK/kaxi.log" 2>&1 || true
rg -q 'KAXI_SMOKE_DONE' "$WORK/kaxi.log"

echo "[erdos90-kernels-reenabled] PASS"
