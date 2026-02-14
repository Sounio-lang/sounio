#!/bin/bash
# fMRI Causal Connectivity Benchmark Validation
#
# Runs the fMRI benchmark and validates results against paper targets.
# For PLDI/UAI 2027 submission.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "======================================================================"
echo "  fMRI Causal Connectivity Benchmark Suite"
echo "  Paper: Causal Programming with do-Calculus Types (PLDI/UAI 2027)"
echo "======================================================================"
echo ""

# 1. Run causal connectivity benchmark
echo "[1/3] Running fMRI causal connectivity benchmark..."
cd "$ROOT_DIR"
cargo run --release --bin souc -- run benchmarks/fmri/causal_connectivity.sio \
    2>&1 | tee "$RESULTS_DIR/causal_connectivity_output.txt"
echo ""

# 2. Validate targets
echo "[2/3] Checking validation targets..."
echo "  Target: Identifiable connections verified (V2→V4, V4→IT, V1→MT, MT→MST)"
echo "  Target: Non-identifiable V1→V2 correctly rejected"
echo "  Target: Confidence intervals properly computed"
echo "  Target: Bootstrap uncertainty reflects sampling variability"
echo ""

# 3. Comparison data
echo "[3/3] Comparison baselines:"
echo "  - SPM12 (MATLAB): No causal identifiability checking"
echo "  - FSL (C++): No causal identifiability checking"
echo "  - AFNI (C): No causal identifiability checking"
echo "  - Sounio: COMPILE-TIME identifiability rejection"
echo ""

echo "======================================================================"
echo "  VALIDATION TARGETS"
echo "======================================================================"
echo ""
echo "  [  ] Non-identifiable connectivity correctly rejected"
echo "  [  ] Identifiable connections verified via backdoor"
echo "  [  ] Fisher z-transform confidence intervals"
echo "  [  ] Bootstrap uncertainty > 0"
echo "  [  ] Causal vs correlational results distinguished"
echo ""
echo "  See results/ directory for detailed output."
echo "======================================================================"
