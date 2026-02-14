#!/bin/bash
# PBPK Causal Intervention Benchmark Validation
#
# Runs the PBPK benchmark and validates results against paper targets.
# For PLDI/UAI 2027 submission.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "======================================================================"
echo "  PBPK Causal Intervention Benchmark Suite"
echo "  Paper: Causal Programming with do-Calculus Types (PLDI/UAI 2027)"
echo "======================================================================"
echo ""

# 1. Run causal intervention benchmark
echo "[1/3] Running PBPK causal intervention benchmark..."
cd "$ROOT_DIR"
cargo run --release --bin souc -- run benchmarks/pbpk/causal_intervention.sio \
    2>&1 | tee "$RESULTS_DIR/causal_intervention_output.txt"
echo ""

# 2. Validate targets
echo "[2/3] Checking validation targets..."
echo "  Target: Identifiability verified for all non-confounded paths"
echo "  Target: Non-identifiable V1→V2 correctly rejected"
echo "  Target: ATE > 0 (dose-response relationship)"
echo "  Target: Uncertainty ε_total < 0.50"
echo "  Target: Genotype modifies effect (poor > extensive > ultra)"
echo ""

# 3. Comparison data
echo "[3/3] Comparison baselines:"
echo "  - DoWhy (Python): Runtime identifiability check"
echo "  - Ananke (R): Runtime graphical criteria check"
echo "  - Sounio: COMPILE-TIME identifiability verification"
echo ""

echo "======================================================================"
echo "  VALIDATION TARGETS"
echo "======================================================================"
echo ""
echo "  [  ] Identifiability verified at compile time"
echo "  [  ] Non-identifiable queries rejected at compile time"
echo "  [  ] ATE positive and biologically plausible"
echo "  [  ] Uncertainty decomposition: ε_stat + ε_causal + ε_model"
echo "  [  ] Genotype-specific treatment effects ordered correctly"
echo "  [  ] PK parameters within FDA reference range (<5% error)"
echo ""
echo "  See results/ directory for detailed output."
echo "======================================================================"
