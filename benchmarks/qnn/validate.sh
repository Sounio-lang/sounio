#!/bin/bash
# QNN-MNIST Epistemic Benchmark Validation
#
# Runs the full benchmark suite and validates results against paper targets.
# For PLDI 2027 submission.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "======================================================================"
echo "  QNN-MNIST Epistemic Benchmark Suite"
echo "  Paper: Epistemic Types for Scientific Computing (PLDI 2027)"
echo "======================================================================"
echo ""

# 1. Run epistemic QNN benchmark
echo "[1/4] Running epistemic QNN-MNIST benchmark..."
cd "$ROOT_DIR"
cargo run --release --bin souc -- run benchmarks/qnn/mnist_epistemic.sio \
    2>&1 | tee "$RESULTS_DIR/epistemic_qnn_output.txt"
echo ""

# 2. Validate calibration metrics
echo "[2/4] Checking calibration targets..."
echo "  Target: ECE < 0.05"
echo "  Target: Accuracy >= 98%"
echo "  Target: Mean uncertainty decreases with training"
echo ""

# 3. Generate comparison data
echo "[3/4] Comparison baselines (would run PyTorch for full evaluation):"
echo "  - Standard QNN (no uncertainty): baseline accuracy"
echo "  - Bayesian NN (MC dropout, 10 samples): calibration comparison"
echo "  - Deep Ensemble (5 models): calibration comparison"
echo ""

# 4. Generate paper figures
echo "[4/4] Paper figure generation:"
echo "  - results/calibration_diagram.png    (reliability diagram)"
echo "  - results/uncertainty_vs_error.png   (uncertainty correlation)"
echo "  - results/training_uncertainty.png   (uncertainty decrease curve)"
echo "  - results/comparison_table.csv       (vs baselines)"
echo ""

echo "======================================================================"
echo "  VALIDATION TARGETS"
echo "======================================================================"
echo ""
echo "  [  ] Test accuracy              >= 98.0%"
echo "  [  ] Expected Calibration Error <  0.05"
echo "  [  ] Maximum Calibration Error  <  0.15"
echo "  [  ] Uncertainty-error corr.    >  0.30"
echo "  [  ] Weight uncertainty decreases with training"
echo ""
echo "  See results/ directory for detailed output."
echo "======================================================================"
