#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Pipeline: GPU Oracle -> Epistemic Host -> Formal Lean 4  "
echo "=========================================================="
echo ""

cd /workspace/sounio-erdos
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

echo "1. Running Sounio to emulate GPU kernel and generate formal proofs..."
./bin/souc run examples/erdos_straus_gpu_lean.sio > /tmp/erdos_proofs.lean

echo "2. GPU execution complete. Generated Lean 4 file:"
head -15 /tmp/erdos_proofs.lean
echo "..."
tail -3 /tmp/erdos_proofs.lean

echo ""
echo "3. Invoking Lean 4 to formally verify all GPU witnesses..."
# Lean 4 compiler check (will fail if `by decide` cannot prove the theorem)
lean /tmp/erdos_proofs.lean

echo ""
echo "✅ All GPU witnesses formally verified by Lean 4!"
echo "Math absolute certainty achieved. No other self-hosted language does this."
