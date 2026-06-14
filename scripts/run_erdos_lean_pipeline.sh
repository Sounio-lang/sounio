#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Pipeline: GPU Oracle -> Epistemic Host -> Formal Lean 4  "
echo "=========================================================="
echo ""

cd /workspace/sounio-erdos
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib

# Private per-run temp dir (mode 700) — avoids predictable /tmp path,
# symlink clobber and TOCTOU (CWE-377/59/367).
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
PROOF_FILE="$TMP_DIR/erdos_proofs.lean"

echo "1. Running Sounio to emulate GPU kernel and generate formal proofs..."
./bin/souc run examples/erdos_straus_gpu_lean.sio > "$PROOF_FILE"

echo "2. GPU execution complete. Generated Lean 4 file:"
head -15 "$PROOF_FILE"
echo "..."
tail -3 "$PROOF_FILE"

echo ""
echo "3. Invoking Lean 4 to formally verify all GPU witnesses..."
# Lean 4 compiler check (will fail if `by decide` cannot prove the theorem)
lean "$PROOF_FILE"

echo ""
echo "✅ All GPU witnesses formally verified by Lean 4!"
echo "Math absolute certainty achieved. No other self-hosted language does this."
