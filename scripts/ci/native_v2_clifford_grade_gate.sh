#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_CLIFFORD_GRADE_DIR:-$(mktemp -d /tmp/sounio-native-v2-clifford-grade.XXXXXX)}"

echo "[native-v2-clifford-grade] out=$OUT_DIR"
echo "[native-v2-clifford-grade] Grade as first-class IR attribute: IrCliffordGradeSel + IrCliffordGradeMul"
echo "[native-v2-clifford-grade] Dead-grade pruning: grade_known && grade_mask==0 → VXORPD zero"
echo "[native-v2-clifford-grade] No existing compiler/IR has grade as a first-class attribute"

MANIFEST_FILE="$(mktemp /tmp/clifford-grade-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
clifford_grade_tracking	tests/native-v2/science_spine/clifford_grade_tracking.sio	0	tests/native-v2/science_spine/clifford_grade_tracking.stdout	clifford_grade_ir
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-clifford-grade] PASS: Clifford grade IR attribute gate"
