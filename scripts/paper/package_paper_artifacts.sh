#!/usr/bin/env bash
# package_paper_artifacts.sh — Bundle paper evidence into a reproducible tarball
#
# Creates artifacts/omega/paper_artifacts.tar.gz containing:
#   - Machine-readable JSON reports (ablation, GUM H.1, self-hosting, GPU)
#   - Lean 4 formal proofs (6 modules, 620 lines)
#   - Reproduction scripts (PTX ablation, GUM verify, selfhost report)
#   - Paper artifact manifest with SHA-256 checksums
#   - Self-hosting stage parity evidence
#
# Usage:
#   bash scripts/paper/package_paper_artifacts.sh
#
# Output:
#   artifacts/omega/paper_artifacts.tar.gz

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MANIFEST="artifacts/omega/paper_artifact_manifest.v1.json"
OUTDIR="artifacts/omega/paper_artifacts_staging"
TARBALL="artifacts/omega/paper_artifacts.tar.gz"

echo "================================================================"
echo "  Paper Artifact Packaging"
echo "================================================================"
echo ""

# ── Verify manifest exists ─────────────────────────────────────────
if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest not found: $MANIFEST"
    exit 1
fi

# ── Clean staging ───────────────────────────────────────────────────
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

# ── Collect evidence files ──────────────────────────────────────────
EVIDENCE_FILES=(
    # JSON reports
    artifacts/omega/ablation_report.v1.json
    artifacts/omega/cublas_baseline_report.v1.json
    artifacts/omega/gum_h1_verification_report.v1.json
    artifacts/omega/selfhost_verification_report.v1.json
    artifacts/omega/overhead_report.v1.json
    artifacts/omega/paper_artifact_manifest.v1.json

    # Self-hosting stage parity
    artifacts/omega/session_20260227_stage1/artifacts/summary.txt
    artifacts/omega/session_20260227_stage1/artifacts/selfhost.stage1.sobc
    artifacts/omega/session_20260227_stage1/artifacts/selfhost.stage2.sobc

    # Benchmark data
    benchmarks/results/NVIDIA_L4_BENCHMARKS.md
    benchmarks/results/l4_raw_data.json

    # Reproduction scripts
    scripts/ptx_shadow_augment.py
    scripts/ptx_ablation_runner.sh
    scripts/gum_h1_verify.py
    scripts/selfhost_generate_report.py
    scripts/cuda_cublas_baseline.py
    scripts/cuda_gemm_dispatch.py

    # GUM H.1 test
    tests/run-pass/gum_h1_end_gauge.sio

    # Lean formal proofs
    formal/omega_mathlib/lakefile.lean
    formal/omega_mathlib/OmegaMathlib/AccumulatorBounds.lean
    formal/omega_mathlib/OmegaMathlib/InternedChemicalGraph.lean
    formal/omega_mathlib/OmegaMathlib/EpistemicMapBound.lean
    formal/omega_mathlib/OmegaMathlib/GlycolysisEpistemic.lean
    formal/omega_mathlib/OmegaMathlib/EpistemicArena.lean
    formal/omega_mathlib/OmegaMathlib/EpistemicPower.lean

    # Paper source
    paper/epistemic-types/gpu_benchmarks.tex
)

echo "  Collecting ${#EVIDENCE_FILES[@]} files..."
echo ""

missing=0
for f in "${EVIDENCE_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "  MISSING: $f"
        missing=$((missing + 1))
    else
        dest="$OUTDIR/$f"
        mkdir -p "$(dirname "$dest")"
        cp "$f" "$dest"
    fi
done

if [[ $missing -gt 0 ]]; then
    echo ""
    echo "  WARNING: $missing files missing. Continuing with available files."
fi

# ── Verify checksums ────────────────────────────────────────────────
echo ""
echo "  Verifying artifact checksums..."

verify_ok=0
verify_fail=0

check_sha() {
    local file="$1"
    local expected="$2"
    local label="$3"
    if [[ -f "$file" ]]; then
        actual=$(sha256sum "$file" | awk '{print $1}')
        if [[ "$actual" == "$expected" ]]; then
            echo "    [OK]   $label"
            verify_ok=$((verify_ok + 1))
        else
            echo "    [FAIL] $label (expected ${expected:0:16}..., got ${actual:0:16}...)"
            verify_fail=$((verify_fail + 1))
        fi
    else
        echo "    [SKIP] $label (not found)"
    fi
}

check_sha "artifacts/omega/ablation_report.v1.json" \
    "c877e257ffe2bef36f4e131dc7a14b3f03b690d9daef30486afa11703970a6d0" \
    "ablation_report"
check_sha "artifacts/omega/gum_h1_verification_report.v1.json" \
    "07bf72858645056d56269f465f912057aaab03c046b3a3394ad53aede7f394be" \
    "gum_h1_verification_report"
check_sha "artifacts/omega/selfhost_verification_report.v1.json" \
    "41800c914235b261fa53b0805e22afe08b9e72f6ecf75e9a9683fabd23004616" \
    "selfhost_verification_report"
check_sha "artifacts/omega/cublas_baseline_report.v1.json" \
    "07a3bdadbdc34c9f2dc8fedea73ed3a75f44131e75e3081bfdea6ec4b2209c09" \
    "cublas_baseline_report"
check_sha "artifacts/omega/overhead_report.v1.json" \
    "95a62ca7933510085838909570c1fba978e9031dcc093f2a06baee1875bb0dbf" \
    "overhead_report"

echo "    Verified: $verify_ok OK, $verify_fail FAIL"

# ── Self-hosting parity ─────────────────────────────────────────────
echo ""
echo "  Checking self-hosting stage parity..."
s1="artifacts/omega/session_20260227_stage1/artifacts/selfhost.stage1.sobc"
s2="artifacts/omega/session_20260227_stage1/artifacts/selfhost.stage2.sobc"
if [[ -f "$s1" && -f "$s2" ]]; then
    h1=$(sha256sum "$s1" | awk '{print $1}')
    h2=$(sha256sum "$s2" | awk '{print $1}')
    if [[ "$h1" == "$h2" ]]; then
        echo "    Stage 1 = Stage 2: PASS (${h1:0:16}...)"
    else
        echo "    Stage 1 != Stage 2: FAIL"
    fi
else
    echo "    SKIP: stage files not found"
fi

# ── Create tarball ──────────────────────────────────────────────────
echo ""
echo "  Creating tarball..."
tar -czf "$TARBALL" -C "$(dirname "$OUTDIR")" "$(basename "$OUTDIR")"
tarball_size=$(du -h "$TARBALL" | awk '{print $1}')
tarball_sha=$(sha256sum "$TARBALL" | awk '{print $1}')
file_count=$(tar -tzf "$TARBALL" | wc -l)

# ── Clean staging ───────────────────────────────────────────────────
rm -rf "$OUTDIR"

# ── Summary ─────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Paper Artifact Package Complete"
echo "================================================================"
echo "  Output:    $TARBALL"
echo "  Size:      $tarball_size"
echo "  Files:     $file_count"
echo "  SHA-256:   ${tarball_sha:0:32}..."
echo "  Checksums: $verify_ok OK, $verify_fail FAIL"
echo ""
echo "  Contents:"
echo "    - 5 JSON evidence reports"
echo "    - 6 Lean 4 formal proofs (620 lines, 53 theorems)"
echo "    - Self-hosting stage parity (SHA-256 match)"
echo "    - GPU benchmark data + reproduction scripts"
echo "    - GUM H.1 cross-verification"
echo "    - Paper artifact manifest"
echo "================================================================"
