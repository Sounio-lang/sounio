#!/usr/bin/env bash
# ptx_ablation_runner.sh — Run shadow register ablation on L4
#
# Dispatches two PTX variants on the L4 GPU:
#   1. Baseline: epistemic_gemm_sm7_4096.ptx (no shadows)
#   2. Augmented: epistemic_gemm_sm7_4096_shadowed.ptx (with shadow ops)
#
# The difference in GFLOPS measures the cost of shadow register operations
# (epsilon propagation, validity tracking, provenance merge).
#
# Usage:
#   bash scripts/gpu/ptx_ablation_runner.sh
#
# Requires:
#   - SSH access to gpu-appliance-l4 (10.100.100.215)
#   - /tmp/epistemic_gemm_sm7_4096.ptx on the L4

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_HOST="${GPU_HOST:-10.100.100.215}"
GPU_USER="${GPU_USER:-demetrios}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"
PTX_BASELINE="${PTX_BASELINE:-/tmp/epistemic_gemm_sm7_4096.ptx}"
PTX_AUGMENTED="${PTX_AUGMENTED:-/tmp/epistemic_gemm_sm7_4096_shadowed.ptx}"
ITERS="${GEMM_ITERS:-8}"
DIMS="${GEMM_DIMS:-1024 2048 4096 8192}"
REPORT_PATH="${ABLATION_REPORT:-${ROOT_DIR}/artifacts/omega/ablation_report.v1.json}"

echo "================================================================"
echo "  PTX Shadow Register Ablation"
echo "================================================================"
echo "  GPU host: ${GPU_HOST}"
echo "  Baseline PTX: ${PTX_BASELINE}"
echo "  Augmented PTX: ${PTX_AUGMENTED}"
echo "  Dimensions: ${DIMS}"
echo "  Iterations: ${ITERS}"
echo ""

# Step 1: Generate augmented PTX on the L4
echo "[1/4] Generating shadow-augmented PTX on L4..."
ssh "${GPU_USER}@${GPU_HOST}" "cd ${REMOTE_DIR} && python3 scripts/gpu/ptx_shadow_augment.py ${PTX_BASELINE} ${PTX_AUGMENTED}"

# Step 2: Run baseline (no shadows)
echo ""
echo "[2/4] Running baseline (no shadows)..."
BASELINE_RESULTS=""
for dim in ${DIMS}; do
    echo "  --- Baseline ${dim}x${dim} ---"
    result=$(ssh "${GPU_USER}@${GPU_HOST}" "cd ${REMOTE_DIR} && GEMM_M=${dim} GEMM_N=${dim} GEMM_K=${dim} GEMM_ITERS=${ITERS} GEMM_PTX_FILE=${PTX_BASELINE} python3 scripts/gpu/cuda_gemm_dispatch.py 2>&1" | grep -oP 'GFLOPS:\s+\K[0-9.]+' || echo "0")
    echo "    GFLOPS: ${result}"
    BASELINE_RESULTS="${BASELINE_RESULTS}${dim}:${result} "
done

# Step 3: Run augmented (with shadows)
echo ""
echo "[3/4] Running augmented (with shadows)..."
AUGMENTED_RESULTS=""
for dim in ${DIMS}; do
    echo "  --- Augmented ${dim}x${dim} ---"
    result=$(ssh "${GPU_USER}@${GPU_HOST}" "cd ${REMOTE_DIR} && GEMM_M=${dim} GEMM_N=${dim} GEMM_K=${dim} GEMM_ITERS=${ITERS} GEMM_PTX_FILE=${PTX_AUGMENTED} python3 scripts/gpu/cuda_gemm_dispatch.py 2>&1" | grep -oP 'GFLOPS:\s+\K[0-9.]+' || echo "0")
    echo "    GFLOPS: ${result}"
    AUGMENTED_RESULTS="${AUGMENTED_RESULTS}${dim}:${result} "
done

# Step 4: Generate report
echo ""
echo "[4/4] Generating ablation report..."
python3 - "${BASELINE_RESULTS}" "${AUGMENTED_RESULTS}" "${REPORT_PATH}" <<'PYTHON'
import json
import math
import sys
import time

baseline_str = sys.argv[1].strip()
augmented_str = sys.argv[2].strip()
report_path = sys.argv[3]

def parse_results(s):
    d = {}
    for pair in s.split():
        if ':' in pair:
            dim, gflops = pair.split(':')
            d[int(dim)] = float(gflops)
    return d

baseline = parse_results(baseline_str)
augmented = parse_results(augmented_str)

dims_data = {}
overhead_ratios = []
for dim in sorted(set(baseline.keys()) | set(augmented.keys())):
    b = baseline.get(dim, 0)
    a = augmented.get(dim, 0)
    if b > 0 and a > 0:
        overhead = b / a
        overhead_ratios.append(overhead)
    else:
        overhead = None
    dims_data[str(dim)] = {
        "baseline_gflops": b,
        "augmented_gflops": a,
        "shadow_overhead_x": round(overhead, 3) if overhead else None,
    }

geomean = None
if overhead_ratios:
    log_sum = sum(math.log(x) for x in overhead_ratios)
    geomean = round(math.exp(log_sum / len(overhead_ratios)), 3)

report = {
    "schema": "sounio.benchmark.shadow-ablation.v1",
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "description": "Shadow register ablation: baseline PTX (no shadows) vs augmented PTX (with GUM eps propagation, validity AND, provenance XOR after every FMA)",
    "baseline_ptx": "epistemic_gemm_sm7_4096.ptx",
    "augmented_ptx": "epistemic_gemm_sm7_4096_shadowed.ptx",
    "augmentation": {
        "ops_per_fma": 10,
        "ops": ["2x mul.f32 (|a|*eps)", "2x add.f32 (eps sum)", "2x mul.f32 (square)", "1x sqrt.approx.f32 (quadrature)", "1x abs.f32 (a)", "1x abs.f32 (b)", "1x and.pred (validity)", "1x xor.b32 (provenance)"],
    },
    "dimensions": dims_data,
    "aggregate": {
        "geomean_shadow_overhead_x": geomean,
    },
}

import os
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"Report written to: {report_path}")
print()
print("  Shadow Ablation Results:")
print(f"  {'Dim':>6}  {'Baseline':>10}  {'Shadowed':>10}  {'Overhead':>10}")
for dim in sorted(dims_data.keys(), key=int):
    d = dims_data[dim]
    oh = f"{d['shadow_overhead_x']:.2f}x" if d['shadow_overhead_x'] else "N/A"
    print(f"  {dim:>6}  {d['baseline_gflops']:>10.1f}  {d['augmented_gflops']:>10.1f}  {oh:>10}")
if geomean:
    print(f"  {'geomean':>6}  {'':>10}  {'':>10}  {geomean:.2f}x")
PYTHON

echo ""
echo "================================================================"
echo "  Ablation complete. Report: ${REPORT_PATH}"
echo "================================================================"
