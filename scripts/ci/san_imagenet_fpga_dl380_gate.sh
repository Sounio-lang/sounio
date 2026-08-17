#!/usr/bin/env bash
# CI gate for the SAN-ImageNet FPGA/DL380 contract.
#
# Runs the executable contract scripts/research/san_imagenet_fpga_dl380.py and
# requires:
#   - SAN_IMAGENET_FPGA_DL380_VERDICT I_GREEN (clauses I1-I8: metering
#     conservation at real architecture scale, convergence, anti-Goodhart,
#     necessary/gratuitous separation, suffering bounds, FPGA bit-accuracy
#     incl. the 1.2M-sample stress scan, real exits, patient channel)
#   - the spec doc and the HLS reference outlines exist with their required
#     sections/markers (so contract and design documents cannot drift apart
#     silently).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC="${REPO_ROOT}/scripts/research/san_imagenet_fpga_dl380.py"
SPEC="${REPO_ROOT}/docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md"
KERNEL="${REPO_ROOT}/hardware/fpga/u250_catastrophe_scan/krnl_san_scan.cpp"
HOST="${REPO_ROOT}/hardware/fpga/u250_catastrophe_scan/host_san_scan.cpp"

for f in "${SRC}" "${SPEC}" "${KERNEL}" "${HOST}"; do
    if [[ ! -f "${f}" ]]; then
        echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: missing ${f}"
        exit 1
    fi
done

PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python3 || true)"
fi
if [[ -z "${PY}" ]]; then
    echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: no python interpreter"
    exit 1
fi

echo "Running SAN-ImageNet FPGA/DL380 contract (I1..I8)..."
OUT="$("${PY}" "${SRC}")" || {
    echo "${OUT}"
    echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: contract exited non-zero"
    exit 1
}
echo "${OUT}"

if ! grep -q '^SAN_IMAGENET_FPGA_DL380_VERDICT I_GREEN' <<<"${OUT}"; then
    echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: verdict not I_GREEN"
    exit 1
fi

# The 1.2M-sample ImageNet-completo stress scan must stay exact (I6).
if ! grep -q '^  I6\[stress-1.2M\]: PASS' <<<"${OUT}"; then
    echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: 1.2M stress scan not green"
    exit 1
fi

# The DL380 preflight must execute and report honestly (T3).
if ! grep -q '^  DL380_PREFLIGHT ' <<<"${OUT}"; then
    echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: DL380 preflight did not execute"
    exit 1
fi

# Spec/outline consistency: required sections and markers.
for marker in \
    "What this is NOT" \
    "Theorems" \
    "san_imagenet_fpga_dl380.py" \
    "krnl_san_scan.cpp" \
    "ESTIMATE" \
    "no_consciousness_claim"; do
    if ! grep -qF "${marker}" "${SPEC}"; then
        echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: spec missing marker: ${marker}"
        exit 1
    fi
done
for marker in "q_delta" "n_catastrophe" "flop_macs" "krnl_san_scan"; do
    if ! grep -qF "${marker}" "${KERNEL}"; then
        echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: kernel outline missing: ${marker}"
        exit 1
    fi
done
for marker in "xilinx_u250" "Q0.15" "golden model"; do
    if ! grep -qF "${marker}" "${HOST}"; then
        echo "SAN_IMAGENET_FPGA_DL380_GATE_FAIL: host outline missing: ${marker}"
        exit 1
    fi
done

echo "SAN_IMAGENET_FPGA_DL380_GATE_OK"
