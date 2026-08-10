#!/usr/bin/env bash
# CI gate for the U250 catastrophe-scan FPGA contract.
#
# Compiles the bit-accurate kernel model scripts/research/fpga_census_kernel_model.c,
# runs it, and requires:
#   - FPGA_CENSUS_MODEL_VERDICT PASS (M1-M5: bit-parity path == audited integer
#     criterion at b=4..9, growth law at b=4..8, Z(9)=249084 out-of-sample,
#     L8 histogram equality, cycle model)
#   - the spec doc and the HLS reference outline exist with their required
#     sections (so the contract and the design document cannot drift apart
#     silently).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC="${REPO_ROOT}/scripts/research/fpga_census_kernel_model.c"
SPEC="${REPO_ROOT}/docs/research/u250_catastrophe_scan_fpga_spec_2026-07-26.md"
KERNEL="${REPO_ROOT}/hardware/fpga/u250_catastrophe_scan/krnl_census.cpp"
HOST="${REPO_ROOT}/hardware/fpga/u250_catastrophe_scan/host.cpp"

for f in "${SRC}" "${SPEC}" "${KERNEL}" "${HOST}"; do
    if [[ ! -f "${f}" ]]; then
        echo "FPGA_CATASTROPHE_SCAN_GATE_FAIL: missing ${f}"
        exit 1
    fi
done

CC_BIN="${CC:-cc}"
if ! command -v "${CC_BIN}" >/dev/null 2>&1; then
    echo "FPGA_CATASTROPHE_SCAN_GATE_FAIL: no C compiler (${CC_BIN})"
    exit 1
fi

SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT
BIN="${SCRATCH}/fpga_census_kernel_model"

echo "Building FPGA census kernel bit-accurate model..."
"${CC_BIN}" -O2 -Wall -Wextra -o "${BIN}" "${SRC}"

echo "Running FPGA census kernel model (levels 4..9)..."
OUT="$("${BIN}")" || {
    echo "${OUT}"
    echo "FPGA_CATASTROPHE_SCAN_GATE_FAIL: model exited non-zero"
    exit 1
}
echo "${OUT}"

if ! grep -q '^FPGA_CENSUS_MODEL_VERDICT PASS$' <<<"${OUT}"; then
    echo "FPGA_CATASTROPHE_SCAN_GATE_FAIL: verdict not PASS"
    exit 1
fi

# The out-of-sample L9 confirmation must stay green (M3).
if ! grep -q '^FPGA_MODEL_LEVEL b=9 .*law_ok=1 path_mismatches=0' <<<"${OUT}"; then
    echo "FPGA_CATASTROPHE_SCAN_GATE_FAIL: L9 out-of-sample check not green"
    exit 1
fi

# Spec/outline consistency: required sections and markers.
for marker in \
    "Resource estimates (Phase 1, L9 configuration, 16 PEs)" \
    "Speedup estimates over CPU" \
    "What this is NOT" \
    "fpga_census_kernel_model.c" \
    "Z(9) = 249084"; do
    if ! grep -qF "${marker}" "${SPEC}"; then
        echo "FPGA_CATASTROPHE_SCAN_GATE_FAIL: spec missing marker: ${marker}"
        exit 1
    fi
done
for marker in "perm_l" "popcount_row" "krnl_census"; do
    if ! grep -qF "${marker}" "${KERNEL}"; then
        echo "FPGA_CATASTROPHE_SCAN_GATE_FAIL: kernel outline missing: ${marker}"
        exit 1
    fi
done

echo "FPGA_CATASTROPHE_SCAN_GATE_OK"
