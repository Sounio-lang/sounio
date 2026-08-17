#!/usr/bin/env bash
# build_san_scan_xclbin.sh — v++ compile+link of krnl_san_scan for the U250.
# Run on the builder host with Vitis + XRT sourced:
#   source /opt/xilinx/xrt/setup.sh  (path may vary with Vitis install)
#   bash build_san_scan_xclbin.sh [hw|hw_emu|sw_emu]
# Environment:
#   PLATFORM — platform name or .xpfm path (default: xilinx_u250_gen3x16_xdma_4_1_202210_1)
set -euo pipefail
cd "$(dirname "$0")"
PLATFORM=${PLATFORM:-xilinx_u250_gen3x16_xdma_4_1_202210_1}
TARGET=${1:-hw}   # hw | hw_emu | sw_emu

v++ -c -k krnl_san_scan --platform "${PLATFORM}" --target "${TARGET}" \
    -o krnl_san_scan."${TARGET}".xo krnl_san_scan.cpp
v++ -l --platform "${PLATFORM}" --target "${TARGET}" \
    -o krnl_san_scan."${TARGET}".xclbin krnl_san_scan."${TARGET}".xo \
    --kernel_frequency 250
echo "XCLBIN_OK krnl_san_scan.${TARGET}.xclbin"
