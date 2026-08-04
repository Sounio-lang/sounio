#!/usr/bin/env bash
set -euo pipefail

platform=${PLATFORM:-/opt/xilinx/platforms/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm}
build_dir=${BUILD_DIR:-build}
kernel_frequency_mhz=${KERNEL_FREQ_MHZ:-200}
mkdir -p "$build_dir"

v++ -c -t hw --platform "$platform" \
    -k target23_scaled_taylor16 \
    kernel.cpp -o "$build_dir/target23_scaled_taylor16.xo"
v++ -l -t hw --platform "$platform" \
    --kernel_frequency "$kernel_frequency_mhz" \
    --config connectivity.cfg \
    "$build_dir/target23_scaled_taylor16.xo" \
    -o "$build_dir/target23_scaled_taylor16.xclbin"
