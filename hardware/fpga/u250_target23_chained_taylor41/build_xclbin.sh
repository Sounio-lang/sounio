#!/usr/bin/env bash
set -euo pipefail

platform=${PLATFORM:-/opt/xilinx/platforms/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm}
build_dir=${BUILD_DIR:-build}
hls_clock_mhz=${HLS_CLOCK_MHZ:-100}
kernel_frequency_mhz=${KERNEL_FREQ_MHZ:-10}
hls_clock_hz=$((hls_clock_mhz * 1000000))
mkdir -p "$build_dir"

v++ -c -t hw --platform "$platform" \
    --hls.clock "$hls_clock_hz:target23_chained_taylor41" \
    --temp_dir "$build_dir/_x_compile" \
    -k target23_chained_taylor41 \
    kernel.cpp -o "$build_dir/target23_chained_taylor41.xo"
v++ -l -t hw --platform "$platform" \
    --kernel_frequency "$kernel_frequency_mhz" \
    --temp_dir "$build_dir/_x_link" \
    --config connectivity.cfg \
    "$build_dir/target23_chained_taylor41.xo" \
    -o "$build_dir/target23_chained_taylor41.xclbin"
