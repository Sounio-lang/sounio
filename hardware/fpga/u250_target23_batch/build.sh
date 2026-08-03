#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
build_dir=${TARGET23_BUILD_DIR:-"$script_dir/build"}
amd_root=${AMD_VITIS_ROOT:-/opt/amd/2025.1}
settings="$amd_root/Vitis/settings64.sh"
platform_name=xilinx_u250_gen3x16_xdma_4_1_202210_1

[[ -f $settings ]] || { echo "missing Vitis settings: $settings" >&2; exit 2; }
set +u
source "$settings"
set -u
platform=${U250_PLATFORM_XPFM:-}
if [[ -z $platform ]]; then
  platform=$(find /opt/xilinx/platforms -type f -name "$platform_name.xpfm" -print -quit)
fi
[[ -f $platform ]] || { echo "missing U250 platform" >&2; exit 2; }

mkdir -p "$build_dir"
cd "$build_dir"
platforminfo --platform "$platform" > platforminfo.txt
v++ --version > vpp-version.txt
[[ -f ${TARGET23_INPUTS:-} && -f ${TARGET23_EXPECTED:-} ]] || {
  echo "TARGET23_INPUTS and TARGET23_EXPECTED must name frozen binaries" >&2
  exit 2
}
vitis-run --mode hls --tcl "$script_dir/run_hls_csim.tcl" 2>&1 | tee csim.log
grep -qx 'TARGET23_U250_CSIM_PASS=true' \
  "$build_dir/target23_batch_csim/solution1/csim/report/target23_batch_csim.log"
v++ -t hw --platform "$platform" --save-temps -g -c -k target23_batch \
  -o target23_batch.xo "$script_dir/kernel.cpp" 2>&1 | tee compile.log
v++ -t hw --platform "$platform" --save-temps -g -l \
  --config "$script_dir/target23_batch.cfg" -o target23_batch.xclbin \
  target23_batch.xo 2>&1 | tee link.log
sha256sum "$script_dir/kernel.cpp" "$script_dir/testbench.cpp" \
  "$script_dir/run_hls_csim.tcl" "$script_dir/target23_batch.cfg" \
  "$TARGET23_INPUTS" "$TARGET23_EXPECTED" \
  target23_batch.xo target23_batch.xclbin > SHA256SUMS
echo "TARGET23_U250_XCLBIN_BUILD_PASS $build_dir/target23_batch.xclbin"
