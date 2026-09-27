#!/usr/bin/env bash
# U250 SAN-scan benchmark campaign.
# Runs correctness + sustained-throughput benchmarks for every on-target
# cohort currently staged on the DL380, and writes a single receipt.
#
# Usage (inside DL380 chroot, XRT sourced):
#   bash /tmp/san_t3/run_u250_benchmark_campaign.sh
set -euo pipefail

XCLBIN="/tmp/san_t3/build/krnl_san_scan.hw.xclbin"
HOST_SCAN="/tmp/san_t3/build/host_san_scan"
HOST_BENCH="/tmp/san_t3/build/host_san_scan_bench"
ARTIFACTS=(
  "/tmp/san_t3/artifacts:val_resnet"
  "/tmp/san_t3/artifacts:val_vit"
  "/tmp/san_t3/artifacts:stress_1p2M"
  "/tmp/san_t3/artifacts_imagenette2:val_imagenette"
)
BENCH_SECONDS="${U250_BENCH_SECONDS:-10}"

OUTDIR="/tmp/san_t3/u250_benchmark_campaign_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
LOG="$OUTDIR/campaign.log"

exec > >(tee -a "$LOG")
exec 2>&1

echo "=== U250 SAN-scan benchmark campaign ==="
echo "xclbin: $XCLBIN"
echo "bench_duration: ${BENCH_SECONDS}s"
echo ""

for entry in "${ARTIFACTS[@]}"; do
  dir="${entry%%:*}"
  ds="${entry##*:}"
  echo "--- dataset: $ds (dir: $dir) ---"
  LD_LIBRARY_PATH=/opt/xilinx/xrt/lib "$HOST_SCAN" "$XCLBIN" "$dir" "$ds" || true
  echo ""
  LD_LIBRARY_PATH=/opt/xilinx/xrt/lib "$HOST_BENCH" "$XCLBIN" "$dir" "$ds" "$BENCH_SECONDS" || true
  echo ""
done

echo "=== Campaign complete. Receipt: $LOG ==="
