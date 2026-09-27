#!/usr/bin/env bash
# measure_u250_power.sh — sample U250 board power while running host_san_scan_bench.
# Run on the DL380 with XRT sourced.
#
# Usage:
#   bash measure_u250_power.sh <xclbin> <artifacts_dir> <dataset> <seconds>
#
# Output: power_samples.csv + summary lines on stdout.
set -euo pipefail

XCLBIN="$1"
ARTIFACTS="$2"
DATASET="$3"
DURATION="$4"

OUTDIR="$(pwd)/u250_power_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTDIR"
CSV="$OUTDIR/power_samples.csv"
SUMMARY="$OUTDIR/summary.txt"

echo "time_s,power_w" > "$CSV"

echo "=== idle power (3 samples) ==="
for i in 1 2 3; do
    /opt/xilinx/xrt/bin/xrt-smi examine -r electrical 2>/dev/null | grep -E '^\s+Power\s+:' | awk '{print $3}'
    sleep 1
done | tee "$OUTDIR/idle_power.txt"
IDLE_AVG=$(awk '{s+=$1; n++} END {if(n) printf "%.3f", s/n; else print "0"}' "$OUTDIR/idle_power.txt")
echo "idle_avg_w=$IDLE_AVG" | tee "$SUMMARY"

echo "=== starting $DURATION s benchmark: dataset=$DATASET ==="
LD_LIBRARY_PATH=/opt/xilinx/xrt/lib \
    ./host_san_scan_bench "$XCLBIN" "$ARTIFACTS" "$DATASET" "$DURATION" > "$OUTDIR/bench.log" 2>&1 &
BENCH_PID=$!

# Sample power until benchmark exits
SAMPLE_COUNT=0
while kill -0 "$BENCH_PID" 2>/dev/null; do
    PWR=$(/opt/xilinx/xrt/bin/xrt-smi examine -r electrical 2>/dev/null | grep -E '^\s+Power\s+:' | awk '{print $3}')
    TS=$(date +%s.%N)
    echo "$TS,$PWR" >> "$CSV"
    SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
    sleep 1
done

wait "$BENCH_PID" || true

echo "=== power under load ($SAMPLE_COUNT samples) ==="
awk -F, 'NR>1 {s+=$2; n++; if($2<min||min=="") min=$2; if($2>max||max=="") max=$2} END {if(n) printf "load_avg_w=%.3f min_w=%.3f max_w=%.3f samples=%d\n", s/n, min, max, n}' "$CSV" | tee -a "$SUMMARY"

# Energy estimate from bench.log: total_samples and wall time
WALL_S=$(grep -oE 'wall=[0-9.]+' "$OUTDIR/bench.log" | head -1 | cut -d= -f2)
TOTAL_SAMPLES=$(grep -oE 'total_samples=[0-9]+' "$OUTDIR/bench.log" | head -1 | cut -d= -f2)

awk -F, -v idle="$IDLE_AVG" -v wall="$WALL_S" -v total="$TOTAL_SAMPLES" '
NR>1 {s+=$2; n++}
END {
    if(n && total>0 && wall>0) {
        load=s/n;
        delta=load-idle;
        if(delta<0) delta=0;
        energy_j=delta*wall;
        energy_per_sample_nj=energy_j/total*1e9;
        printf "delta_power_w=%.3f energy_total_j=%.3f total_samples=%s wall_s=%s energy_per_sample_nj=%.4f\n",
               delta, energy_j, total, wall, energy_per_sample_nj;
    } else {
        print "energy_calc_failed";
    }
}' "$CSV" | tee -a "$SUMMARY"

cat "$OUTDIR/bench.log" | tee -a "$SUMMARY"
echo "results in $OUTDIR"
