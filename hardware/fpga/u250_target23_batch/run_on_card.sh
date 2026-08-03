#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || { echo "usage: $0 XCLBIN RUN_DIR REPEATS" >&2; exit 2; }
xclbin=$1
run_dir=$2
repeats=$3
[[ -f $xclbin && -x $run_dir/host && $repeats =~ ^[1-9][0-9]*$ ]] || exit 2

export LD_LIBRARY_PATH=/opt/dl380-libs:/opt/xilinx/xrt/lib
xrt_smi=/opt/xilinx/xrt/bin/xrt-smi
results="$run_dir/card-results.tsv"
marker="$results.kernel-running"
done_file="$run_dir/sampling.done"

"$xrt_smi" examine --report electrical > "$run_dir/pre-run-electrical.txt"
"$xrt_smi" examine --report thermal > "$run_dir/pre-run-thermal.txt"
printf 'UTC\tPHASE\tPOWER_WATTS\n' > "$run_dir/power-samples.tsv"
(
  while [[ ! -f $done_file ]]; do
    if [[ -f $marker ]]; then
      power=$("$xrt_smi" examine --report electrical | awk '$1 == "Power" {print $3; exit}')
      printf '%s\tkernel\t%s\n' "$(date -u +%FT%T.%NZ)" "$power" >> "$run_dir/power-samples.tsv"
    fi
    sleep 0.2
  done
) &
sampler=$!

set +e
"$run_dir/host" "$xclbin" "$run_dir/inputs.bin" "$run_dir/expected.bin" \
  "$run_dir/decimal-results.tsv" "$repeats" "$results" \
  > "$run_dir/hardware-summary.txt" 2> "$run_dir/hardware-stderr.txt"
host_rc=$?
set -e
printf 'done\n' > "$done_file"
wait "$sampler"
"$xrt_smi" examine --report dynamic-regions > "$run_dir/post-run-dynamic-regions.txt"
"$xrt_smi" examine --report electrical > "$run_dir/post-run-electrical.txt"
"$xrt_smi" examine --report thermal > "$run_dir/post-run-thermal.txt"
printf 'HOST_RC=%s\n' "$host_rc" > "$run_dir/host-execution.txt"
exit "$host_rc"
