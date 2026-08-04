#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
receipt="$root/scripts/research/receipts/cs6_u250_target23_chained_taylor41_v1"
generated=$(mktemp -d /tmp/cs6-chained-taylor41.XXXXXXXX)
trap 'rm -rf "$generated"' EXIT

(cd "$root" && sha256sum -c "$receipt/committed-files.sha256") >/dev/null
bash "$root/scripts/research/cs6_u250_target23_scaled_taylor16_gate.sh" >/dev/null
python3 "$root/scripts/research/cs6_u250_target23_chained_taylor41_generate.py" --out-dir "$generated"
cmp "$generated/chain.tsv" "$receipt/chain.tsv"
cmp "$generated/events.tsv" "$receipt/events.tsv"
cmp "$generated/inputs.bin" "$receipt/inputs.bin"
cmp "$generated/expected.bin" "$receipt/expected.bin"
cmp "$generated/hardware_inputs.bin" "$receipt/hardware_inputs.bin"
cmp "$generated/partitions.tsv" "$receipt/partitions.tsv"
cmp "$generated/summary.txt" "$receipt/summary.txt"
python3 "$root/scripts/research/cs6_u250_target23_chained_taylor41_verify.py" --receipt "$receipt"
python3 "$root/scripts/research/cs6_u250_target23_chained_taylor41_mutations.py" --receipt "$receipt"
grep -q '^TARGET23_CHAINED_TAYLOR41_CSIM_PASS=true$' "$receipt/hls-csim-vitis-2025.1.log"
grep -Fq '|Total                |        3|   2921|' "$receipt/hls-csynth-target23-chained-taylor41.rpt"
grep -Fq '|Utilization (%)      |       ~0|     23|        8|       16|' "$receipt/hls-csynth-target23-chained-taylor41.rpt"
grep -Fq '|Total                |        3|  22593|' "$receipt/hls-csynth-overbudget-v0.rpt"
grep -Fq 'Post Placement Timing Summary WNS=-121.338' "$receipt/vivado-impl-10mhz.log"
grep -Fq 'ERROR: [v++ 60-909] Specified kernel frequency not supported. Minimum supported value is 10 MHz.' "$receipt/vitis-link-5mhz.log"
grep -Fq 'ERROR: [v++ 60-626] Kernel link failed to complete' "$receipt/vitis-link-10mhz.log"
grep -Fq '| target23_chained_taylor41      | 1362459 [ 84.16%]' "$receipt/vivado-10mhz-kernel_util_placed.rpt"
grep -q '^LINK_ATTEMPT_10MHZ_POST_PLACEMENT_WNS_NS=-121.338$' "$receipt/physical-summary.txt"
grep -q '^XCLBIN_GENERATED=false$' "$receipt/physical-summary.txt"
echo TARGET23_CHAINED_TAYLOR41_CSIM_RECEIPT_PASS=true
echo TARGET23_CHAINED_TAYLOR41_CSYNTH_ESTIMATE_RECEIPT_PASS=true
echo TARGET23_CHAINED_TAYLOR41_PHYSICAL_TIMING_CLOSURE=false
echo TARGET23_CHAINED_TAYLOR41_PHYSICAL_FPGA_EXECUTION=false
echo TARGET23_CHAINED_TAYLOR41_DUAL_U250_EXECUTION=false
echo TARGET23_CHAINED_TAYLOR41_LOCAL_GATE_PASS=true
