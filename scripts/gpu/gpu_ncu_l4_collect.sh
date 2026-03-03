#!/usr/bin/env bash
# gpu_ncu_l4_collect.sh
# Collect Nsight Compute counters for L4 benchmark variants.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/artifacts/omega/l4_runs}"
TIMESTAMP_UTC="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
REPORT_PATH="${REPORT_PATH:-$REPORT_DIR/l4_ncu_metrics.${TIMESTAMP_UTC}.v1.json}"

GPU_HOST="${GPU_HOST:-10.100.100.215}"
GPU_USER="${GPU_USER:-demetrios}"
REMOTE_DIR="${REMOTE_DIR:-~/work/sounio}"

NCU_DIM="${NCU_DIM:-4096}"
NCU_ITERS="${NCU_ITERS:-5}"

VALUE_ONLY_PTX_FILE="${VALUE_ONLY_PTX_FILE:-/tmp/epistemic_gemm_sm7_4096.ptx}"
SHADOW_STRICT_PTX_FILE="${SHADOW_STRICT_PTX_FILE:-/tmp/epistemic_tensor_core_shadow_strict_sm89.ptx}"
SHADOW_FAST_PTX_FILE="${SHADOW_FAST_PTX_FILE:-/tmp/epistemic_tensor_core_shadow_fast_sm89.ptx}"

NCU_METRICS="${NCU_METRICS:-sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct}"

mkdir -p "$REPORT_DIR"

SSH_OPTS=(
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=no
)

run_remote() {
  local cmd="$1"
  ssh "${SSH_OPTS[@]}" "${GPU_USER}@${GPU_HOST}" "$cmd"
}

ROWS_TSV="$(mktemp "${TMPDIR:-/tmp}/l4_ncu_rows.${TIMESTAMP_UTC}.XXXXXX.tsv")"
cleanup() {
  rm -f "$ROWS_TSV"
}
trap cleanup EXIT

printf 'variant\tstatus\tptx_file\tlog_path\n' > "$ROWS_TSV"

ncu_available="false"
if run_remote "command -v ncu >/dev/null 2>&1"; then
  ncu_available="true"
fi

run_variant_ncu() {
  local variant="$1"
  local ptx_file="$2"
  local log_path="$3"

  if ! run_remote "test -f ${ptx_file}" >/dev/null 2>&1; then
    printf 'missing PTX for %s: %s\n' "$variant" "$ptx_file" > "$log_path"
    printf '%s\tNOT_FOUND\t%s\t%s\n' "$variant" "$ptx_file" "$log_path" >> "$ROWS_TSV"
    return
  fi

  local cmd
  cmd="cd ${REMOTE_DIR} && GEMM_M=${NCU_DIM} GEMM_N=${NCU_DIM} GEMM_K=${NCU_DIM} GEMM_ITERS=${NCU_ITERS} GEMM_PTX_FILE=${ptx_file} ncu --target-processes all --metrics ${NCU_METRICS} --csv python3 scripts/gpu/cuda_gemm_dispatch.py"

  if run_remote "$cmd" >"$log_path" 2>&1; then
    printf '%s\tPASS\t%s\t%s\n' "$variant" "$ptx_file" "$log_path" >> "$ROWS_TSV"
  else
    printf '%s\tFAIL\t%s\t%s\n' "$variant" "$ptx_file" "$log_path" >> "$ROWS_TSV"
  fi
}

if [ "$ncu_available" = "true" ]; then
  run_variant_ncu "value_only_baseline" "$VALUE_ONLY_PTX_FILE" "$REPORT_DIR/l4_ncu.${TIMESTAMP_UTC}.value_only_baseline.log"
  run_variant_ncu "shadow_strict" "$SHADOW_STRICT_PTX_FILE" "$REPORT_DIR/l4_ncu.${TIMESTAMP_UTC}.shadow_strict.log"
  run_variant_ncu "shadow_fast" "$SHADOW_FAST_PTX_FILE" "$REPORT_DIR/l4_ncu.${TIMESTAMP_UTC}.shadow_fast.log"
fi

export GPU_HOST GPU_USER REMOTE_DIR NCU_DIM NCU_ITERS NCU_METRICS REPORT_PATH
export VALUE_ONLY_PTX_FILE SHADOW_STRICT_PTX_FILE SHADOW_FAST_PTX_FILE
export ncu_available

python3 - "$ROWS_TSV" <<'PY'
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone

rows_path = sys.argv[1]
report_path = os.environ["REPORT_PATH"]
ncu_available = os.environ.get("ncu_available", "false") == "true"

variant_rows = []
with open(rows_path, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        variant_rows.append(row)

metric_aliases = {
    "tensor_utilization_pct": [
        "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active",
    ],
    "achieved_occupancy_pct": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    ],
    "dram_throughput_pct": [
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "shared_bank_conflicts": [
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    ],
    "stall_barrier_pct": [
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    ],
    "stall_memory_throttle_pct": [
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    ],
}

number_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_metric(text: str, names: list[str]):
    lines = text.splitlines()
    for metric_name in names:
        escaped = re.escape(metric_name)
        for line in lines:
            if re.search(escaped, line):
                nums = number_pattern.findall(line)
                if nums:
                    try:
                        return float(nums[-1])
                    except ValueError:
                        pass
    return None


def summarize_variant(row):
    status = row["status"]
    log_path = row["log_path"]
    metrics = {k: None for k in metric_aliases.keys()}
    if status in {"PASS", "FAIL"}:
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            text = ""
            status = "FAIL"
        if text:
            for key, aliases in metric_aliases.items():
                metrics[key] = parse_metric(text, aliases)
    return {
        "status": status,
        "ptx_file": row["ptx_file"],
        "log_path": log_path,
        "ncu_metrics": metrics,
    }

variants = {
    "value_only_baseline": {
        "status": "NOT RUN",
        "ptx_file": os.environ["VALUE_ONLY_PTX_FILE"],
        "log_path": "",
        "ncu_metrics": {k: None for k in metric_aliases.keys()},
    },
    "shadow_strict": {
        "status": "NOT RUN",
        "ptx_file": os.environ["SHADOW_STRICT_PTX_FILE"],
        "log_path": "",
        "ncu_metrics": {k: None for k in metric_aliases.keys()},
    },
    "shadow_fast": {
        "status": "NOT RUN",
        "ptx_file": os.environ["SHADOW_FAST_PTX_FILE"],
        "log_path": "",
        "ncu_metrics": {k: None for k in metric_aliases.keys()},
    },
}

if ncu_available:
    for row in variant_rows:
        variants[row["variant"]] = summarize_variant(row)

statuses = [v["status"] for v in variants.values()]
if not ncu_available:
    overall_status = "NOT RUN"
    reason = "ncu_not_available_on_remote"
elif any(s == "FAIL" for s in statuses):
    overall_status = "FAIL"
    reason = "one_or_more_variant_profile_failures"
elif any(s == "PASS" for s in statuses):
    overall_status = "PASS"
    reason = ""
else:
    overall_status = "NOT RUN"
    reason = "no_runnable_variants"

payload = {
    "schema": "sounio.benchmark.l4-ncu.v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "overall_status": overall_status,
    "reason": reason,
    "host": os.environ["GPU_HOST"],
    "gpu_user": os.environ["GPU_USER"],
    "remote_dir": os.environ["REMOTE_DIR"],
    "protocol": {
        "dim": int(os.environ["NCU_DIM"]),
        "iters": int(os.environ["NCU_ITERS"]),
        "metrics": os.environ["NCU_METRICS"].split(","),
    },
    "variants": variants,
}

with open(report_path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

echo "gpu_ncu_l4_collect: report=${REPORT_PATH} status=$(python3 - "$REPORT_PATH" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    print(json.load(f).get("overall_status", "NOT RUN"))
PY
)"

exit 0
