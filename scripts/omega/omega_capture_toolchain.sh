#!/usr/bin/env bash
# omega_capture_toolchain.sh — capture CUDA/GPU toolchain provenance as JSON.
#
# Why: GPU attestation/perf artifacts pin souc_version but NOT the CUDA toolchain
# (ptxas/nvcc/driver). ptxas codegen and cuBLAS baselines change across toolkit
# major versions, so any measured number is only comparable when the toolchain is
# recorded alongside it. See docs/audit/GPU_PIPELINE_SOTA_ASSESSMENT_2026-05-30.md.
#
# Usage:
#   omega_capture_toolchain.sh                 # capture on THIS host
#   omega_capture_toolchain.sh user@gpu-host   # capture on a remote GPU host (ssh)
#   SSH_OPTS="-o BatchMode=yes" omega_capture_toolchain.sh user@host
#
# Always emits a well-formed JSON object on stdout (never fails the caller):
#   capture_status = ok | partial | unavailable
# Fields absent on the probed host are "" (capture_status downgrades accordingly).
set -uo pipefail

TARGET="${1:-}"

# Self-contained POSIX probe: prints KEY<TAB>VALUE lines. Runs locally or via ssh.
read -r -d '' PROBE <<'PROBE_SH' || true
emit() { printf '%s\t%s\n' "$1" "$2"; }
emit host "$(hostname 2>/dev/null || echo unknown)"
if command -v ptxas >/dev/null 2>&1; then
  emit ptxas_version "$(ptxas --version 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')"
fi
if command -v nvcc >/dev/null 2>&1; then
  nvcc_raw="$(nvcc --version 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g')"
  emit nvcc_version "$nvcc_raw"
  emit cuda_release "$(printf '%s' "$nvcc_raw" | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p')"
fi
if [ -r /usr/local/cuda/version.json ]; then
  emit cuda_version_json "$(tr -d '\n' < /usr/local/cuda/version.json 2>/dev/null | sed 's/  */ /g')"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  emit driver_version "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  emit gpu_name "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  emit gpu_compute_cap "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
fi
PROBE_SH

if [[ -n "$TARGET" ]]; then
  # shellcheck disable=SC2086
  RAW="$(ssh ${SSH_OPTS:-} "$TARGET" "bash -s" <<<"$PROBE" 2>/dev/null || true)"
  CAPTURED_ON="$TARGET"
else
  RAW="$(bash -c "$PROBE" 2>/dev/null || true)"
  CAPTURED_ON="local"
fi

OMEGA_PROBE_RAW="$RAW" python3 - "$CAPTURED_ON" <<'PY'
import datetime, os, sys
captured_on = sys.argv[1]
raw = os.environ.get("OMEGA_PROBE_RAW", "")
fields = {}
for line in raw.splitlines():
    if "\t" in line:
        k, _, v = line.partition("\t")
        v = v.strip()
        if v:
            fields[k.strip()] = v

keys = ["cuda_release", "nvcc_version", "ptxas_version", "driver_version",
        "gpu_name", "gpu_compute_cap", "cuda_version_json", "host"]
have_core = any(fields.get(k) for k in ("ptxas_version", "nvcc_version", "driver_version"))
have_all = all(fields.get(k) for k in ("cuda_release", "ptxas_version", "driver_version"))
status = "ok" if have_all else ("partial" if have_core else "unavailable")

obj = {
    "captured_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat().replace("+00:00", "Z"),
    "captured_on": captured_on,
    "capture_status": status,
    "cuda_release": fields.get("cuda_release", ""),
    "nvcc_version": fields.get("nvcc_version", ""),
    "ptxas_version": fields.get("ptxas_version", ""),
    "driver_version": fields.get("driver_version", ""),
    "gpu_name": fields.get("gpu_name", ""),
    "gpu_compute_cap": fields.get("gpu_compute_cap", ""),
    "cuda_version_json": fields.get("cuda_version_json", ""),
    "probe_host": fields.get("host", ""),
}
import json
print(json.dumps(obj, indent=2))
PY
