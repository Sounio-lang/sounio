#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

OUT_DIR="${SOUNIO_NATIVE_V2_METAL_ALGEBRA_DIR:-$(mktemp -d /tmp/sounio-native-v2-metal-algebra.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
RESULTS_TSV="$ARTIFACT_DIR/results.tsv"
SUMMARY_JSON="$ARTIFACT_DIR/native_v2_metal_algebra.v1.json"
MANIFEST="${SOUNIO_NATIVE_V2_METAL_ALGEBRA_MANIFEST:-tests/gpu/metal_first/manifest.tsv}"
ORACLE_OSSM="tests/run-pass/gpu_epistemic_f64_ossm_parity.sio"
ORACLE_SED="tests/run-pass/gpu_epistemic_f64_sedenion_parity.sio"
LOCAL_METAL_RUNNER="scripts/gpu-metal-validation/run_algebra_metal.sh"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

append_header() {
  cat >"$RESULTS_TSV" <<'EOF'
case_id	lane	path	status	detail
EOF
}

append_row() {
  local case_id="$1"
  local lane="$2"
  local path="$3"
  local status="$4"
  local detail="$5"
  printf '%s\t%s\t%s\t%s\t%s\n' "$case_id" "$lane" "$path" "$status" "$detail" >>"$RESULTS_TSV"
}

echo "[native-v2-metal-algebra] out=$OUT_DIR"
sounio_require_souc

append_header
pass_count=0
fail_count=0
not_run_count=0

run_oracle() {
  local case_id="$1"
  local program="$2"
  local expected="$3"
  local log="$LOG_DIR/$case_id.oracle.log"
  if [[ ! -f "$program" ]]; then
    append_row "$case_id" "cpu_oracle" "$program" "fail" "missing_oracle"
    fail_count=$((fail_count + 1))
    return 0
  fi
  if "$SOUC_BIN" run "$program" >"$log" 2>&1 && grep -qF "$expected" "$log"; then
    append_row "$case_id" "cpu_oracle" "$program" "pass" "oracle_stdout_ok"
    pass_count=$((pass_count + 1))
  else
    append_row "$case_id" "cpu_oracle" "$program" "fail" "oracle_run_failed"
    fail_count=$((fail_count + 1))
  fi
}

run_oracle "ossm_cpu_f64_oracle" "$ORACLE_OSSM" "PASS: GPU epistemic f64 O-SSM parity"
run_oracle "sedenion_cpu_f64_oracle" "$ORACLE_SED" "PASS: GPU epistemic f64 S-SSM parity"

set +e
python3 - "$MANIFEST" "$RESULTS_TSV" <<'PY'
import pathlib
import sys

manifest = pathlib.Path(sys.argv[1])
results = pathlib.Path(sys.argv[2])

def append(case_id, lane, path, status, detail):
    with results.open("a", encoding="utf-8") as f:
        f.write(f"{case_id}\t{lane}\t{path}\t{status}\t{detail}\n")

if not manifest.exists():
    append("metal_manifest", "structural", str(manifest), "fail", "missing_manifest")
    raise SystemExit(1)

fail = 0
with manifest.open("r", encoding="utf-8") as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        row = dict(zip(header, line.split("\t")))
        case_id = row["case_id"]
        path = pathlib.Path(row["kernel_path"])
        kernel = row["kernel_name"]
        policy = row["policy"]
        if not path.exists():
            append(case_id, "structural_msl", str(path), "fail", "missing_msl")
            fail += 1
            continue
        text = path.read_text(encoding="utf-8")
        checks = [
            ("kernel_name", f"kernel void {kernel}" in text),
            ("policy", f"sounio.metal_f64_policy={policy}" in text),
            ("source_f64", "sounio.source_semantics=f64" in text),
            ("thread_id", "thread_position_in_grid" in text),
            ("device_float", "device float*" in text),
            ("no_double", "double" not in text),
            ("no_device_double", "device double*" not in text),
        ]
        bad = [name for name, ok in checks if not ok]
        if bad:
            append(case_id, "structural_msl", str(path), "fail", "missing_" + ",".join(bad))
            fail += 1
        else:
            append(case_id, "structural_msl", str(path), "pass", "metal_f64_policy_structural_ok")

raise SystemExit(1 if fail else 0)
PY
structural_rc=$?
set -e
if [[ "$structural_rc" -eq 0 ]]; then
  structural_passes="$(tail -n +2 "$RESULTS_TSV" | awk -F'\t' '$2=="structural_msl" && $4=="pass"{c++} END{print c+0}')"
  pass_count=$((pass_count + structural_passes))
else
  structural_fails="$(tail -n +2 "$RESULTS_TSV" | awk -F'\t' '$2=="structural_msl" && $4=="fail"{c++} END{print c+0}')"
  fail_count=$((fail_count + structural_fails))
fi

runtime_detail=""
if [[ "$fail_count" -eq 0 ]]; then
  runtime_log="$LOG_DIR/apple_metal_runtime.log"
  if command -v xcrun >/dev/null 2>&1 && command -v swiftc >/dev/null 2>&1; then
    if bash "$LOCAL_METAL_RUNNER" >"$runtime_log" 2>&1; then
      append_row "apple_metal_algebra_runtime" "metal_runtime" "$LOCAL_METAL_RUNNER" "pass" "local_metal_runtime_ok"
      pass_count=$((pass_count + 1))
    else
      append_row "apple_metal_algebra_runtime" "metal_runtime" "$LOCAL_METAL_RUNNER" "fail" "local_metal_runtime_failed"
      fail_count=$((fail_count + 1))
    fi
  else
    runtime_detail="xcrun_or_swiftc_unavailable"
    append_row "apple_metal_algebra_runtime" "metal_runtime" "$LOCAL_METAL_RUNNER" "not_run" "$runtime_detail"
    not_run_count=$((not_run_count + 1))
  fi
else
  runtime_detail="skipped_after_structural_or_oracle_failure"
  append_row "apple_metal_algebra_runtime" "metal_runtime" "$LOCAL_METAL_RUNNER" "not_run" "$runtime_detail"
  not_run_count=$((not_run_count + 1))
fi

python3 - "$SUMMARY_JSON" "$RESULTS_TSV" "$MANIFEST" "$SOUC_BIN" "$pass_count" "$not_run_count" "$fail_count" "$OUT_DIR" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

summary_path = pathlib.Path(sys.argv[1])
results_path = pathlib.Path(sys.argv[2])
manifest_path = pathlib.Path(sys.argv[3])
souc_bin = sys.argv[4]
pass_count = int(sys.argv[5])
not_run_count = int(sys.argv[6])
fail_count = int(sys.argv[7])
out_dir = pathlib.Path(sys.argv[8])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

with results_path.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

status = "pass"
if fail_count:
    status = "fail"
elif not_run_count:
    status = "partial"

payload = {
    "schema": "sounio.native_v2_metal_algebra.v1",
    "status": status,
    "target": "apple-metal-algebra",
    "fallback_path": "none",
    "host_callback": "none",
    "souc_bin": souc_bin,
    "manifest": str(manifest_path),
    "manifest_sha256": sha256(manifest_path) if manifest_path.exists() else "",
    "pass_count": pass_count,
    "not_run_count": not_run_count,
    "fail_count": fail_count,
    "results_tsv": str(results_path),
    "artifact_dir": str(out_dir),
    "cases": rows,
    "remaining_boundaries": [
        "Apple Metal runtime requires xcrun, swiftc, and an Apple GPU host" if not_run_count else "",
        "Metal f64 is represented by explicit Sounio lowering policy, not native MSL double",
        "float2 compensated f64 lowering is named but not yet the promoted runtime policy",
        "this gate does not prove tensor-core performance, ROCm, WebGPU, CUDA PTX parity, or DDC",
    ],
}
payload["remaining_boundaries"] = [x for x in payload["remaining_boundaries"] if x]
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if fail_count:
    raise SystemExit(1)
PY

summary_status="$(python3 - "$SUMMARY_JSON" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)"

echo "[native-v2-metal-algebra] status=$summary_status summary=$SUMMARY_JSON"
if [[ "$summary_status" == "fail" ]]; then
  exit 1
fi
