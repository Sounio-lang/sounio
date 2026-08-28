#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: moonshot_a_sinkhorn16_slurm_gate_common.sh <scalar|f32_epistemic> <out-dir>}"
OUT_DIR="${2:?usage: moonshot_a_sinkhorn16_slurm_gate_common.sh <scalar|f32_epistemic> <out-dir>}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p "$OUT_DIR"

case "$MODE" in
  scalar)
    SCHEMA="sounio.kretikos.sinkhorn16-slurm-gate.v1"
    OUT_JSON="$OUT_DIR/kretikos_kaxi_sinkhorn16_slurm_gate.v1.json"
    PTX_FLAGS=(--f32)
    PTX_NAME="sinkhorn16_f32.ptx"
    RUNNER_MODE=(--type f32)
    ;;
  f32_epistemic)
    SCHEMA="sounio.kretikos.sinkhorn16-epistemic-slurm-gate.v1"
    OUT_JSON="$OUT_DIR/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.v1.json"
    PTX_FLAGS=(--f32-epistemic)
    PTX_NAME="sinkhorn16_f32_epistemic.ptx"
    RUNNER_MODE=(--mode epistemic --type f32)
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

emit_not_run() {
  local reason="$1"
  python3 - "$OUT_JSON" "$SCHEMA" "$reason" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
schema = sys.argv[2]
reason = sys.argv[3]
semantic = {"status": "", "maxdu": "", "maxdv": ""}
if "epistemic" in schema:
    semantic["maxvar"] = ""
path.write_text(json.dumps({
    "schema": schema,
    "status": "fail",
    "job": {"id": "", "state": "", "comment": "", "exit_code": ""},
    "runtime": {"status": "", "reason": reason, "device": "", "compute_capability": ""},
    "semantic": semantic,
    "ptx": {"path": "", "bytes": 0, "sha256": ""},
    "boundaries": [
        "slurm_worker_is_cuda_runtime_authority",
        "runtime_acceptance_requires_cuda_driver_load_launch_and_copyback",
        "semantic_acceptance_requires_sinkhorn16_diagonal_uniform_oracle",
    ],
}, indent=2, sort_keys=True) + "\n")
print(f"moonshot_a_sinkhorn16_slurm_gate: SKIPPED mode={schema} reason={reason} artifact={path}")
PY
}

if [[ "${SOUNIO_MOONSHOT_A_RUN_SLURM:-0}" != "1" ]]; then
  emit_not_run "not_run_opt_in_required"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  emit_not_run "sbatch_missing"
  exit 0
fi

WAIT_TIMEOUT_SECONDS="${SOUNIO_MOONSHOT_A_SLURM_WAIT_TIMEOUT_SECONDS:-420}"
JOB_TIME="${SOUNIO_MOONSHOT_A_SLURM_JOB_TIME:-00:10:00}"
PARTITION="${SOUNIO_MOONSHOT_A_SLURM_PARTITION:-}"
if [[ -z "$PARTITION" ]]; then
  PARTITION="$(sinfo -h -o '%P|%G' 2>/dev/null | awk -F'|' '$2 ~ /gpu/ && $1 ~ /\*/ {gsub(/\*/, "", $1); print $1; exit}')"
fi
if [[ -z "$PARTITION" ]]; then
  PARTITION="$(sinfo -h -o '%P|%G' 2>/dev/null | awk -F'|' '$2 ~ /gpu/ && $1 ~ /gpu/ {gsub(/\*/, "", $1); print $1; exit}')"
fi
if [[ -z "$PARTITION" ]]; then
  PARTITION="$(sinfo -h -o '%P|%G' 2>/dev/null | awk -F'|' '$2 ~ /gpu/ {gsub(/\*/, "", $1); print $1; exit}')"
fi
PARTITION="${PARTITION:-gpu-orangefs}"
GRES="${SOUNIO_MOONSHOT_A_SLURM_GRES:-gpu:1}"
JOB_MEM="${SOUNIO_MOONSHOT_A_SLURM_MEM:-8192M}"
EMIT_TIMEOUT_SECONDS="${SOUNIO_MOONSHOT_A_PTX_EMIT_TIMEOUT_SECONDS:-300}"

WORKER="$OUT_DIR/worker_${MODE}.py"
JOB="$OUT_DIR/job_${MODE}.sh"
SUBMIT_LOG="$OUT_DIR/submit_${MODE}.out"
JOB_OUT="$OUT_DIR/job_${MODE}_%j.out"
JOB_ERR="$OUT_DIR/job_${MODE}_%j.err"
PTX_PATH="$OUT_DIR/$PTX_NAME"

if [[ -z "${SOUNIO_KRETIKOS_COMPILER:-}" ]]; then
  if [[ -x "$ROOT_DIR/bin/souc-linux-x86_64" ]]; then
    export SOUNIO_KRETIKOS_COMPILER="$ROOT_DIR/bin/souc-linux-x86_64"
  elif [[ -x "$ROOT_DIR/bin/souc" ]]; then
    export SOUNIO_KRETIKOS_COMPILER="$ROOT_DIR/bin/souc"
  fi
fi

if [[ "${SOUNIO_MOONSHOT_A_PREEMIT_PTX:-1}" == "1" ]]; then
  set +e
  timeout "$EMIT_TIMEOUT_SECONDS" ./bin/kretikos kaxi-emit-ptx sinkhorn16 "${PTX_FLAGS[@]}" -o "$PTX_PATH" >"$OUT_DIR/preemit_${MODE}.out" 2>"$OUT_DIR/preemit_${MODE}.err"
  preemit_rc=$?
  set -e
  if [[ "$preemit_rc" -ne 0 || ! -s "$PTX_PATH" ]]; then
    python3 - "$OUT_JSON" "$SCHEMA" "$MODE" "$preemit_rc" "$OUT_DIR/preemit_${MODE}.err" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
schema = sys.argv[2]
mode = sys.argv[3]
rc = sys.argv[4]
err_path = pathlib.Path(sys.argv[5])
reason = err_path.read_text(errors="replace")[-1000:] if err_path.exists() else f"preemit_failed_rc={rc}"
semantic = {"status": "", "maxdu": "", "maxdv": ""}
if mode == "f32_epistemic":
    semantic["maxvar"] = ""
path.write_text(json.dumps({
    "schema": schema,
    "status": "fail",
    "job": {"id": "", "state": "", "comment": "", "exit_code": ""},
    "runtime": {"status": "", "reason": reason, "device": "", "compute_capability": ""},
    "semantic": semantic,
    "ptx": {"path": "", "bytes": 0, "sha256": ""},
    "boundaries": ["ptx_preemit_failed_before_slurm_submission"],
}, indent=2, sort_keys=True) + "\n")
PY
    echo "moonshot_a_sinkhorn16_slurm_gate: PREEMIT_FAIL artifact=$OUT_JSON"
    exit 1
  fi
fi

cat >"$WORKER" <<'PY'
#!/usr/bin/env python3
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys

mode, schema, root, out_json, ptx_name, emit_timeout = sys.argv[1:7]
root = pathlib.Path(root)
out_json = pathlib.Path(out_json)
out_json.parent.mkdir(parents=True, exist_ok=True)
ptx = out_json.parent / ptx_name
runner = out_json.parent / "kaxi_ptx_runner"

def write_doc(status, reason, runtime=None, semantic=None, ptx_meta=None):
    runtime = runtime or {}
    semantic = semantic or {}
    ptx_meta = ptx_meta or {}
    doc = {
        "schema": schema,
        "status": status,
        "job": {
            "id": os.environ.get("SLURM_JOB_ID", ""),
            "state": os.environ.get("SLURM_JOB_STATE", ""),
            "comment": "",
            "exit_code": "",
        },
        "runtime": {
            "status": runtime.get("status", ""),
            "reason": runtime.get("reason", reason),
            "device": runtime.get("device", ""),
            "compute_capability": runtime.get("compute_capability", ""),
        },
        "semantic": semantic,
        "ptx": {
            "path": ptx_meta.get("path", ptx.name if ptx.exists() else ""),
            "bytes": ptx_meta.get("bytes", ptx.stat().st_size if ptx.exists() else 0),
            "sha256": ptx_meta.get("sha256", hashlib.sha256(ptx.read_bytes()).hexdigest() if ptx.exists() else ""),
        },
        "boundaries": [
            "slurm_worker_is_cuda_runtime_authority",
            "runtime_acceptance_requires_cuda_driver_load_launch_and_copyback",
            "semantic_acceptance_requires_sinkhorn16_diagonal_uniform_oracle",
            "ptx_hash_binds_current_worker_emission",
            "does_not_claim_full_abide_cohort_revalidation",
        ],
    }
    if mode == "f32_epistemic":
        doc["boundaries"].append("variance_acceptance_requires_finite_nonnegative_nonzero_output_slice")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

def run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=root, text=True, capture_output=True, **kwargs)

def runner_looks_executable(path):
    try:
        return path.is_file() and os.access(path, os.X_OK) and path.read_bytes()[:4] == b"\x7fELF"
    except OSError:
        return False

if not runner_looks_executable(runner):
    build = run(["cc", "-O2", "scripts/gpu/kaxi_ptx_runner.c", "-ldl", "-lm", "-o", str(runner)])
    if build.returncode != 0:
        write_doc("fail", "runner_build_failed", runtime={"reason": build.stderr[-1000:]})
        sys.exit(1)

if not ptx.exists() or ptx.stat().st_size == 0:
    flags = ["--f32"] if mode == "scalar" else ["--f32-epistemic"]
    emit = run(["timeout", emit_timeout, "./bin/kretikos", "kaxi-emit-ptx", "sinkhorn16", *flags, "-o", str(ptx)])
    if emit.returncode != 0:
        write_doc("fail", "ptx_emit_failed", runtime={"reason": (emit.stderr + emit.stdout)[-1000:]})
        sys.exit(1)

vals = []
for i in range(16):
    vals.append("-4.0")
for i in range(16):
    vals.append("-4.0")
for i in range(16):
    for j in range(16):
        vals.append("0.0" if i == j else "-1.0")
for i in range(32):
    vals.append("0.0")
init_mem = ",".join(vals)
init_var_value = os.environ.get("SOUNIO_MOONSHOT_A_EPI_INIT_VAR", "1e-12")
init_var = ",".join([init_var_value] * 320)

runner_cmd = [str(runner), str(ptx), "--threads", "1", "--mem-words", "320", "--init-mem", init_mem]
if mode == "scalar":
    runner_cmd += ["--type", "f32"]
else:
    runner_cmd += ["--mode", "epistemic", "--type", "f32", "--init-var", init_var]

launch = run(runner_cmd)
text = launch.stdout + "\n" + launch.stderr
status_match = re.search(r"sounio_kaxi_runtime status=([^ ]+) reason=([^ ]+) stage=([^ ]+) cuda_result=([0-9-]+)", text)
device_match = re.search(r"device=(.*) cc=([0-9]+\\.[0-9]+)", text)
runtime = {
    "status": status_match.group(1) if status_match else "",
    "reason": status_match.group(2) if status_match else "missing_runtime_status",
    "device": device_match.group(1).strip() if device_match else "",
    "compute_capability": device_match.group(2) if device_match else "",
}
if launch.returncode != 0 or runtime["status"] != "pass":
    write_doc("fail", runtime["reason"], runtime=runtime)
    sys.exit(1)

mem_line = next((line for line in launch.stdout.splitlines() if line.startswith("MEM:")), "")
mem = [float(x) for x in mem_line.split()[1:]]
if len(mem) < 320:
    write_doc("fail", "missing_mem_output", runtime=runtime)
    sys.exit(1)
u = mem[288:304]
v = mem[304:320]
maxdu = max(abs(x - (-7.0875)) for x in u)
maxdv = max(abs(x) for x in v)
semantic = {"status": "pass" if maxdu <= 1e-3 and maxdv <= 1e-3 else "fail", "maxdu": maxdu, "maxdv": maxdv}

if mode == "f32_epistemic":
    var_line = next((line for line in launch.stdout.splitlines() if line.startswith("VAR:")), "")
    var = [float(x) for x in var_line.split()[1:]]
    finite = all(math.isfinite(x) for x in var)
    nonnegative = all(x >= -1e-12 for x in var)
    maxvar = max(var) if var else float("nan")
    out_var = var[288:320]
    inf_indices = [i for i, x in enumerate(var) if math.isinf(x)]
    nan_indices = [i for i, x in enumerate(var) if math.isnan(x)]
    finite_values = [x for x in var if math.isfinite(x)]
    semantic["maxvar"] = maxvar
    semantic["output_maxvar"] = max(out_var) if out_var else float("nan")
    semantic["inf_count"] = len(inf_indices)
    semantic["nan_count"] = len(nan_indices)
    semantic["first_bad_indices"] = (inf_indices + nan_indices)[:16]
    semantic["finite_maxvar"] = max(finite_values) if finite_values else float("nan")
    if not (var and finite and nonnegative and maxvar > 0):
        semantic["status"] = "fail"

write_doc("pass" if semantic["status"] == "pass" else "fail", "launch_pass", runtime=runtime, semantic=semantic)
sys.exit(0 if semantic["status"] == "pass" else 1)
PY
chmod +x "$WORKER"

printf '%q ' "${RUNNER_MODE[@]}" >/dev/null
cat >"$JOB" <<SH
#!/usr/bin/env bash
#SBATCH --job-name=moonshot-a-${MODE}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${JOB_MEM}
#SBATCH --gres=${GRES}
#SBATCH --time=${JOB_TIME}
#SBATCH --output=${JOB_OUT}
#SBATCH --error=${JOB_ERR}
set -euo pipefail
cd "$ROOT_DIR"
if [[ -z "\${SOUNIO_KRETIKOS_COMPILER:-}" ]]; then
  if [[ -x "$ROOT_DIR/bin/souc-linux-x86_64" ]]; then
    export SOUNIO_KRETIKOS_COMPILER="$ROOT_DIR/bin/souc-linux-x86_64"
  elif [[ -x "$ROOT_DIR/bin/souc" ]]; then
    export SOUNIO_KRETIKOS_COMPILER="$ROOT_DIR/bin/souc"
  fi
fi
python3 "$WORKER" "$MODE" "$SCHEMA" "$ROOT_DIR" "$OUT_JSON" "$PTX_NAME" "$EMIT_TIMEOUT_SECONDS"
SH

set +e
job_id="$(sbatch --parsable "$JOB" >"$SUBMIT_LOG" 2>&1; rc=$?; cat "$SUBMIT_LOG"; exit "$rc")"
submit_rc=$?
set -e
job_id="$(printf '%s\n' "$job_id" | tail -n1 | cut -d';' -f1)"
if [[ "$submit_rc" -ne 0 || -z "$job_id" ]]; then
  emit_not_run "sbatch_submit_failed"
  exit 0
fi

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
state=""
reason=""
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  if [[ -f "$OUT_JSON" ]]; then
    break
  fi
  sq="$(squeue -h -j "$job_id" -o '%T|%R' 2>/dev/null || true)"
  if [[ -n "$sq" ]]; then
    state="${sq%%|*}"
    reason="${sq#*|}"
  else
    if command -v sacct >/dev/null 2>&1; then
      sacct_line="$(sacct -n -j "$job_id" -X -o State,ExitCode 2>/dev/null | awk 'NF {print $0; exit}' || true)"
      state="$(awk '{print $1}' <<<"$sacct_line")"
      reason="$(awk '{print $2}' <<<"$sacct_line")"
    fi
    [[ -n "$state" ]] && break
  fi
  sleep 5
done

if [[ ! -f "$OUT_JSON" ]]; then
  python3 - "$OUT_JSON" "$SCHEMA" "$job_id" "${state:-PENDING}" "${reason:-timeout_waiting_for_artifact}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
schema, job_id, state, reason = sys.argv[2:6]
semantic = {"status": "", "maxdu": "", "maxdv": ""}
if "epistemic" in schema:
    semantic["maxvar"] = ""
path.write_text(json.dumps({
    "schema": schema,
    "status": "fail",
    "job": {"id": job_id, "state": state, "comment": reason, "exit_code": ""},
    "runtime": {"status": "", "reason": reason, "device": "", "compute_capability": ""},
    "semantic": semantic,
    "ptx": {"path": "", "bytes": 0, "sha256": ""},
    "submit_log": "submit.out",
    "boundaries": [
        "slurm_worker_is_cuda_runtime_authority",
        "runtime_acceptance_requires_cuda_driver_load_launch_and_copyback",
        "semantic_acceptance_requires_sinkhorn16_diagonal_uniform_oracle",
    ],
}, indent=2, sort_keys=True) + "\n")
PY
fi

python3 - "$OUT_JSON" "$job_id" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
job_id = sys.argv[2]
doc = json.loads(path.read_text())
doc.setdefault("job", {})
doc["job"]["id"] = doc["job"].get("id") or job_id
path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
print(f"moonshot_a_sinkhorn16_slurm_gate: status={doc.get('status')} artifact={path}")
sys.exit(0 if doc.get("status") == "pass" else 1)
PY
