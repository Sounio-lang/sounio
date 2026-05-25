#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_MOONSHOT_A_ABIDE_EPI_COHORT_SLURM_DIR:-$(mktemp -d /tmp/moonshot-a-abide-epi-cohort-slurm.XXXXXX)}"
OUT_JSON="$OUT_DIR/moonshot_a_abide_epistemic_cohort_slurm.v1.json"
mkdir -p "$OUT_DIR"

write_doc() {
  local status="$1"
  local reason="$2"
  python3 - "$OUT_JSON" "$status" "$reason" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status = sys.argv[2]
reason = sys.argv[3]
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "schema": "sounio.moonshot_a.abide_epistemic_cohort_slurm.v1",
    "status": status,
    "reason": reason,
    "job": {"id": "", "state": "", "exit_code": ""},
    "campaign": {"limit": "", "subjects_reported": 0, "report": ""},
    "runtime": {"status": "", "device": "", "compute_capability": ""},
    "variance": {
        "max_output_var": "",
        "inf_count": "",
        "nan_count": "",
        "negative_count": "",
    },
    "boundaries": [
        "slurm_worker_is_cuda_runtime_authority",
        "f32_epistemic_cohort_outputs_are_sensitivity_diagnostics_not_calibrated_coverage",
        "full_cohort_claim_requires_limit_zero_or_all_subjects_reported",
    ],
}, indent=2, sort_keys=True) + "\n")
print(f"moonshot_a_abide_epistemic_cohort_slurm_gate: {status} reason={reason} artifact={path}")
PY
}

if [[ "${SOUNIO_MOONSHOT_A_RUN_SLURM:-0}" != "1" ]]; then
  write_doc "fail" "not_run_opt_in_required"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  write_doc "fail" "sbatch_missing"
  exit 0
fi

LIMIT="${SOUNIO_MOONSHOT_A_ABIDE_COHORT_LIMIT:-2}"
SUBJECTS_FROM_REPORT="${SOUNIO_MOONSHOT_A_ABIDE_SUBJECTS_FROM_REPORT:-artifacts/research/abide_orc/cohort_summary.tsv}"
WAIT_TIMEOUT_SECONDS="${SOUNIO_MOONSHOT_A_SLURM_WAIT_TIMEOUT_SECONDS:-7200}"
JOB_TIME="${SOUNIO_MOONSHOT_A_SLURM_JOB_TIME:-02:00:00}"
PARTITION="${SOUNIO_MOONSHOT_A_SLURM_PARTITION:-}"
if [[ -z "$PARTITION" ]]; then
  PARTITION="$(sinfo -h -o '%P|%G' 2>/dev/null | awk -F'|' '$2 ~ /gpu/ && $1 ~ /\*/ {gsub(/\*/, "", $1); print $1; exit}')"
fi
PARTITION="${PARTITION:-gpu-orangefs}"
GRES="${SOUNIO_MOONSHOT_A_SLURM_GRES:-gpu:1}"
JOB_MEM="${SOUNIO_MOONSHOT_A_SLURM_MEM:-16384M}"
CPUS_PER_TASK="${SOUNIO_MOONSHOT_A_SLURM_CPUS_PER_TASK:-4}"
EMIT_TIMEOUT_SECONDS="${SOUNIO_MOONSHOT_A_PTX_EMIT_TIMEOUT_SECONDS:-300}"

PTX_PATH="$OUT_DIR/sinkhorn16_f32_epistemic.ptx"
RUNNER="$OUT_DIR/kaxi_ptx_runner"
RESULT_DIR="$OUT_DIR/abide_f32_epistemic_orc"
REPORT="$OUT_DIR/abide_f32_epistemic_cohort_summary.tsv"
WORKER="$OUT_DIR/worker_abide_epi_cohort.sh"
JOB="$OUT_DIR/job_abide_epi_cohort.sh"
SUBMIT_LOG="$OUT_DIR/submit_abide_epi_cohort.out"
JOB_OUT="$OUT_DIR/job_abide_epi_cohort_%j.out"
JOB_ERR="$OUT_DIR/job_abide_epi_cohort_%j.err"

if [[ "${SOUNIO_MOONSHOT_A_PREEMIT_PTX:-1}" == "1" ]]; then
  if [[ ! -s "$PTX_PATH" ]]; then
    set +e
    timeout "$EMIT_TIMEOUT_SECONDS" ./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32-epistemic -o "$PTX_PATH" \
      >"$OUT_DIR/preemit_abide_epi_cohort.out" \
      2>"$OUT_DIR/preemit_abide_epi_cohort.err"
    preemit_rc=$?
    set -e
    if [[ "$preemit_rc" -ne 0 || ! -s "$PTX_PATH" ]]; then
      write_doc "fail" "ptx_preemit_failed"
      exit 1
    fi
  fi
fi

cat >"$WORKER" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$1"
OUT_JSON="$2"
PTX_PATH="$3"
RUNNER="$4"
RESULT_DIR="$5"
REPORT="$6"
LIMIT="$7"
EMIT_TIMEOUT_SECONDS="$8"
cd "$ROOT_DIR"
mkdir -p "$(dirname "$OUT_JSON")" "$RESULT_DIR"
if [[ ! -s "$PTX_PATH" ]]; then
  timeout "$EMIT_TIMEOUT_SECONDS" ./bin/kretikos kaxi-emit-ptx sinkhorn16 --f32-epistemic -o "$PTX_PATH"
fi
runner_looks_elf() {
  python3 - "$1" <<'PY'
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and p.read_bytes()[:4] == b"\x7fELF" else 1)
PY
}
RUNNER_SEED="${SOUNIO_KAXI_RUNNER_SEED:-/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/phase/slurm_f32_epistemic_reuse/kaxi_ptx_runner}"
if [[ -r "$RUNNER_SEED" ]] && runner_looks_elf "$RUNNER_SEED"; then
  cp "$RUNNER_SEED" "$RUNNER"
  chmod +x "$RUNNER"
else
  cc -O2 scripts/gpu/kaxi_ptx_runner.c -ldl -lm -o "$RUNNER"
fi
if ! runner_looks_elf "$RUNNER"; then
  echo "kaxi_ptx_runner is not a valid ELF executable" >&2
  exit 126
fi
PY_RUNNER=(python3)
if ! python3 - <<'PY' >/dev/null 2>&1
import numpy
PY
then
  UV_BIN="${SOUNIO_UV_BIN:-}"
  if [[ -z "$UV_BIN" && -x "./bin/uv" ]]; then
    UV_BIN="./bin/uv"
  fi
  if [[ -z "$UV_BIN" ]]; then
    UV_BIN="$(command -v uv || true)"
  fi
  if [[ -z "$UV_BIN" ]]; then
    echo "numpy missing and uv unavailable; stage ./bin/uv or set SOUNIO_UV_BIN" >&2
    exit 127
  fi
  PY_RUNNER=("$UV_BIN" run --with numpy python)
fi
SUBJECTS_REPORT_PATH="${SOUNIO_MOONSHOT_A_ABIDE_SUBJECTS_FROM_REPORT:-artifacts/research/abide_orc/cohort_summary.tsv}"
args=(
  "${PY_RUNNER[@]}" scripts/research/abide_cohort_orc_sweep.py
  --mode f32_epistemic
  --ptx "$PTX_PATH"
  --runner "$RUNNER"
  --out-dir "$RESULT_DIR"
  --report "$REPORT"
  --progress-every 1
)
if [[ -n "$SUBJECTS_REPORT_PATH" && -r "$SUBJECTS_REPORT_PATH" ]]; then
  args+=(--subjects-from-report "$SUBJECTS_REPORT_PATH")
fi
if [[ "$LIMIT" != "0" ]]; then
  args+=(--limit "$LIMIT")
fi
"${args[@]}" >"$(dirname "$OUT_JSON")/abide_f32_epistemic_cohort.out" 2>"$(dirname "$OUT_JSON")/abide_f32_epistemic_cohort.err"
python3 - "$OUT_JSON" "$REPORT" "$LIMIT" "$PTX_PATH" "$SUBJECTS_REPORT_PATH" <<'PY'
import csv
import hashlib
import json
import math
import pathlib
import sys

out_json = pathlib.Path(sys.argv[1])
report = pathlib.Path(sys.argv[2])
limit = sys.argv[3]
ptx = pathlib.Path(sys.argv[4])
subjects_report = pathlib.Path(sys.argv[5]) if sys.argv[5] else None
rows = list(csv.DictReader(report.open(), delimiter="\t"))
expected_subjects = ""
if subjects_report and subjects_report.is_file():
    with subjects_report.open() as f:
        expected_subjects = sum(1 for line in f if line.strip()) - 1
def floats(key):
    out = []
    for row in rows:
        value = row.get(key, "")
        if value:
            out.append(float(value))
    return out
def ints(key):
    out = []
    for row in rows:
        value = row.get(key, "")
        if value:
            out.append(int(float(value)))
    return out
var_max = floats("var_output_max")
inf = ints("var_inf_count")
nan = ints("var_nan_count")
neg = ints("var_negative_count")
bad = sum(1 for row in rows if row.get("var_bad_first_index") not in ("", "-1"))
status = "pass"
reason = "cohort_f32_epistemic_runtime_pass"
if not rows:
    status = "fail"
    reason = "empty_report"
elif limit == "0" and expected_subjects != "" and len(rows) != expected_subjects:
    status = "fail"
    reason = "full_cohort_subject_count_mismatch"
elif bad or sum(inf) or sum(nan) or sum(neg):
    status = "fail"
    reason = "variance_diagnostics_nonfinite_or_negative"
doc = {
    "schema": "sounio.moonshot_a.abide_epistemic_cohort_slurm.v1",
    "status": status,
    "reason": reason,
    "job": {
        "id": "",
        "state": "",
        "exit_code": "",
    },
    "campaign": {
        "limit": limit,
        "subjects_reported": len(rows),
        "expected_subjects": expected_subjects,
        "subjects_from_report": str(subjects_report) if subjects_report else "",
        "report": str(report),
        "result_dir": str(report.parent / "abide_f32_epistemic_orc"),
    },
    "runtime": {
        "status": "pass",
        "device": "",
        "compute_capability": "",
    },
    "ptx": {
        "path": str(ptx),
        "bytes": ptx.stat().st_size,
        "sha256": hashlib.sha256(ptx.read_bytes()).hexdigest(),
    },
    "variance": {
        "max_output_var": max(var_max) if var_max else "",
        "inf_count": sum(inf),
        "nan_count": sum(nan),
        "negative_count": sum(neg),
    },
    "boundaries": [
        "slurm_worker_is_cuda_runtime_authority",
        "f32_epistemic_cohort_outputs_are_sensitivity_diagnostics_not_calibrated_coverage",
        "full_cohort_claim_requires_limit_zero_or_all_subjects_reported",
    ],
}
out_json.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
sys.exit(0 if status == "pass" else 1)
PY
SH
chmod +x "$WORKER"

cat >"$JOB" <<SH
#!/usr/bin/env bash
#SBATCH --job-name=moonshot-a-abide-epi
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${JOB_MEM}
#SBATCH --gres=${GRES}
#SBATCH --time=${JOB_TIME}
#SBATCH --output=${JOB_OUT}
#SBATCH --error=${JOB_ERR}
set -euo pipefail
"$WORKER" "$ROOT_DIR" "$OUT_JSON" "$PTX_PATH" "$RUNNER" "$RESULT_DIR" "$REPORT" "$LIMIT" "$EMIT_TIMEOUT_SECONDS"
SH

rm -f "$OUT_JSON"

set +e
job_id="$(sbatch --parsable "$JOB" >"$SUBMIT_LOG" 2>&1; rc=$?; cat "$SUBMIT_LOG"; exit "$rc")"
submit_rc=$?
set -e
job_id="$(printf '%s\n' "$job_id" | tail -n1 | cut -d';' -f1)"
if [[ "$submit_rc" -ne 0 || -z "$job_id" ]]; then
  write_doc "fail" "sbatch_submit_failed"
  exit 0
fi

if [[ "${SOUNIO_MOONSHOT_A_SUBMIT_ONLY:-0}" == "1" ]]; then
  python3 - "$OUT_JSON" "$job_id" "$LIMIT" "$REPORT" "$RESULT_DIR" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
job_id, limit, report, result_dir = sys.argv[2:6]
doc = {
    "schema": "sounio.moonshot_a.abide_epistemic_cohort_slurm.v1",
    "status": "queued",
    "reason": "submit_only_job_queued",
    "job": {"id": job_id, "state": "PENDING", "exit_code": ""},
    "campaign": {
        "limit": limit,
        "subjects_reported": 0,
        "report": report,
        "result_dir": result_dir,
    },
    "runtime": {"status": "", "device": "", "compute_capability": ""},
    "variance": {
        "max_output_var": "",
        "inf_count": "",
        "nan_count": "",
        "negative_count": "",
    },
    "boundaries": [
        "slurm_worker_is_cuda_runtime_authority",
        "f32_epistemic_cohort_outputs_are_sensitivity_diagnostics_not_calibrated_coverage",
        "full_cohort_claim_requires_limit_zero_or_all_subjects_reported",
        "submit_only_status_is_not_runtime_acceptance",
    ],
}
path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
print(
    "moonshot_a_abide_epistemic_cohort_slurm_gate: "
    f"queued job={job_id} artifact={path}"
)
PY
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
    sacct_line="$(sacct -n -j "$job_id" -X -o State,ExitCode 2>/dev/null | awk 'NF {print $0; exit}' || true)"
    state="$(awk '{print $1}' <<<"$sacct_line")"
    reason="$(awk '{print $2}' <<<"$sacct_line")"
    [[ -n "$state" ]] && break
  fi
  sleep 10
done

if [[ ! -f "$OUT_JSON" ]]; then
  write_doc "fail" "${state:-PENDING}:${reason:-timeout_waiting_for_artifact}"
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
print(
    "moonshot_a_abide_epistemic_cohort_slurm_gate: "
    f"status={doc.get('status')} subjects={doc.get('campaign', {}).get('subjects_reported')} "
    f"artifact={path}"
)
sys.exit(0 if doc.get("status") == "pass" else 1)
PY
