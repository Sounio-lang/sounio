#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

REGIME="${SOUNIO_SEMANTIC_ORC_REGIME:-normative}"
SUPPORT_INDEX="${SOUNIO_SEMANTIC_ORC_SUPPORT_INDEX:-0}"
SUPPORT_STRIDE="${SOUNIO_SEMANTIC_ORC_SUPPORT_STRIDE:-16}"
RUN_SLURM="${SOUNIO_SEMANTIC_ORC_RUN_SLURM:-0}"
WAIT_TIMEOUT_SECONDS="${SOUNIO_SEMANTIC_ORC_SLURM_WAIT_TIMEOUT_SECONDS:-1800}"
JOB_TIME="${SOUNIO_SEMANTIC_ORC_SLURM_JOB_TIME:-00:20:00}"
PARTITION="${SOUNIO_SEMANTIC_ORC_SLURM_PARTITION:-gpu-orangefs}"
GRES="${SOUNIO_SEMANTIC_ORC_SLURM_GRES:-gpu:1}"
JOB_MEM="${SOUNIO_SEMANTIC_ORC_SLURM_MEM:-8192M}"
NODELIST="${SOUNIO_SEMANTIC_ORC_SLURM_NODELIST:-}"
OUT_DIR="${SOUNIO_SEMANTIC_ORC_SLURM_DIR:-$ROOT_DIR/artifacts/semantic_orc/slurm/swow16-kaxi-${REGIME}-support${SUPPORT_INDEX}-$(date -u +%Y%m%dT%H%M%SZ)}"
JOB="$OUT_DIR/job.sh"
JOB_OUT="$OUT_DIR/job_%j.out"
JOB_ERR="$OUT_DIR/job_%j.err"
SUBMIT_LOG="$OUT_DIR/submit.out"
RUNTIME_DIR="$OUT_DIR/runtime"
RESULT="$RUNTIME_DIR/swow16_kaxi_runtime_result.json"
SLURM_RESULT="$OUT_DIR/swow16_kaxi_slurm_result.json"
PREBUILT_RUNNER="${SOUNIO_KAXI_RUNNER:-}"
PREBUILT_PTX="${SOUNIO_KAXI_PTX:-}"

mkdir -p "$OUT_DIR" "$RUNTIME_DIR"

write_not_run() {
  local status="$1"
  local reason="$2"
  python3 - "$SLURM_RESULT" "$status" "$reason" "$REGIME" "$SUPPORT_INDEX" "$SUPPORT_STRIDE" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
status, reason, regime, support_index, support_stride = sys.argv[2:7]
path.write_text(json.dumps({
    "schema": "sounio.semantic_orc.swow16_kaxi_slurm_result.v1",
    "status": status,
    "regime": regime,
    "support_index": int(support_index),
    "support_stride": int(support_stride),
    "reason": reason,
    "job": {"id": "", "state": "", "exit_code": ""},
    "runtime_result": "",
    "boundaries": [
        "slurm_gpu_runtime_required_for_gpu_claim",
        "no_gpu_claim_when_status_is_not_pass",
    ],
}, indent=2, sort_keys=True) + "\n")
PY
  echo "[semantic-orc-kaxi-slurm] ${status^^} ($reason) artifact=$SLURM_RESULT"
}

echo "[semantic-orc-kaxi-slurm] regime=$REGIME"
echo "[semantic-orc-kaxi-slurm] support_index=$SUPPORT_INDEX"
echo "[semantic-orc-kaxi-slurm] out=$OUT_DIR"

if [[ "$RUN_SLURM" != "1" ]]; then
  write_not_run "skipped" "not_run_opt_in_required"
  exit 0
fi

if ! command -v sbatch >/dev/null 2>&1; then
  write_not_run "skipped" "sbatch_missing"
  exit 0
fi

node_line=""
if [[ -n "$NODELIST" ]]; then
  node_line="#SBATCH --nodelist=${NODELIST}"
fi

cat >"$JOB" <<SH
#!/usr/bin/env bash
#SBATCH --job-name=swow16-kaxi-${REGIME}-s${SUPPORT_INDEX}
#SBATCH --partition=${PARTITION}
$node_line
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=${JOB_MEM}
#SBATCH --gres=${GRES}
#SBATCH --time=${JOB_TIME}
#SBATCH --output=${JOB_OUT}
#SBATCH --error=${JOB_ERR}
set -euo pipefail
cd "$ROOT_DIR"
export SOUNIO_SEMANTIC_ORC_REGIME="$REGIME"
export SOUNIO_SEMANTIC_ORC_SUPPORT_INDEX="$SUPPORT_INDEX"
export SOUNIO_SEMANTIC_ORC_SUPPORT_STRIDE="$SUPPORT_STRIDE"
export SOUNIO_SEMANTIC_ORC_KAXI_RUNTIME_DIR="$RUNTIME_DIR"
if [[ -n "$PREBUILT_RUNNER" ]]; then
  export SOUNIO_KAXI_RUNNER="$PREBUILT_RUNNER"
fi
if [[ -n "$PREBUILT_PTX" ]]; then
  export SOUNIO_KAXI_PTX="$PREBUILT_PTX"
fi
bash scripts/ci/semantic_orc_swow16_kaxi_runtime_gate.sh
SH

set +e
job_id="$(sbatch --parsable "$JOB" >"$SUBMIT_LOG" 2>&1; rc=$?; cat "$SUBMIT_LOG"; exit "$rc")"
submit_rc=$?
set -e
job_id="$(printf '%s\n' "$job_id" | tail -n1 | cut -d';' -f1)"
if [[ "$submit_rc" -ne 0 || -z "$job_id" ]]; then
  write_not_run "skipped" "sbatch_submit_failed"
  exit 0
fi

deadline=$(( $(date +%s) + WAIT_TIMEOUT_SECONDS ))
state=""
reason=""
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  if [[ -f "$RESULT" ]]; then
    break
  fi
  sq="$(squeue -h -j "$job_id" -o '%T|%R' 2>/dev/null || true)"
  if [[ -n "$sq" ]]; then
    state="${sq%%|*}"
    reason="${sq#*|}"
  else
    sc="$(scontrol show job "$job_id" 2>/dev/null || true)"
    if [[ -n "$sc" ]]; then
      state="$(sed -n 's/.*JobState=\([^ ]*\).*/\1/p' <<<"$sc" | head -n1)"
      reason="$(sed -n 's/.*Reason=\([^ ]*\).*/\1/p' <<<"$sc" | head -n1)"
      case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED)
          break
          ;;
      esac
    fi
    if command -v sacct >/dev/null 2>&1; then
      sacct_line="$(sacct -n -j "$job_id" -X -o State,ExitCode 2>/dev/null | awk 'NF {print $0; exit}' || true)"
      state="$(awk '{print $1}' <<<"$sacct_line")"
      reason="$(awk '{print $2}' <<<"$sacct_line")"
    fi
    [[ -n "$state" ]] && break
  fi
  sleep 5
done

if [[ ! -f "$RESULT" ]]; then
  python3 - "$SLURM_RESULT" "$REGIME" "$SUPPORT_INDEX" "$SUPPORT_STRIDE" "$job_id" "${state:-PENDING}" "${reason:-timeout_waiting_for_runtime_artifact}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
regime, support_index, support_stride, job_id, state, reason = sys.argv[2:8]
path.write_text(json.dumps({
    "schema": "sounio.semantic_orc.swow16_kaxi_slurm_result.v1",
    "status": "fail",
    "regime": regime,
    "support_index": int(support_index),
    "support_stride": int(support_stride),
    "reason": reason,
    "job": {"id": job_id, "state": state, "exit_code": ""},
    "runtime_result": "",
    "boundaries": [
        "slurm_gpu_runtime_required_for_gpu_claim",
        "runtime_artifact_missing",
    ],
}, indent=2, sort_keys=True) + "\n")
PY
  echo "[semantic-orc-kaxi-slurm] FAIL artifact=$SLURM_RESULT"
  exit 1
fi

python3 - "$SLURM_RESULT" "$RESULT" "$REGIME" "$SUPPORT_INDEX" "$SUPPORT_STRIDE" "$job_id" <<'PY'
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
runtime_path = pathlib.Path(sys.argv[2])
regime, support_index, support_stride, job_id = sys.argv[3:7]
runtime = json.loads(runtime_path.read_text())
status = "pass" if runtime.get("status") == "pass" else "fail"
doc = {
    "schema": "sounio.semantic_orc.swow16_kaxi_slurm_result.v1",
    "status": status,
    "regime": regime,
    "support_index": int(support_index),
    "support_stride": int(support_stride),
    "support_offset": runtime.get("support_offset", 0),
    "reason": runtime.get("reason", runtime.get("runtime", {}).get("reason", "")),
    "job": {"id": job_id, "state": "", "exit_code": ""},
    "runtime_result": str(runtime_path),
    "runtime": runtime,
    "boundaries": [
        "gpu_runtime_compared_to_pack_oracle_only",
        "entropic_regularized_transport_only",
        "clinical_claims_not_enforced",
        "no_convergence_theorem",
    ],
}
out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
print(f"[semantic-orc-kaxi-slurm] {status.upper()} artifact={out_path}")
raise SystemExit(0 if status == "pass" else 1)
PY
