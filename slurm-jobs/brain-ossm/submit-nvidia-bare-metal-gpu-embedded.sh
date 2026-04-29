#!/usr/bin/env bash
set -euo pipefail

# Submit the NVIDIA bare CUBIN gate without staging a repo or payload tree on
# OrangeFS. The payload is embedded into the sbatch script and decoded on the
# GPU worker's local scratch storage.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-$ROOT_DIR}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
RUN_ID="${RUN_ID:-nvidia-bare-embedded-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
LOCAL_TARBALL="${LOCAL_TARBALL:-/tmp/${RUN_ID}.tgz}"
LOCAL_TARBALL_TMP="${LOCAL_TARBALL}.tmp"
LOCAL_DRIVER_LOADER="${LOCAL_DRIVER_LOADER:-/tmp/${RUN_ID}.driver_loader}"
DRIVER_LOADER_PAYLOAD_NAME="$(basename "${LOCAL_DRIVER_LOADER}")"
LOCAL_SBATCH="/tmp/${RUN_ID}.sbatch"
LOCAL_SBATCH_B64="${LOCAL_SBATCH}.b64"
LOCAL_SBATCH_CHUNKS="${LOCAL_SBATCH}.chunks"
REMOTE_SBATCH="/tmp/${RUN_ID}.sbatch"
REMOTE_SBATCH_B64="${REMOTE_SBATCH}.b64"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"
SBATCH_OUTPUT="${SBATCH_OUTPUT:-/dev/null}"
JOB_MEM="${JOB_MEM:-4G}"
JOB_TIME="${JOB_TIME:-00:10:00}"
RUNTIME_RUNG="${SOUNIO_NVIDIA_BARE_RUNTIME_RUNG:-admission}"
MATRIX_MODE="${SOUNIO_NVIDIA_BARE_MATRIX:-0}"
MATRIX_RUNGS="${SOUNIO_NVIDIA_BARE_MATRIX_RUNGS:-epistemic_dual_output_f32 epistemic_dual_output_f32_l4cc_alias epistemic_dual_output_f32_sm89_l4 epistemic_dual_output_f32_sm89_l4_layout epistemic_dual_output_f32_sm89_l4_flags}"
case "${RUNTIME_RUNG}" in
  epistemic_dual_output_f32_sm89_l4_flags)
    RESULT_STEM="sounio_bare_epistemic_dual_output_f32_sm89_l4_flags"
    ;;
  epistemic_dual_output_f32_sm89_l4_layout)
    RESULT_STEM="sounio_bare_epistemic_dual_output_f32_sm89_l4_layout"
    ;;
  epistemic_dual_output_f32_l4cc_alias)
    RESULT_STEM="sounio_bare_epistemic_dual_output_f32_l4cc"
    ;;
  epistemic_dual_output_f32_sm89_l4)
    RESULT_STEM="sounio_bare_epistemic_dual_output_f32_sm89_l4"
    ;;
  epistemic_dual_output_f32)
    RESULT_STEM="sounio_bare_epistemic_dual_output_f32_sm80"
    ;;
  vec_add_f32|epistemic_elementwise_f32|epistemic_dual_lane_f32)
    RESULT_STEM="sounio_bare_vec_add_f32_sm80"
    ;;
  *)
    RESULT_STEM="sounio_bare_store_u32_const_sm80"
    ;;
esac
VEC_N="${SOUNIO_NVIDIA_BARE_VEC_N:-64}"
WAIT_FOR_RESULT="${WAIT_FOR_RESULT:-1}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-300}"

REQUIRED_FILES=(
  "bin/souc"
  "bin/souc-linux-x86_64"
  "scripts/lib/resolve_souc.sh"
  "scripts/ci/native_v2_nvidia_bare_metal_gate.sh"
  "scripts/gpu/nvidia_bare_cubin_cartographer.py"
  "scripts/gpu/nvidia_bare_driver_loader.c"
  "self-hosted/gpu/nvidia_bare.sio"
)

for rel in "${REQUIRED_FILES[@]}"; do
  if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
    echo "missing required file at ${SOUNIO_DIR}/${rel}" >&2
    exit 1
  fi
done

if [[ -n "${LOGIN_POD_NAME}" ]]; then
  LOGIN_POD="${LOGIN_POD_NAME}"
else
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods \
    -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi

if [[ -z "${LOGIN_POD}" ]]; then
  echo "could not resolve a live login pod in namespace ${NS}" >&2
  exit 1
fi

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc '
  set -euo pipefail
  test -S /run/slurm/sack.socket
  scontrol ping >/dev/null
'

echo "[1/4] building tiny NVIDIA bare payload"
rm -f "${LOCAL_TARBALL}" "${LOCAL_TARBALL_TMP}"
cc -O2 "${SOUNIO_DIR}/scripts/gpu/nvidia_bare_driver_loader.c" -ldl -o "${LOCAL_DRIVER_LOADER}"
tar -C "${SOUNIO_DIR}" -czf "${LOCAL_TARBALL_TMP}" \
  bin/souc \
  bin/souc-linux-x86_64 \
  scripts/lib/resolve_souc.sh \
  scripts/ci/native_v2_nvidia_bare_metal_gate.sh \
  scripts/gpu/nvidia_bare_cubin_cartographer.py \
  scripts/gpu/nvidia_bare_driver_loader.c \
  self-hosted/gpu/nvidia_bare.sio \
  artifacts/sass/cubin \
  artifacts/sass/omega \
  -C "$(dirname "${LOCAL_DRIVER_LOADER}")" "${DRIVER_LOADER_PAYLOAD_NAME}"
tar -tzf "${LOCAL_TARBALL_TMP}" >/dev/null
mv -f "${LOCAL_TARBALL_TMP}" "${LOCAL_TARBALL}"

PAYLOAD_B64="$(base64 -w 0 "${LOCAL_TARBALL}" 2>/dev/null || base64 "${LOCAL_TARBALL}" | tr -d '\n')"

cat > "${LOCAL_SBATCH}" <<EOF
#!/usr/bin/env bash
#SBATCH -J nvidia-bare
#SBATCH -p gpu-orangefs
#SBATCH -A plruntime
#SBATCH --qos=gpuorangefs
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
#SBATCH -w ${SBATCH_NODELIST}
#SBATCH -o ${SBATCH_OUTPUT}
set -euo pipefail

LOCAL_ROOT="\${TMPDIR:-/tmp}/${RUN_ID}-\${SLURM_JOB_ID}"
SOUNIO_DIR="\${LOCAL_ROOT}/repo"
RESULTS_DIR="\${LOCAL_ROOT}/results"
PAYLOAD_TGZ="\${LOCAL_ROOT}/payload.tgz"

rm -rf "\${LOCAL_ROOT}"
mkdir -p "\${SOUNIO_DIR}" "\${RESULTS_DIR}"

mark() {
  local msg="\$1"
  echo "\${msg}"
  if [[ -n "\${SLURM_JOB_ID:-}" ]]; then
    scontrol update "JobId=\${SLURM_JOB_ID}" "Comment=\${msg}" >/dev/null 2>&1 || true
  fi
}

trap 'rc=\$?; mark "nvidia_bare_phase=failed rc=\${rc} line=\${LINENO}"; exit "\${rc}"' ERR

mark "nvidia_bare_phase=decode_payload"
cat > "\${LOCAL_ROOT}/payload.tgz.b64" <<'PAYLOAD_B64_EOF'
${PAYLOAD_B64}
PAYLOAD_B64_EOF
base64 -d "\${LOCAL_ROOT}/payload.tgz.b64" > "\${PAYLOAD_TGZ}"
mark "nvidia_bare_phase=extract_payload"
tar -xzf "\${PAYLOAD_TGZ}" -C "\${SOUNIO_DIR}"
chmod +x "\${SOUNIO_DIR}/bin/souc" "\${SOUNIO_DIR}/bin/souc-linux-x86_64" "\${SOUNIO_DIR}/${DRIVER_LOADER_PAYLOAD_NAME}" "\${SOUNIO_DIR}/scripts/ci/native_v2_nvidia_bare_metal_gate.sh"

mark "nvidia_bare_phase=cuda_visibility"
echo "Sounio NVIDIA bare embedded runtime job"
echo "Host: \$(hostname)"
echo "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
ldconfig -p 2>/dev/null | grep -E 'libcuda\\.so|libnvidia-ml\\.so' || true

result_stem_for_rung() {
  case "\$1" in
    epistemic_dual_output_f32_sm89_l4_flags)
      printf '%s\\n' "sounio_bare_epistemic_dual_output_f32_sm89_l4_flags"
      ;;
    epistemic_dual_output_f32_sm89_l4_layout)
      printf '%s\\n' "sounio_bare_epistemic_dual_output_f32_sm89_l4_layout"
      ;;
    epistemic_dual_output_f32_l4cc_alias)
      printf '%s\\n' "sounio_bare_epistemic_dual_output_f32_l4cc"
      ;;
    epistemic_dual_output_f32_sm89_l4)
      printf '%s\\n' "sounio_bare_epistemic_dual_output_f32_sm89_l4"
      ;;
    epistemic_dual_output_f32)
      printf '%s\\n' "sounio_bare_epistemic_dual_output_f32_sm80"
      ;;
    vec_add_f32|epistemic_elementwise_f32|epistemic_dual_lane_f32)
      printf '%s\\n' "sounio_bare_vec_add_f32_sm80"
      ;;
    *)
      printf '%s\\n' "sounio_bare_store_u32_const_sm80"
      ;;
  esac
}

if [[ '${MATRIX_MODE}' == "1" || '${MATRIX_MODE}' == "true" || '${MATRIX_MODE}' == "TRUE" || '${MATRIX_MODE}' == "yes" || '${MATRIX_MODE}' == "on" ]]; then
  mark "nvidia_bare_phase=run_matrix"
  MATRIX_DIR="\${RESULTS_DIR}/matrix"
  SUMMARY_TSV="\${RESULTS_DIR}/nvidia_bare_admission_matrix.tsv"
  SUMMARY_JSONL="\${RESULTS_DIR}/nvidia_bare_admission_matrix.jsonl"
  mkdir -p "\${MATRIX_DIR}"
  printf 'rung\\tkernel\\tartifact_sha256\\tstructural_status\\tstructural_reason\\truntime_status\\truntime_reason\\truntime_stage\\tcuda_result\\tdual_n\\tvalue_err\\tuncertainty_err\\tdevice\\n' > "\${SUMMARY_TSV}"
  : > "\${SUMMARY_JSONL}"
  matrix_rc=0
  matrix_comment=""
  for matrix_rung in ${MATRIX_RUNGS}; do
    matrix_stem="\$(result_stem_for_rung "\${matrix_rung}")"
    rung_dir="\${MATRIX_DIR}/\${matrix_rung}"
    mkdir -p "\${rung_dir}"
    mark "nvidia_bare_phase=matrix_rung rung=\${matrix_rung}"
    trap - ERR
    set +e
    SOUNIO_NVIDIA_BARE_RUNTIME=1 \\
    SOUNIO_NVIDIA_BARE_RUNTIME_RUNG="\${matrix_rung}" \\
    SOUNIO_NVIDIA_BARE_VEC_N='${VEC_N}' \\
    SOUNIO_NVIDIA_BARE_RUNTIME_LOADER="\${SOUNIO_DIR}/${DRIVER_LOADER_PAYLOAD_NAME}" \\
    SOUNIO_NVIDIA_BARE_GATE_JSON="\${rung_dir}/native_v2_nvidia_bare_metal_gate.v1.json" \\
    SOUNIO_NVIDIA_BARE_CUBIN="\${rung_dir}/\${matrix_stem}.cubin" \\
    SOUNIO_NVIDIA_BARE_HEX="\${rung_dir}/\${matrix_stem}.hex" \\
    SOUNIO_NVIDIA_BARE_CARTOGRAPHY_JSON="\${rung_dir}/nvidia_bare_cubin_cartography.v1.json" \\
    SOUNIO_NVIDIA_BARE_GATE_LOG="\${rung_dir}/native_v2_nvidia_bare_metal_gate.log" \\
      bash "\${SOUNIO_DIR}/scripts/ci/native_v2_nvidia_bare_metal_gate.sh" \\
      > "\${rung_dir}/gate.stdout" 2>&1
    rung_rc="\$?"
    set -e
    trap 'rc=\$?; mark "nvidia_bare_phase=failed rc=\${rc} line=\${LINENO}"; exit "\${rc}"' ERR

    export MATRIX_RUNG="\${matrix_rung}"
    export MATRIX_RC="\${rung_rc}"
    export RUNG_DIR="\${rung_dir}"
    export SUMMARY_TSV
    export SUMMARY_JSONL
    classification="\$(python3 - <<'PY'
import json
import os
from pathlib import Path

rung = os.environ["MATRIX_RUNG"]
rung_dir = Path(os.environ["RUNG_DIR"])
p = rung_dir / "native_v2_nvidia_bare_metal_gate.v1.json"
obj = json.loads(p.read_text()) if p.exists() else {}
runtime = obj.get("runtime", {})
structural = obj.get("structural", {})
probe = runtime.get("cuda_probe", {})
short = {
    "epistemic_dual_output_f32": "control",
    "epistemic_dual_output_f32_l4cc_alias": "alias",
    "epistemic_dual_output_f32_sm89_l4": "squeezed",
    "epistemic_dual_output_f32_sm89_l4_layout": "layout",
    "epistemic_dual_output_f32_sm89_l4_flags": "flags",
}.get(rung, rung)
row = [
    rung,
    obj.get("kernel"),
    structural.get("sha256"),
    structural.get("status"),
    structural.get("reason"),
    runtime.get("status"),
    runtime.get("reason"),
    runtime.get("stage"),
    runtime.get("cuda_result"),
    runtime.get("dual_output_runtime_n"),
    runtime.get("dual_output_runtime_value_max_abs_err"),
    runtime.get("dual_output_runtime_uncertainty_max_abs_err"),
    probe.get("device_name"),
]
with open(os.environ["SUMMARY_TSV"], "a", encoding="utf-8") as f:
    f.write("\\t".join("" if value is None else str(value) for value in row) + "\\n")
with open(os.environ["SUMMARY_JSONL"], "a", encoding="utf-8") as f:
    f.write(json.dumps({"rung": rung, "rc": int(os.environ["MATRIX_RC"]), "gate": obj}, sort_keys=True) + "\\n")
print(f"{short}:{obj.get('status')}/{runtime.get('status')}/{runtime.get('reason')}/cuda{runtime.get('cuda_result')}")
PY
)"
    echo "\${classification}"
    if [[ -n "\${matrix_comment}" ]]; then
      matrix_comment="\${matrix_comment},\${classification}"
    else
      matrix_comment="\${classification}"
    fi
    if [[ "\${matrix_rung}" == "epistemic_dual_output_f32" && "\${classification}" != control:pass/pass/* ]]; then
      matrix_rc=1
    fi
  done
  mark "nvidia_bare_phase=matrix_done matrix_rc=\${matrix_rc} matrix=\${matrix_comment}"
  exit "\${matrix_rc}"
fi

mark "nvidia_bare_phase=run_gate"
trap - ERR
set +e
SOUNIO_NVIDIA_BARE_RUNTIME=1 \\
SOUNIO_NVIDIA_BARE_RUNTIME_RUNG='${RUNTIME_RUNG}' \\
SOUNIO_NVIDIA_BARE_VEC_N='${VEC_N}' \\
SOUNIO_NVIDIA_BARE_RUNTIME_LOADER="\${SOUNIO_DIR}/${DRIVER_LOADER_PAYLOAD_NAME}" \\
SOUNIO_NVIDIA_BARE_GATE_JSON="\${RESULTS_DIR}/native_v2_nvidia_bare_metal_gate.v1.json" \\
SOUNIO_NVIDIA_BARE_CUBIN="\${RESULTS_DIR}/${RESULT_STEM}.cubin" \\
SOUNIO_NVIDIA_BARE_HEX="\${RESULTS_DIR}/${RESULT_STEM}.hex" \\
SOUNIO_NVIDIA_BARE_CARTOGRAPHY_JSON="\${RESULTS_DIR}/nvidia_bare_cubin_cartography.v1.json" \\
SOUNIO_NVIDIA_BARE_GATE_LOG="\${RESULTS_DIR}/native_v2_nvidia_bare_metal_gate.log" \\
  bash "\${SOUNIO_DIR}/scripts/ci/native_v2_nvidia_bare_metal_gate.sh" \\
  > "\${RESULTS_DIR}/gate.stdout" 2>&1
gate_rc="\$?"
set -e
trap 'rc=\$?; mark "nvidia_bare_phase=failed rc=\${rc} line=\${LINENO}"; exit "\${rc}"' ERR

export RESULTS_DIR
export gate_rc
python3 - <<'PY'
import json
import os
import subprocess
from pathlib import Path

p = Path(os.environ["RESULTS_DIR"]) / "native_v2_nvidia_bare_metal_gate.v1.json"
gate_rc = os.environ.get("gate_rc", "unknown")
if p.exists():
    obj = json.loads(p.read_text())
else:
    obj = {}
runtime = obj.get("runtime", {})
def first_present(*keys):
    for key in keys:
        value = runtime.get(key)
        if value is not None:
            return value
    return None
parts = [
    "nvidia_bare_phase=gate_done",
    "gate_rc=" + str(gate_rc),
    "nvidia_bare_status=" + str(obj.get("status")),
    "nvidia_bare_reason=" + str(obj.get("reason")),
    "runtime_status=" + str(runtime.get("status")),
    "runtime_reason=" + str(runtime.get("reason")),
    "runtime_stage=" + str(runtime.get("stage")),
    "cuda_result=" + str(runtime.get("cuda_result")),
    "dual_n=" + str(first_present("dual_output_runtime_n", "epistemic_dual_output_f32_sm89_l4_n", "epistemic_dual_output_f32_n")),
    "value_err=" + str(first_present("dual_output_runtime_value_max_abs_err", "epistemic_dual_output_sm89_l4_value_max_abs_err", "epistemic_dual_output_value_max_abs_err")),
    "uncert_err=" + str(first_present("dual_output_runtime_uncertainty_max_abs_err", "epistemic_dual_output_sm89_l4_uncertainty_max_abs_err", "epistemic_dual_output_uncertainty_max_abs_err")),
]
comment = " ".join(parts)
print(comment)
job_id = os.environ.get("SLURM_JOB_ID")
if job_id:
    subprocess.run(["scontrol", "update", f"JobId={job_id}", "Comment=" + comment], check=False)
PY
exit "\${gate_rc}"
EOF

echo "[2/4] uploading embedded sbatch via no-stdin base64 chunks"
rm -rf "${LOCAL_SBATCH_CHUNKS}"
mkdir -p "${LOCAL_SBATCH_CHUNKS}"
base64 "${LOCAL_SBATCH}" > "${LOCAL_SBATCH_B64}"
split -b 24000 "${LOCAL_SBATCH_B64}" "${LOCAL_SBATCH_CHUNKS}/chunk."
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sh -lc "rm -f '${REMOTE_SBATCH}' '${REMOTE_SBATCH_B64}'"
for chunk in "${LOCAL_SBATCH_CHUNKS}"/chunk.*; do
  chunk_text="$(tr -d '\n' < "${chunk}")"
  "${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sh -lc "printf '%s' '${chunk_text}' >> '${REMOTE_SBATCH_B64}'"
done
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sh -lc "base64 -d '${REMOTE_SBATCH_B64}' > '${REMOTE_SBATCH}'; rm -f '${REMOTE_SBATCH_B64}'"
rm -rf "${LOCAL_SBATCH_CHUNKS}" "${LOCAL_SBATCH_B64}"

echo "[3/4] submitting to Slurm"
SBATCH_TEXT="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "sbatch '${REMOTE_SBATCH}'; rm -f '${REMOTE_SBATCH}'")"
echo "${SBATCH_TEXT}"
JOB_ID="$(printf '%s\n' "${SBATCH_TEXT}" | awk '/Submitted batch job/ {print $4; exit}')"
if [[ -z "${JOB_ID}" ]]; then
  echo "failed to parse job id from sbatch output" >&2
  exit 1
fi

echo "[4/4] submitted job ${JOB_ID}"
echo "  RUN_ID: ${RUN_ID}"
echo "  Slurm node: ${SBATCH_NODELIST}"
echo "  Output: ${SBATCH_OUTPUT}"

if [[ "${WAIT_FOR_RESULT}" != "1" ]]; then
  exit 0
fi

deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  STATE="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sacct -j "${JOB_ID}" --format=State --noheader | head -n1 | xargs)"
  case "${STATE}" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT)
      break
      ;;
  esac
  sleep 5
done

"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sacct -j "${JOB_ID}" --format=JobID,State,ExitCode,Elapsed,NodeList --noheader
"${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "scontrol show job '${JOB_ID}' | tr '\n' ' '"

STATE="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- sacct -j "${JOB_ID}" --format=State --noheader | head -n1 | xargs)"
[[ "${STATE}" == "COMPLETED" ]]
