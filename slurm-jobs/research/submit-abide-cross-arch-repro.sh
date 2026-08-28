#!/usr/bin/env bash
# Submit the cross-arch ABIDE-I ORC reproducibility test to the slurm
# cluster (L4 on gpuorangefs-r770-proxmox, sm_89/sm_86 GPU).
#
# Output:
#   /work/<run-id>/abide_orc_repro/<arch_tag>/<subject>_orc.npy
#   /work/<run-id>/abide_orc_repro/<arch_tag>/{sinkhorn16iter.ptx, kaxi_runner}
#   slurm job comment: cross_arch_repro identical=N/M
#
# After the job completes, sync results back and diff against the
# local cohort baseline:
#   python3 scripts/research/abide_cross_arch_repro.py --mode diff \
#       --baseline artifacts/research/abide_orc \
#       --candidate artifacts/research/abide_orc_repro/<arch_tag_from_cluster>
#
# Usage:
#   bash slurm-jobs/research/submit-abide-cross-arch-repro.sh
#
# Env:
#   NS                       (default: slurm-pilot)
#   LOGIN_POD_NAME           (default: auto-detect)
#   SBATCH_NODELIST          (default: gpuorangefs-r770-proxmox)
#   SBATCH_PARTITION         (default: gpu-orangefs)
#   JOB_TIME                 (default: 00:20:00)
#   JOB_MEM                  (default: 8G)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS="${NS:-slurm-pilot}"
KUBECTL_BIN="${KUBECTL_BIN:-kubectl}"
LOGIN_POD_NAME="${LOGIN_POD_NAME:-}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
SBATCH_NODELIST="${SBATCH_NODELIST:-gpuorangefs-r770-proxmox}"
SBATCH_PARTITION="${SBATCH_PARTITION:-gpu-orangefs}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-plruntime}"
SBATCH_QOS="${SBATCH_QOS:-gpuorangefs}"
JOB_MEM="${JOB_MEM:-8G}"
JOB_TIME="${JOB_TIME:-00:20:00}"

SUBJECTS=(
  "CMU_a_0050642"
  "NYU_0050959"
  "Pitt_0050026"
  "UCLA_1_0051214"
  "Yale_0050628"
)

RUN_ID="${RUN_ID:-abide-cross-arch-repro-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_DIR="/tmp/${RUN_ID}.stage"
LOCAL_TARBALL="/tmp/${RUN_ID}.tgz"
LOCAL_SBATCH="/tmp/${RUN_ID}.sbatch"
cleanup() { rm -rf "${STAGE_DIR}" "${LOCAL_TARBALL}" "${LOCAL_SBATCH}"; }
trap cleanup EXIT

echo "[0/4] resolving login pod"
if [[ -n "${LOGIN_POD_NAME}" ]]; then
  LOGIN_POD="${LOGIN_POD_NAME}"
else
  LOGIN_POD="$("${KUBECTL_BIN}" -n "${NS}" get pods -l "${LOGIN_SELECTOR}" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
fi
[[ -z "${LOGIN_POD}" ]] && { echo "no live login pod in ${NS}" >&2; exit 1; }

echo "[1/4] staging payload (kretikos binary + cross-arch script + .1D files)"
mkdir -p "${STAGE_DIR}"
cp "${ROOT_DIR}/bin/kretikos" "${STAGE_DIR}/kretikos"
# Bundle the souc binary path that kretikos depends on
mkdir -p "${STAGE_DIR}/bin"
cp "${ROOT_DIR}/bin/souc" "${STAGE_DIR}/bin/souc"
cp -r "${ROOT_DIR}/bin/souc-bin" "${STAGE_DIR}/bin/" 2>/dev/null || true
cp "${ROOT_DIR}/scripts/gpu/kaxi_ptx_runner.c" "${STAGE_DIR}/"
cp "${ROOT_DIR}/scripts/research/abide_cohort_orc_sweep.py" "${STAGE_DIR}/"
cp "${ROOT_DIR}/scripts/research/abide_cross_arch_repro.py" "${STAGE_DIR}/"
# Bundle the self-hosted GPU emitter sources kretikos reads
mkdir -p "${STAGE_DIR}/self-hosted/gpu"
cp -r "${ROOT_DIR}/self-hosted/gpu/." "${STAGE_DIR}/self-hosted/gpu/"
# Bundle the 5 subject .1D files
mkdir -p "${STAGE_DIR}/abide"
for subj in "${SUBJECTS[@]}"; do
  cp "${ROOT_DIR}/artifacts/research/abide/${subj}_rois_cc200.1D" \
     "${STAGE_DIR}/abide/"
done

cat > "${STAGE_DIR}/run.sh" <<'RUNSH'
#!/usr/bin/env bash
# Cluster-side runner. Runs in the staged tarball directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "[cluster] nvidia-smi (compact):"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
echo

# Build runner.
cc -O2 kaxi_ptx_runner.c -ldl -lm -o kaxi_runner

# Compute arch tag from local GPU (matches abide_cross_arch_repro.py).
ARCH_TAG="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader \
  | head -1 \
  | awk -F',' 'BEGIN{OFS="_"} {gsub(/^ +| +$/,"",$1); gsub(/^ +| +$/,"",$2); gsub(/^ +| +$/,"",$3); gsub(/ /,"_",$1); gsub(/\./,"",$2); print $1, "sm" $2, "drv" $3}')"
echo "[cluster] arch_tag=${ARCH_TAG}"

# Emit PTX.
OUT_DIR="abide_orc_repro/${ARCH_TAG}"
mkdir -p "${OUT_DIR}"
PATH="$(pwd):$(pwd)/bin:${PATH}" ./kretikos kaxi-emit-ptx sinkhorn16 --f32 \
  -o "${OUT_DIR}/sinkhorn16iter.ptx"
cp kaxi_runner "${OUT_DIR}/kaxi_runner"
PTX_MD5="$(md5sum "${OUT_DIR}/sinkhorn16iter.ptx" | awk '{print $1}')"
RUNNER_MD5="$(md5sum "${OUT_DIR}/kaxi_runner" | awk '{print $1}')"
echo "[cluster] PTX md5    = ${PTX_MD5}"
echo "[cluster] runner md5 = ${RUNNER_MD5}"

# Point abide_cross_arch_repro at the staged abide/ dir.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# Symlink so the cohort script's ABIDE_DIR (artifacts/research/abide) resolves.
mkdir -p artifacts/research
ln -sfn "$(pwd)/abide" artifacts/research/abide
mkdir -p artifacts/research/abide_orc_repro
ln -sfn "$(pwd)/abide_orc_repro/${ARCH_TAG}" \
        artifacts/research/abide_orc_repro/${ARCH_TAG}

python3 abide_cross_arch_repro.py --mode local \
    --ptx "${OUT_DIR}/sinkhorn16iter.ptx" \
    --runner "${OUT_DIR}/kaxi_runner" \
    --arch-tag "${ARCH_TAG}"
echo "[cluster] cross_arch_repro arch_tag=${ARCH_TAG} ptx_md5=${PTX_MD5} runner_md5=${RUNNER_MD5}"
RUNSH
chmod +x "${STAGE_DIR}/run.sh"

tar -C "${STAGE_DIR}" -czf "${LOCAL_TARBALL}" .
echo "[2/4] staged payload size: $(du -sh "${LOCAL_TARBALL}" | awk '{print $1}')"

# Build sbatch script.
cat > "${LOCAL_SBATCH}" <<SBATCH
#!/bin/bash
#SBATCH --job-name=abide_cross_arch_repro
#SBATCH --output=abide_cross_arch_repro_%j.out
#SBATCH --error=abide_cross_arch_repro_%j.err
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --nodelist=${SBATCH_NODELIST}
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --qos=${SBATCH_QOS}
#SBATCH --gres=gpu:1
#SBATCH --mem=${JOB_MEM}
#SBATCH --time=${JOB_TIME}
set -euo pipefail
RID="${RUN_ID}"
WORK="/work/\${RID}"
mkdir -p "\${WORK}"
cd "\${WORK}"
echo "host: \$(hostname)"
echo "gpu:"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
tar -xzf "\${SLURM_SUBMIT_DIR}/${RUN_ID}.tgz"
bash run.sh
SBATCH

echo "[3/4] copying tarball + sbatch into login pod"
"${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_TARBALL}" "${LOGIN_POD}:/tmp/${RUN_ID}.tgz"
"${KUBECTL_BIN}" -n "${NS}" cp "${LOCAL_SBATCH}" "${LOGIN_POD}:/tmp/${RUN_ID}.sbatch"

echo "[4/4] submitting"
JOB_OUT="$("${KUBECTL_BIN}" -n "${NS}" exec "${LOGIN_POD}" -- bash -lc \
  "cp /tmp/${RUN_ID}.tgz ./ && sbatch /tmp/${RUN_ID}.sbatch")"
echo "${JOB_OUT}"

JOB_ID="$(echo "${JOB_OUT}" | grep -oE 'job [0-9]+' | head -1 | awk '{print $2}')"
echo "submitted: job_id=${JOB_ID}"
echo
echo "to fetch results once the job finishes:"
echo "  kubectl -n ${NS} cp ${LOGIN_POD}:abide_cross_arch_repro_${JOB_ID}.out /tmp/job_${JOB_ID}.out"
echo "  kubectl -n ${NS} exec ${LOGIN_POD} -- ls /work/${RUN_ID}/abide_orc_repro/"
echo "  kubectl -n ${NS} cp ${LOGIN_POD}:/work/${RUN_ID}/abide_orc_repro ./artifacts/research/abide_orc_repro_cluster"
echo "  python3 scripts/research/abide_cross_arch_repro.py --mode diff \\"
echo "      --baseline artifacts/research/abide_orc \\"
echo "      --candidate artifacts/research/abide_orc_repro_cluster/<arch_tag>"
