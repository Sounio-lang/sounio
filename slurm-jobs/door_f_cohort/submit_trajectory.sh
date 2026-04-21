#!/usr/bin/env bash
# P5.3 Door F sedenion-associator trajectory sweep.
#
# Per patient in the N=24 cohort, compiles and runs one Sounio chunk per
# CHUNK_EPOCHS-wide slice of the +/- RADIUS_S second window around seizure
# onset. Each chunk is a single souc invocation that emits NULL_ASSOC
# <slot> <value> for every 80-sample epoch in the chunk. The aggregator
# stitches per-chunk stdouts into per-patient trajectory TSVs keyed by
# global epoch index.
#
# Usage (from Sounio workspace):
#   cd /workspace/sounio
#   bash slurm-jobs/door_f_cohort/submit_trajectory.sh
#
# Tunables (env):
#   RUN_ID         label for the staging root (default: traj-<ts>)
#   CHUNK_EPOCHS   epochs per souc invocation (default: 20; avoid >35)
#   RADIUS_S       half-window around onset in seconds (default: 90)
#   PARALLEL       concurrent souc runs on the assigned node (default: 16)
#   PARTITION      slurm partition (default: gpu-orangefs)
#   NODELIST       restrict to node(s) (default: r770-proxmox)
#   CPUS           -c value (default: 16)
#   MEM            sbatch --mem (default: 16G)
#   TIMELIMIT      sbatch --time (default: 02:00:00)
#
# Output under /orangefs/training/sounio/door-f-runs/<RUN_ID>/:
#   sio/<pat>/chunk_<NNN>.sio            generated Sounio programs
#   stdout/<pat>/chunk_<NNN>.stdout      chunk run outputs
#   trajectories/<pat>.tsv               stitched per-patient trajectory
#   trajectories/cohort_trajectory.tsv   stacked across cohort
#   logs/job-<id>.log                    sbatch log

set -euo pipefail

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
PYTHON_VENV="${PYTHON_VENV:-/opt/sounio-venv/bin/python}"
EDF_DIR="${EDF_DIR:-/orangefs/training/sounio/chbmit}"
RUN_ID="${RUN_ID:-traj-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/door-f-runs/${RUN_ID}"
MANIFEST_REL="scripts/research/door_f_cohort/chbmit_manifest.tsv"
CHUNK_EPOCHS="${CHUNK_EPOCHS:-20}"
RADIUS_S="${RADIUS_S:-90}"
PARALLEL="${PARALLEL:-16}"
PARTITION="${PARTITION:-gpu-orangefs}"
NODELIST="${NODELIST:-r770-proxmox}"
CPUS="${CPUS:-16}"
MEM="${MEM:-16G}"
TIMELIMIT="${TIMELIMIT:-02:00:00}"

REQUIRED=(
    "bin/souc"
    "scripts/research/door_f_cohort/generate.py"
    "scripts/research/door_f_cohort/header.sio.part"
    "scripts/research/door_f_cohort/template_with_null.sio.part"
    "scripts/research/door_f_cohort/plan_trajectory_chunks.py"
    "scripts/research/door_f_cohort/aggregate_trajectory.py"
    "${MANIFEST_REL}"
    "stdlib/math/sedenion.sio"
)
for rel in "${REQUIRED[@]}"; do
    if [[ ! -e "${SOUNIO_DIR}/${rel}" ]]; then
        echo "missing: ${SOUNIO_DIR}/${rel}" >&2
        exit 2
    fi
done

LOGIN_POD="$(kubectl -n "${NS}" get pods -l app.kubernetes.io/name=login \
    --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')"
if [[ -z "${LOGIN_POD}" ]]; then
    echo "no live login pod in ns ${NS}" >&2; exit 3
fi
echo "login pod:    ${LOGIN_POD}"
echo "run id:       ${RUN_ID}"
echo "stage root:   ${STAGE_ROOT}"
echo "chunk/radius: ${CHUNK_EPOCHS} epochs / ${RADIUS_S}s"
echo "node:         ${NODELIST} (${CPUS} cpu, ${MEM}, ${PARALLEL}-way)"

echo
echo "[1/5] staging repo snapshot ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    mkdir -p '${STAGE_ROOT}/repo' '${STAGE_ROOT}/sio' \
             '${STAGE_ROOT}/stdout' '${STAGE_ROOT}/trajectories' \
             '${STAGE_ROOT}/logs'"
( cd "${SOUNIO_DIR}" && tar cf - \
    bin/souc \
    stdlib \
    scripts/research/door_f_cohort \
) | kubectl -n "${NS}" exec -i "${LOGIN_POD}" -- \
        bash -c "tar xf - -C '${STAGE_ROOT}/repo'"
kubectl -n "${NS}" exec "${LOGIN_POD}" -- \
    chmod +x "${STAGE_ROOT}/repo/bin/souc"

echo
echo "[2/5] planning chunks on login pod (generates all .sio files) ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -e
    cd '${STAGE_ROOT}/repo'
    '${PYTHON_VENV}' scripts/research/door_f_cohort/plan_trajectory_chunks.py \
        --manifest     '${STAGE_ROOT}/repo/${MANIFEST_REL}' \
        --edf-dir      '${EDF_DIR}' \
        --out-root     '${STAGE_ROOT}' \
        --chunk-epochs ${CHUNK_EPOCHS} \
        --radius-s     ${RADIUS_S} 2>&1 | tail -30
    echo
    echo 'chunk manifest summary:'
    wc -l '${STAGE_ROOT}/chunk_manifest.tsv'
"

echo
echo "[3/5] submitting sbatch ..."
SBATCH_FILE="/tmp/${RUN_ID}.sbatch"
cat > "${SBATCH_FILE}" <<SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_ID}
#SBATCH -p ${PARTITION}
#SBATCH --nodelist=${NODELIST}
#SBATCH -n 1
#SBATCH -c ${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIMELIMIT}
#SBATCH --output=${STAGE_ROOT}/logs/job-%j.log

set -euo pipefail
cd ${STAGE_ROOT}/repo
export SOUNIO_STDLIB_PATH=\${PWD}/stdlib

MANIFEST=${STAGE_ROOT}/chunk_manifest.tsv
STDOUT_ROOT=${STAGE_ROOT}/stdout

echo "[\$(date)] trajectory sweep on \$(hostname) with ${PARALLEL}-way parallelism"
awk -F'\t' 'NR>1' "\$MANIFEST" | wc -l | awk '{print "  chunks to run: "\$1}'

run_one() {
    local pid=\$1 cid=\$2 sio=\$3
    local dir=\${STDOUT_ROOT}/\${pid}
    mkdir -p "\${dir}"
    local base=\$(printf 'chunk_%03d' \$cid)
    local elf=\${dir}/\${base}.elf
    local out=\${dir}/\${base}.stdout
    local err=\${dir}/\${base}.stderr
    if ./bin/souc "\${sio}" "\${elf}" >"\${err}.compile" 2>&1; then
        chmod +x "\${elf}"
        "\${elf}" >"\${out}" 2>"\${err}" && echo "  OK   \${pid} \${base}" \
            || echo "  FAIL-run   \${pid} \${base} (exit=\$?)"
    else
        echo "  FAIL-compile \${pid} \${base}"
    fi
}
export -f run_one
export STDOUT_ROOT

awk -F'\t' 'NR>1 {print \$1"\t"\$2"\t"\$5}' "\$MANIFEST" | \
    xargs -P ${PARALLEL} -n 3 bash -c 'run_one "\$@"' _

echo
echo "[\$(date)] aggregation"
python3 scripts/research/door_f_cohort/aggregate_trajectory.py \
    --chunk-manifest  "\${MANIFEST}" \
    --stdout-dir      "\${STDOUT_ROOT}" \
    --master-manifest "${STAGE_ROOT}/repo/${MANIFEST_REL}" \
    --out-dir         "${STAGE_ROOT}/trajectories" \
    --chunk-epochs    ${CHUNK_EPOCHS} \
    --radius-s        ${RADIUS_S} 2>&1

echo "[\$(date)] done."
SBATCH_EOF

kubectl -n "${NS}" cp "${SBATCH_FILE}" "${LOGIN_POD}:${STAGE_ROOT}/job.sbatch"
JOB_ID=$(kubectl -n "${NS}" exec "${LOGIN_POD}" -- sbatch --parsable "${STAGE_ROOT}/job.sbatch")
echo "  submitted: job ${JOB_ID}"

echo
echo "[4/5] summary"
cat <<SUM
  RUN_ID:   ${RUN_ID}
  Stage:    ${STAGE_ROOT}
  Job:      ${JOB_ID}

Monitor:
  kubectl -n ${NS} exec ${LOGIN_POD} -- squeue -j ${JOB_ID}
  kubectl -n ${NS} exec ${LOGIN_POD} -- tail -f ${STAGE_ROOT}/logs/job-${JOB_ID}.log

Pull trajectories:
  kubectl -n ${NS} exec ${LOGIN_POD} -- \\
    tar -C ${STAGE_ROOT}/trajectories -cf - . | \\
    tar -C artifacts/research/door_f_cohort_N24/trajectories -xf -
SUM
