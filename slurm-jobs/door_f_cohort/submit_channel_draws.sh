#!/usr/bin/env bash
# P5.5 / PROTOCOL §2 T3 — Channel-subset robustness sweep.
#
# For each patient in the N=24 CHB-MIT cohort, draws N_DRAWS random
# 16-channel subsets (PROTOCOL seed 20260420), generates one .sio per
# draw, compiles+runs with ./bin/souc in parallel, parses assoc_{FAR,
# PRE30, PRE10, PRE5, IC, POST} per draw, and reports the fraction of
# draws that preserve the cohort-median dip/spike sign.
#
# Usage (from Sounio workspace):
#   cd /workspace/sounio
#   bash slurm-jobs/door_f_cohort/submit_channel_draws.sh
#
# Tunables (env):
#   RUN_ID      label (default: t3-<ts>)
#   N_DRAWS     draws per patient (default: 100; PROTOCOL §2 T3)
#   PARALLEL    concurrent compile+run per node (default: 16)
#   PARTITION   slurm partition (default: gpu-orangefs)
#   NODELIST    restrict to nodes (default: gpuorangefs-r770-proxmox)
#   CPUS        -c value (default: 16)
#   MEM         sbatch --mem (default: 16G)
#   TIMELIMIT   sbatch --time (default: 02:00:00)
#   ONLY        comma-separated patient whitelist (default: full N=24)
#
# Output under /orangefs/training/sounio/door-f-runs/<RUN_ID>/:
#   sio/<pat>/draw_<NNN>.sio            generated Sounio programs
#   stdout/<pat>/draw_<NNN>.stdout      per-draw stdouts
#   draw_manifest.tsv                   patient<TAB>draw<TAB>channels<TAB>sio<TAB>edf
#   agg/t3_per_draw.tsv                 per-(patient,draw) dip/spike
#   agg/t3_cohort_by_draw.tsv           per-draw cohort median
#   agg/t3_robust.json                  headline pass/fail

set -euo pipefail

NS="${NS:-slurm-pilot}"
SOUNIO_DIR="${SOUNIO_DIR:-/workspace/sounio}"
PYTHON_VENV="${PYTHON_VENV:-/opt/sounio-venv/bin/python}"
EDF_DIR="${EDF_DIR:-/orangefs/training/sounio/chbmit}"
RUN_ID="${RUN_ID:-t3-$(date -u +%Y%m%dT%H%M%S)-${BASHPID}}"
STAGE_ROOT="/orangefs/training/sounio/door-f-runs/${RUN_ID}"
MANIFEST_REL="scripts/research/door_f_cohort/chbmit_manifest.tsv"
N_DRAWS="${N_DRAWS:-100}"
PARALLEL="${PARALLEL:-16}"
PARTITION="${PARTITION:-gpu-orangefs}"
NODELIST="${NODELIST:-gpuorangefs-r770-proxmox}"
CPUS="${CPUS:-16}"
MEM="${MEM:-16G}"
TIMELIMIT="${TIMELIMIT:-02:00:00}"
ONLY="${ONLY:-}"
ONLY_FLAG=""
if [[ -n "${ONLY}" ]]; then
    ONLY_FLAG="--only ${ONLY}"
fi

REQUIRED=(
    "bin/souc"
    "scripts/research/door_f_cohort/generate.py"
    "scripts/research/door_f_cohort/header.sio.part"
    "scripts/research/door_f_cohort/template.sio.part"
    "scripts/research/door_f_cohort/plan_channel_draws.py"
    "scripts/research/door_f_cohort/aggregate_channel_draws.py"
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
echo "login pod:   ${LOGIN_POD}"
echo "run id:      ${RUN_ID}"
echo "stage root:  ${STAGE_ROOT}"
echo "n_draws:     ${N_DRAWS}"
echo "node:        ${NODELIST} (${CPUS} cpu, ${MEM}, ${PARALLEL}-way)"

echo
echo "[1/5] staging repo snapshot ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    mkdir -p '${STAGE_ROOT}/repo' '${STAGE_ROOT}/sio' \
             '${STAGE_ROOT}/stdout' '${STAGE_ROOT}/agg' \
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
echo "[2/5] planning channel draws on login pod (generates ${N_DRAWS} x N .sio) ..."
kubectl -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "
    set -e
    cd '${STAGE_ROOT}/repo'
    '${PYTHON_VENV}' scripts/research/door_f_cohort/plan_channel_draws.py \
        --manifest '${STAGE_ROOT}/repo/${MANIFEST_REL}' \
        --edf-dir  '${EDF_DIR}' \
        --out-root '${STAGE_ROOT}' \
        --n-draws  ${N_DRAWS} \
        ${ONLY_FLAG} 2>&1 | tail -40
    wc -l '${STAGE_ROOT}/draw_manifest.tsv'
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

MANIFEST=${STAGE_ROOT}/draw_manifest.tsv
STDOUT_ROOT=${STAGE_ROOT}/stdout

echo "[\$(date)] T3 channel-subset sweep on \$(hostname) with ${PARALLEL}-way parallelism"
awk -F'\t' 'NR>1' "\$MANIFEST" | wc -l | awk '{print "  draws to run: "\$1}'

run_one() {
    local pid=\$1 did=\$2 sio=\$3
    local dir=\${STDOUT_ROOT}/\${pid}
    mkdir -p "\${dir}"
    local base=\$(printf 'draw_%03d' \$did)
    local elf=\${dir}/\${base}.elf
    local out=\${dir}/\${base}.stdout
    local err=\${dir}/\${base}.stderr
    if ./bin/souc "\${sio}" "\${elf}" >"\${err}.compile" 2>&1; then
        chmod +x "\${elf}"
        "\${elf}" >"\${out}" 2>"\${err}" && echo "  OK   \${pid} \${base}" \
            || echo "  FAIL-run   \${pid} \${base} (exit=\$?)"
        rm -f "\${elf}"
    else
        echo "  FAIL-compile \${pid} \${base}"
    fi
}
export -f run_one
export STDOUT_ROOT

awk -F'\t' 'NR>1 {print \$1"\t"\$2"\t"\$4}' "\$MANIFEST" | \
    xargs -P ${PARALLEL} -n 3 bash -c 'run_one "\$@"' _

echo
echo "[\$(date)] aggregation"
python3 scripts/research/door_f_cohort/aggregate_channel_draws.py \
    --draw-manifest "\${MANIFEST}" \
    --stdout-root   "\${STDOUT_ROOT}" \
    --out-dir       "${STAGE_ROOT}/agg" 2>&1

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

Pull results:
  kubectl -n ${NS} exec ${LOGIN_POD} -- \\
    tar -C ${STAGE_ROOT}/agg -cf - . | \\
    tar -C artifacts/research/door_f_cohort_N24/p5_channel_draws -xf -
SUM
