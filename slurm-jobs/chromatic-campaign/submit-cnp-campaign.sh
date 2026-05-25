#!/usr/bin/env bash
# Phase-D Hadwiger-Nelson campaign — fan a grid of candidate unit-distance graphs
# across the Slurm cluster (384 CPU / ~1TB) as a SLURM ARRAY, one (size,variant)
# candidate per task, and SAT-test 5-colourability of each.
#
#   5col=True   -> negative (expected; de Grey enlargement stays 5-colourable)
#   5col=False  -> χ≥6 CANDIDATE: the task keeps its CNF+DRAT; verify offline with
#                  drat-trim + bv_decide (formal/lean4/SounioHeule510NotColorable-style)
#                  BEFORE any claim.
#
# This scales the COMPUTE half of the campaign. It does NOT supply the ML/FunSearch
# generator (the creative half) — variants here are systematic enlargements/rotations
# of the de Grey ring, the known-negative direction. Expected outcome: all negative.
#
# Usage (from a shell with cluster/login-pod access, OR adapt for the BeagleCockpit
# MCP cockpit_submit_job path which is the canonical submitter):
#   cd /workspace/sounio
#   bash slurm-jobs/chromatic-campaign/submit-cnp-campaign.sh
#
# Env: NS (default slurm-pilot), PARTITION, NODELIST, JOB_TIME, JOB_MEM, ARRAY_MAX.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL_BIN:-kubectl}"
LOGIN_SELECTOR="${LOGIN_SELECTOR:-app.kubernetes.io/name=login}"
PARTITION="${PARTITION:-cpu}"            # adjust to a CPU partition on the cluster
NODELIST="${NODELIST:-}"                 # optional pin
JOB_TIME="${JOB_TIME:-12:00:00}"
JOB_MEM="${JOB_MEM:-8G}"                 # per array task
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
ARRAY_MAX="${ARRAY_MAX:-128}"            # concurrent array tasks (cluster has 384 cpu)
RUN_ID="${RUN_ID:-cnp-campaign-$(date -u +%Y%m%dT%H%M%S)}"
STAGE="/orangefs/training/sounio/${RUN_ID}"
VTX_URL="https://raw.githubusercontent.com/marijnheule/CNP-SAT/master/vtx/510.vtx"
EDGE_URL="https://raw.githubusercontent.com/marijnheule/CNP-SAT/master/edge/510.edge"

# Candidate grid (one line per array task): "SIZE RCAP VARIANT COLORS TIMEOUT_SEC".
# Sizes climb past HeuleGraph510; variants rotate the move-order to diversify the chunk;
# colors=5 asks "is this chunk 5-colourable?" (UNSAT => χ≥6). Edit freely.
CONFIGS=$(cat <<'CFG'
2600 6 0 5 36000
5000 8 0 5 36000
10000 12 0 5 36000
20000 18 0 5 36000
20000 18 1 5 36000
20000 18 2 5 36000
40000 28 0 5 36000
40000 28 1 5 36000
70000 40 0 5 36000
70000 40 1 5 36000
120000 55 0 5 36000
120000 55 1 5 36000
CFG
)
NTASK=$(printf '%s\n' "$CONFIGS" | grep -c .)

echo "run id:   ${RUN_ID}"
echo "stage:    ${STAGE}"
echo "configs:  ${NTASK} array tasks (max ${ARRAY_MAX} concurrent)"
echo "partition:${PARTITION}  time:${JOB_TIME}  mem/task:${JOB_MEM}  cpus/task:${CPUS_PER_TASK}"

LOGIN_POD="$(${KUBECTL} -n "${NS}" get pods -l "${LOGIN_SELECTOR}" \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
[[ -z "${LOGIN_POD}" ]] && { echo "ERROR: no live login pod in ns ${NS} (cluster not reachable from here)"; exit 3; }
echo "login pod:${LOGIN_POD}"

K() { ${KUBECTL} -n "${NS}" exec "${LOGIN_POD}" -- bash -lc "$1"; }

echo "[1/4] stage dirs + data + worker"
K "mkdir -p '${STAGE}/results' '${STAGE}/logs'"
K "curl -fsSL '${VTX_URL}' -o '${STAGE}/cnp510.vtx' && curl -fsSL '${EDGE_URL}' -o '${STAGE}/cnp510.edge' && wc -l '${STAGE}/cnp510.vtx' '${STAGE}/cnp510.edge'"
${KUBECTL} -n "${NS}" cp "${ROOT_DIR}/scripts/research/cnp_campaign_worker.py" "${NS}/${LOGIN_POD}:${STAGE}/worker.py"
printf '%s\n' "$CONFIGS" | K "cat > '${STAGE}/configs.txt'"

echo "[2/4] write sbatch array"
K "cat > '${STAGE}/run.sbatch' <<'SB'
#!/usr/bin/env bash
#SBATCH --job-name=cnp-campaign
#SBATCH --array=1-${NTASK}%${ARRAY_MAX}
#SBATCH --time=${JOB_TIME}
#SBATCH --mem=${JOB_MEM}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --output=${STAGE}/logs/task-%a.log
set -euo pipefail
cfg=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" ${STAGE}/configs.txt)
read SIZE RCAP VAR COL TMO <<<\"\$cfg\"
echo \"task \${SLURM_ARRAY_TASK_ID}: size=\$SIZE rcap=\$RCAP variant=\$VAR colors=\$COL timeout=\$TMO\"
python3 ${STAGE}/worker.py --data ${STAGE} --size \$SIZE --rcap \$RCAP --variant \$VAR \\
   --colors \$COL --timeout \$TMO --out ${STAGE}/results
SB"

echo "[3/4] submit"
[[ -n "${NODELIST}" ]] && NL="--nodelist=${NODELIST}" || NL=""
JOBID=$(K "cd ${STAGE} && sbatch --partition=${PARTITION} ${NL} run.sbatch | awk '{print \$NF}'")
echo "submitted array job ${JOBID}"

echo "[4/4] monitor + collect"
cat <<EOF

  Watch:    ${KUBECTL} -n ${NS} exec ${LOGIN_POD} -- squeue -j ${JOBID}
  Results:  ${KUBECTL} -n ${NS} exec ${LOGIN_POD} -- bash -lc 'cat ${STAGE}/results/*.json'
  HITS:     ${KUBECTL} -n ${NS} exec ${LOGIN_POD} -- bash -lc 'grep -l false ${STAGE}/results/*.json || echo NONE'
            ^ any 'kcolorable:false' is a χ≥6 CANDIDATE — pull its .cnf/.drat back and run
              drat-trim + the Lean bv_decide pipeline before claiming anything.
EOF
