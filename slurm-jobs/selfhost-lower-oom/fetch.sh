#!/usr/bin/env bash
# Fetch lower-oom diagnosis results from the cpu-ops worker pod (worker-local FS).
# Usage: fetch.sh <RUN_ID>
set -euo pipefail
RUN_ID="${1:?usage: fetch.sh <RUN_ID>}"
NS="${NS:-slurm-pilot}"
KUBECTL="${KUBECTL:-kubectl}"
OUT="${OUT:-/workspace/sounio/slurm-jobs/selfhost-lower-oom/results/${RUN_ID}}"

WORKER="$(${KUBECTL} -n "${NS}" get pods -l 'app.kubernetes.io/name=slurmd' \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -i cpuops | head -1)"
[ -z "${WORKER}" ] && WORKER="$(${KUBECTL} -n "${NS}" get pods -o name | sed 's#pod/##' | grep -i 'worker-cpuops' | head -1)"
test -n "${WORKER}" || { echo "no cpuops worker pod" >&2; exit 1; }
echo "worker pod: ${WORKER}"

# pick the container that has the result dir
RES="/tmp/${RUN_ID}.result"
CTR=""
for c in $(${KUBECTL} -n "${NS}" get pod "${WORKER}" -o jsonpath='{.spec.containers[*].name}'); do
  if ${KUBECTL} -n "${NS}" exec "${WORKER}" -c "${c}" -- test -d "${RES}" 2>/dev/null; then CTR="${c}"; break; fi
done
test -n "${CTR}" || { echo "result dir ${RES} not found in any container of ${WORKER}" >&2; exit 1; }
echo "container: ${CTR}"

mkdir -p "${OUT}"
echo "===== SUMMARY.txt ====="
${KUBECTL} -n "${NS}" exec "${WORKER}" -c "${CTR}" -- cat "${RES}/SUMMARY.txt" 2>/dev/null | tee "${OUT}/SUMMARY.txt"
echo "===== pulling full result tar ====="
${KUBECTL} -n "${NS}" exec "${WORKER}" -c "${CTR}" -- tar -C "/tmp" -cz "$(basename "${RES}")" 2>/dev/null > "${OUT}/result.tgz"
tar -C "${OUT}" -xzf "${OUT}/result.tgz" 2>/dev/null && echo "extracted to ${OUT}/" || echo "(tar fetch partial)"
ls -la "${OUT}"
