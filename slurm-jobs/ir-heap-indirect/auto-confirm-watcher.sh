#!/usr/bin/env bash
# Auto-confirm watcher for the Phase 1 fix. Waits for the slurm-pilot cpu-ops worker (and
# controller) to be genuinely healthy after the disk-pressure outage, then submits the
# confirm run (via a Running login pod if one exists, else directly via the controller
# pod), waits for REAL completion (SUMMARY reaches MULTIMODULE_GATE_RC) with infra-kill
# retry, and fetches SUMMARY + DBG markers. Self-heals through controller/worker flapping.
set -uo pipefail
cd /workspace/sounio-ir
NS=slurm-pilot
FINAL_RID=""

infra_ready() {
  local ct wk wready
  ct=$(kubectl -n $NS get pod slurm-pilot-controller-0 -o jsonpath='{.status.phase}' 2>/dev/null)
  [ "$ct" = "Running" ] || return 1
  wk=$(kubectl -n $NS get pods -o name 2>/dev/null | sed 's#pod/##' | grep -i worker-cpuops | head -1)
  [ -n "$wk" ] || return 1
  wready=$(kubectl -n $NS get pod "$wk" -o jsonpath='{.status.containerStatuses[?(@.name=="slurmd")].ready}' 2>/dev/null)
  [ "$wready" = "true" ] || return 1
  kubectl -n $NS exec "$wk" -c slurmd -- true 2>/dev/null || return 1
  # node must be responding to slurm (sinfo state not *down*); accept idle/alloc/mix
  kubectl -n $NS exec slurm-pilot-controller-0 -c slurmctld -- bash -lc 'sinfo -p cpu-ops -h -o "%T" 2>/dev/null' 2>/dev/null | grep -qiE 'idle|alloc|mix' || return 1
  return 0
}

submit_run() {
  local lp lc sout job rid sum worker i
  lp=$(kubectl -n $NS get pods -l app.kubernetes.io/name=login --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  if [ -n "$lp" ] && kubectl -n $NS exec "$lp" -- true 2>/dev/null; then
    sout=$(LOGIN_POD="$lp" LOGIN_CTR="" bash slurm-jobs/ir-heap-indirect/submit-phase1-verify.sh 2>&1)
  else
    sout=$(LOGIN_POD="slurm-pilot-controller-0" LOGIN_CTR="slurmctld" bash slurm-jobs/ir-heap-indirect/submit-phase1-verify.sh 2>&1)
  fi
  echo "$sout" | grep -E "Submitted|error" || true
  job=$(echo "$sout" | grep -oE 'batch job [0-9]+' | grep -oE '[0-9]+$')
  rid=$(cat /tmp/phase1.last_run_id 2>/dev/null)
  [ -z "$job" ] && { echo "submit_failed"; return 1; }
  echo "submitted JOB=$job RID=$rid"
  worker=$(kubectl -n $NS get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)
  for i in $(seq 1 80); do
    sum=$(kubectl -n $NS exec "$worker" -c slurmd -- cat /tmp/$rid.result/SUMMARY.txt 2>/dev/null || true)
    if echo "$sum" | grep -q "MULTIMODULE_GATE_RC="; then echo "GENUINE_DONE rid=$rid"; FINAL_RID="$rid"; return 0; fi
    if echo "$sum" | grep -q "mc build FAILED"; then echo "BUILD_FAILED_SOURCE rid=$rid"; FINAL_RID="$rid"; return 0; fi
    # infra-kill: worker vanished mid-run
    infra_ready || { echo "infra_lost_midrun rid=$rid — will retry"; return 1; }
    sleep 45
  done
  echo "job_timeout rid=$rid"; return 1
}

for attempt in $(seq 1 12); do
  echo "=== ATTEMPT $attempt $(date -u +%H:%M:%S) ==="
  ready=0
  for w in $(seq 1 40); do          # wait up to ~40 min for infra each attempt
    if infra_ready; then ready=1; break; fi
    sleep 60
  done
  [ "$ready" != 1 ] && { echo "attempt $attempt: infra not ready after 40min"; continue; }
  echo "attempt $attempt: infra READY, submitting"
  if submit_run; then break; fi
  sleep 30
done

echo "===== FINAL ($FINAL_RID) ====="
if [ -n "$FINAL_RID" ]; then
  WORKER=$(kubectl -n $NS get pods -o name | sed 's#pod/##' | grep -i worker-cpuops | head -1)
  OUT="slurm-jobs/ir-heap-indirect/results/$FINAL_RID"; mkdir -p "$OUT"
  kubectl -n $NS exec "$WORKER" -c slurmd -- cat /tmp/$FINAL_RID.result/SUMMARY.txt 2>/dev/null | tee "$OUT/SUMMARY.txt"
  echo "--- DBG ---"
  kubectl -n $NS exec "$WORKER" -c slurmd -- grep -E '^DBG_(A1|M) ' /tmp/$FINAL_RID.result/nc.log 2>/dev/null | head -16 | tee "$OUT/dbg.txt"
else
  echo "NO_CONFIRM: infra never stabilized across 12 attempts — needs operator attention"
fi
echo "WATCHER_DONE"
