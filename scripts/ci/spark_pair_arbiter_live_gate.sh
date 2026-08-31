#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
POLICY="${SOUNIO_SPARK_PAIR_POLICY:-$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1}"
INSTALL="$ROOT_DIR/scripts/dev/install_spark_pair_arbiter.sh"
ARBITER="$ROOT_DIR/scripts/dev/spark_pair_arbiter.sh"
MODE="${1:---check}"
EVIDENCE="${SOUNIO_SPARK_PAIR_EVIDENCE:-$ROOT_DIR/tools/cluster/evidence/spark_pair_arbiter_live_gate.txt}"
HOLD_PID=''
SLURM_JOB=''
HOLDER_LOG="${TMPDIR:-/tmp}/pireus-spark-pair-live-holder.$$.log"

fail() {
  printf 'spark-pair-live-gate: FAIL: %s\n' "$*" >&2
  exit 42
}

policy_value() {
  local key="$1" count value
  count="$(sed -n "s/^${key}=//p" "$POLICY" | wc -l | tr -d ' ')"
  [[ "$count" == 1 ]] || fail "policy key missing or duplicated: $key"
  value="$(sed -n "s/^${key}=//p" "$POLICY")"
  [[ -n "$value" ]] || fail "empty policy key: $key"
  printf '%s\n' "$value"
}

slurm_exec() {
  kubectl -n "$(policy_value slurm_login_namespace)" exec \
    "deploy/$(policy_value slurm_login_deployment)" -- "$@"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$SLURM_JOB" ]]; then
    slurm_exec scancel "$SLURM_JOB" >/dev/null 2>&1 || true
  fi
  if [[ -n "$HOLD_PID" ]]; then
    wait "$HOLD_PID" >/dev/null 2>&1 || true
  fi
  exit "$status"
}

wait_lease_state() {
  local wanted="$1" deadline state
  deadline=$((SECONDS + $(policy_value operation_timeout_seconds)))
  while (( SECONDS < deadline )); do
    state="$(kubectl -n "$(policy_value namespace)" get lease "$(policy_value lease_name)" \
      -o json 2>/dev/null | jq -r --arg key "$(policy_value state_annotation)" \
      '.metadata.annotations[$key] // ""' || true)"
    [[ "$state" == "$wanted" ]] && return 0
    sleep 1
  done
  fail "Lease did not reach $wanted"
}

verify_slurm_owned() {
  local nodeset slurmd reservations state slurm_nodes jobs
  state="$(kubectl -n "$(policy_value namespace)" get lease "$(policy_value lease_name)" \
    -o json | jq -r --arg key "$(policy_value state_annotation)" '.metadata.annotations[$key]')"
  [[ "$state" == SLURM_OWNED ]] || fail "final Lease state is $state"
  nodeset="$(kubectl -n "$(policy_value nodeset_namespace)" get nodeset \
    "$(policy_value nodeset_name)" -o json)"
  [[ "$(jq -r '.spec.slurmd.resources.requests["nvidia.com/gpu"]' <<<"$nodeset")" == 1 ]] || fail 'NodeSet GPU request is not one'
  [[ "$(jq -r '.spec.slurmd.resources.limits["nvidia.com/gpu"]' <<<"$nodeset")" == 1 ]] || fail 'NodeSet GPU limit is not one'
  slurmd="$(kubectl -n "$(policy_value nodeset_namespace)" get pods \
    -l "$(kubectl -n "$(policy_value nodeset_namespace)" get nodeset "$(policy_value nodeset_name)" -o jsonpath='{.status.selector}')" -o json)"
  jq -e --arg n0 "$(policy_value node_0_k8s)" --arg n1 "$(policy_value node_1_k8s)" '
    (.items | length) == 2 and
    ([.items[].spec.nodeName] | sort) == ([$n0, $n1] | sort) and
    all(.items[];
      .status.phase == "Running" and
      .spec.containers[0].resources.requests["nvidia.com/gpu"] == "1" and
      .spec.containers[0].resources.limits["nvidia.com/gpu"] == "1")
  ' <<<"$slurmd" >/dev/null || fail 'both slurmd pods are not exact GPU owners'
  reservations="$(kubectl -n "$(policy_value namespace)" get pods \
    -l 'pireus.sounio.dev/spark-pair-reservation=true' -o json)"
  [[ "$(jq '.items | length' <<<"$reservations")" == 0 ]] || fail 'reservation pods remain after release'
  slurm_nodes="$(slurm_exec scontrol show node \
    "$(policy_value node_0_slurm),$(policy_value node_1_slurm)" -o)"
  [[ "$(grep -c 'State=IDLE' <<<"$slurm_nodes")" == 2 ]] || fail 'both Slurm nodes are not IDLE'
  jobs="$(slurm_exec squeue -h -w "$(policy_value node_0_slurm),$(policy_value node_1_slurm)")"
  [[ -z "$jobs" ]] || fail 'Spark Slurm jobs remain after gate'
}

check_exclusions() {
  kubectl -n beagle get dynamographdeployment sglang-agg-poc -o json | \
    jq -e 'all(.spec.services[]; .replicas == 0)' >/dev/null || \
    fail 'sglang-agg-poc is not scale-to-zero'
  if kubectl -n llm-router get configmap -o yaml | grep -qi inkling; then
    fail 'Inkling already appears in LiteLLM configuration'
  fi
}

write_evidence() {
  local result="$1" freeze_hash source_hash hardware toolchain
  mkdir -p "$(dirname "$EVIDENCE")"
  freeze_hash="$(sha256sum "$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1" | cut -d ' ' -f 1)"
  source_hash="$(sed -n 's/^authority_sha256=//p' "$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1")"
  toolchain="$(sed -n 's/^compiler_identity=//p' "$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1")"
  hardware="$(kubectl get nodes "$(policy_value node_0_k8s)" "$(policy_value node_1_k8s)" \
    -o jsonpath='{range .items[*]}{.metadata.name}:{.status.nodeInfo.architecture}:{.status.capacity.nvidia\.com/gpu}{";"}{end}')"
  {
    printf 'schema=sounio-spark-pair-live-gate-v1\n'
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'sounio_source_sha256=%s\n' "$source_hash"
    printf 'semantics_freeze_sha256=%s\n' "$freeze_hash"
    printf 'producer_language=Sounio\n'
    printf 'language_role=SEMANTIC_AUTHORITY\n'
    printf 'toolchain=%s\n' "$toolchain"
    printf 'hardware=%s\n' "$hardware"
    printf 'command=%s --apply\n' "$0"
    printf 'result=%s\n' "$result"
    printf 'slurm_concurrent_attempt=DENY_PENDING_DRAINED\n'
    printf 'second_kubernetes_holder=DENY_LEASE\n'
    printf 'model_download=NOT_RUN\n'
    printf 'litellm_change=NOT_RUN\n'
    printf 'pireus_semantics_change=NOT_RUN\n'
  } > "$EVIDENCE"
}

run_live_gate() {
  local second_output second_status job_state
  check_exclusions
  if ! kubectl -n "$(policy_value namespace)" get lease "$(policy_value lease_name)" >/dev/null 2>&1; then
    "$INSTALL" --apply
  fi
  verify_slurm_owned

  SOUNIO_SPARK_PAIR_HOLDER=live-holder-primary \
    SOUNIO_SPARK_PAIR_RECEIPT_DIR="${SOUNIO_SPARK_PAIR_RECEIPT_DIR:-$HOME/.local/state/pireus-spark-pair/receipts}" \
    "$ARBITER" hold 20 >"$HOLDER_LOG" 2>&1 &
  HOLD_PID=$!
  wait_lease_state K8S_OWNED

  set +e
  second_output="$(SOUNIO_SPARK_PAIR_HOLDER=live-holder-secondary \
    SOUNIO_SPARK_PAIR_RECEIPT_DIR="${SOUNIO_SPARK_PAIR_RECEIPT_DIR:-$HOME/.local/state/pireus-spark-pair/receipts}" \
    "$ARBITER" hold 1 2>&1)"
  second_status=$?
  set -e
  [[ $second_status -eq 42 ]] || fail "second Kubernetes holder was not denied: $second_output"

  SLURM_JOB="$(slurm_exec sbatch --parsable --job-name=pireus-negative-concurrency \
    --constraint=spark --nodes=1 --gres=gpu:1 --time=00:01:00 --wrap='/bin/sleep 30')"
  SLURM_JOB="${SLURM_JOB%%;*}"
  sleep 2
  job_state="$(slurm_exec squeue -h -j "$SLURM_JOB" -o '%T:%R')"
  [[ "$job_state" == PENDING:*DRAINED* || "$job_state" == PENDING:*drain* || "$job_state" == PENDING:*ReqNodeNotAvail* ]] || \
    fail "Slurm concurrency probe was not fenced: $job_state"
  slurm_exec scancel "$SLURM_JOB" >/dev/null
  SLURM_JOB=''

  wait "$HOLD_PID" || fail "primary holder failed: $(sed -n '1,200p' "$HOLDER_LOG")"
  HOLD_PID=''
  verify_slurm_owned
  check_exclusions
  write_evidence PASS
  printf 'SPARK_PAIR_LIVE_GATE_PASS evidence=%s\n' "$EVIDENCE"
}

main() {
  [[ "$MODE" == --check || "$MODE" == --apply ]] || fail 'usage: spark_pair_arbiter_live_gate.sh --check|--apply'
  check_exclusions
  if [[ "$MODE" == --check ]]; then
    "$INSTALL" --check
    printf 'SPARK_PAIR_LIVE_GATE_CHECK_PASS mutation=none\n'
    return 0
  fi
  trap cleanup EXIT INT TERM
  run_live_gate
  trap - EXIT INT TERM
}

main "$@"
