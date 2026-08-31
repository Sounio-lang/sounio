#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
POLICY="$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1"
INSTALL="$ROOT_DIR/scripts/dev/install_spark_pair_arbiter.sh"
ARBITER="$ROOT_DIR/scripts/dev/spark_pair_arbiter.sh"
MODE="${1:---check}"
EVIDENCE="${SOUNIO_SPARK_PAIR_EVIDENCE:-$ROOT_DIR/tools/cluster/evidence/spark_pair_arbiter_live_gate.txt}"
HOLD_PID=''
SLURM_JOB=''
SECOND_OUTPUT='NOT_RUN'
SECOND_STATUS='NOT_RUN'
SLURM_JOB_STATE='NOT_RUN'
ADMISSION_RESULTS='NOT_RUN'
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
    kill -TERM "$HOLD_PID" >/dev/null 2>&1 || true
    printf 'spark-pair-live-gate: waiting for the arbiter fenced rollback; it will not be killed\n' >&2
    wait "$HOLD_PID" >/dev/null 2>&1 || true
  fi
  exit "$status"
}

slurm_states_resumed() {
  local nodes="$1" token state base flag count=0
  local -a parts
  while IFS= read -r token; do
    state="${token#State=}"
    IFS='+' read -r base flag <<<"$state"
    [[ "$base" == IDLE ]] || return 1
    IFS='+' read -ra parts <<<"$state"
    for flag in "${parts[@]:1}"; do
      [[ "$flag" != DRAIN && "$flag" != DRAINING && "$flag" != COMPLETING && "$flag" != DOWN && "$flag" != FAIL ]] || return 1
    done
    count=$((count + 1))
  done < <(tr ' ' '\n' <<<"$nodes" | sed -n '/^State=/p')
  [[ $count -eq 2 ]]
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
  local nodeset slurmd reservations state slurm_nodes jobs steps lease node_0 node_1 admission binding binding_policy binding_binding plugin
  "$ARBITER" verify >/dev/null
  lease="$(kubectl -n "$(policy_value namespace)" get lease "$(policy_value lease_name)" -o json)"
  state="$(jq -r --arg key "$(policy_value state_annotation)" '.metadata.annotations[$key]' <<<"$lease")"
  [[ "$state" == SLURM_OWNED ]] || fail "final Lease state is $state"
  nodeset="$(kubectl -n "$(policy_value nodeset_namespace)" get nodeset \
    "$(policy_value nodeset_name)" -o json)"
  [[ "$(jq -r '.spec.slurmd.resources.requests["nvidia.com/gpu"]' <<<"$nodeset")" == 1 ]] || fail 'NodeSet GPU request is not one'
  [[ "$(jq -r '.spec.slurmd.resources.limits["nvidia.com/gpu"]' <<<"$nodeset")" == 1 ]] || fail 'NodeSet GPU limit is not one'
  [[ "$(jq -r --arg key "$(policy_value nodeset_generation_annotation)" '.metadata.annotations[$key]' <<<"$lease")" == \
    "$(jq -r '.metadata.generation' <<<"$nodeset")" ]] || fail 'Lease does not bind the current NodeSet generation'
  node_0="$(kubectl get node "$(policy_value node_0_k8s)" -o json)"
  node_1="$(kubectl get node "$(policy_value node_1_k8s)" -o json)"
  for node in "$node_0" "$node_1"; do
    jq -e --arg key "$(policy_value spark_taint_key)" --arg value "$(policy_value spark_taint_value)" \
      --arg effect "$(policy_value spark_taint_effect)" \
      'any(.spec.taints[]?; .key == $key and .value == $value and .effect == $effect)' \
      <<<"$node" >/dev/null || fail 'Spark pair taint is missing or drifted'
  done
  admission="$(kubectl get validatingadmissionpolicy "$(policy_value admission_policy)" -o json)"
  binding="$(kubectl get validatingadmissionpolicybinding "$(policy_value admission_binding)" -o json)"
  binding_policy="$(kubectl get validatingadmissionpolicy "$(policy_value admission_binding_policy)" -o json)"
  binding_binding="$(kubectl get validatingadmissionpolicybinding "$(policy_value admission_binding_binding)" -o json)"
  jq -e '.spec.failurePolicy == "Fail" and .status.observedGeneration == .metadata.generation' \
    <<<"$admission" >/dev/null || fail 'admission policy is not accepted fail-closed'
  jq -e '.spec.paramRef.parameterNotFoundAction == "Deny" and .spec.validationActions == ["Deny"]' \
    <<<"$binding" >/dev/null || fail 'admission binding is not fail-closed'
  jq -e '.spec.failurePolicy == "Fail" and .status.observedGeneration == .metadata.generation and
    ((.status.typeChecking.expressionWarnings // []) | length) == 0' \
    <<<"$binding_policy" >/dev/null || fail 'manual binding admission policy is not accepted fail-closed'
  jq -e '.spec.validationActions == ["Deny"]' <<<"$binding_binding" >/dev/null || \
    fail 'manual binding admission binding is not fail-closed'
  for plugin in "$(policy_value device_plugin_0_name)" "$(policy_value device_plugin_1_name)"; do
    kubectl -n kube-system get daemonset "$plugin" -o json | jq -e \
      --arg key "$(policy_value spark_taint_key)" --arg value "$(policy_value spark_taint_value)" \
      --arg effect "$(policy_value spark_taint_effect)" \
      'any(.spec.template.spec.tolerations[]?; .key == $key and .value == $value and .effect == $effect)' \
      >/dev/null || fail "device plugin $plugin lacks the exact pair toleration"
  done
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
  slurm_states_resumed "$slurm_nodes" || fail 'both Slurm nodes are not exactly resumed IDLE'
  jobs="$(slurm_exec squeue -h -w "$(policy_value node_0_slurm),$(policy_value node_1_slurm)")"
  [[ -z "$jobs" ]] || fail 'Spark Slurm jobs remain after gate'
  steps="$(slurm_exec squeue --steps -h -w "$(policy_value node_0_slurm),$(policy_value node_1_slurm)")"
  [[ -z "$steps" ]] || fail 'Spark Slurm steps remain after gate'
}

expect_admission_deny() {
  local name="$1" output status
  set +e
  output="$(kubectl create --dry-run=server -f - 2>&1)"
  status=$?
  set -e
  [[ $status -ne 0 ]] || fail "admission probe $name was unexpectedly allowed"
  [[ "$output" == *'Spark GPU Pods require the current Pireus Lease epoch'* || \
     "$output" == *'pireus-spark-pair-fence'* ]] || \
    fail "admission probe $name failed for an unrelated reason: $output"
  ADMISSION_RESULTS="${ADMISSION_RESULTS};${name}=DENY"
}

verify_admission_negatives() {
  local epoch
  epoch="$(kubectl -n "$(policy_value namespace)" get lease "$(policy_value lease_name)" \
    -o json | jq -r --arg key "$(policy_value epoch_annotation)" '.metadata.annotations[$key]')"
  ADMISSION_RESULTS=''
  expect_admission_deny direct-node <<EOF
apiVersion: v1
kind: Pod
metadata: {name: pireus-negative-direct-node, namespace: default}
spec:
  nodeName: $(policy_value node_0_k8s)
  restartPolicy: Never
  containers: [{name: test, image: ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517, command: [sleep, infinity]}]
EOF
  expect_admission_deny explicit-toleration <<EOF
apiVersion: v1
kind: Pod
metadata: {name: pireus-negative-toleration, namespace: default}
spec:
  restartPolicy: Never
  tolerations: [{key: $(policy_value spark_taint_key), operator: Equal, value: $(policy_value spark_taint_value), effect: NoSchedule}]
  containers: [{name: test, image: ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517, command: [sleep, infinity]}]
EOF
  expect_admission_deny stale-epoch-reservation <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: pireus-spark-reservation-3c59
  namespace: beagle
  labels: {pireus.sounio.dev/spark-pair-reservation: "true", pireus.sounio.dev/spark-pair-epoch: "stale-$epoch"}
  annotations: {pireus.sounio.dev/spark-pair-holder: stale-holder}
spec:
  serviceAccountName: $(policy_value reservation_service_account)
  nodeSelector: {kubernetes.io/hostname: $(policy_value node_0_k8s)}
  tolerations: [{key: $(policy_value spark_taint_key), operator: Equal, value: $(policy_value spark_taint_value), effect: NoSchedule}]
  restartPolicy: Never
  containers:
    - name: reservation
      image: $(policy_value reservation_image)
      resources: {requests: {nvidia.com/gpu: "1"}, limits: {nvidia.com/gpu: "1"}}
EOF
  expect_admission_deny phase1-workload <<EOF
apiVersion: v1
kind: Pod
metadata: {name: pireus-negative-phase1-workload, namespace: beagle, labels: {$(policy_value workload_label): "true"}}
spec:
  nodeName: $(policy_value node_1_k8s)
  restartPolicy: Never
  containers: [{name: test, image: $(policy_value reservation_image), resources: {limits: {nvidia.com/gpu: "1"}}}]
EOF
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
  local result="$1" freeze_hash source_hash hardware toolchain lease nodeset
  mkdir -p "$(dirname "$EVIDENCE")"
  freeze_hash="$(sha256sum "$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1" | cut -d ' ' -f 1)"
  source_hash="$(sed -n 's/^authority_sha256=//p' "$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1")"
  toolchain="$(sed -n 's/^compiler_identity=//p' "$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1")"
  hardware="$(kubectl get nodes "$(policy_value node_0_k8s)" "$(policy_value node_1_k8s)" \
    -o jsonpath='{range .items[*]}{.metadata.name}:{.status.nodeInfo.architecture}:{.status.capacity.nvidia\.com/gpu}{";"}{end}')"
  lease="$(kubectl -n "$(policy_value namespace)" get lease "$(policy_value lease_name)" -o json)"
  nodeset="$(kubectl -n "$(policy_value nodeset_namespace)" get nodeset "$(policy_value nodeset_name)" -o json)"
  {
    printf 'schema=sounio-spark-pair-live-gate-v1\n'
    printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'sounio_source_sha256=%s\n' "$source_hash"
    printf 'semantics_freeze_sha256=%s\n' "$freeze_hash"
    printf 'receipt_emitter_language=Bash\n'
    printf 'receipt_emitter_role=LIVE_GATE\n'
    printf 'decision_producer_language=Sounio\n'
    printf 'decision_language_role=SEMANTIC_AUTHORITY\n'
    printf 'toolchain=%s\n' "$toolchain"
    printf 'hardware=%s\n' "$hardware"
    printf 'command=%s --apply\n' "$0"
    printf 'result=%s\n' "$result"
    printf 'lease_uid=%s\n' "$(jq -r '.metadata.uid' <<<"$lease")"
    printf 'lease_epoch=%s\n' "$(jq -r --arg key "$(policy_value epoch_annotation)" '.metadata.annotations[$key]' <<<"$lease")"
    printf 'nodeset_uid=%s\n' "$(jq -r '.metadata.uid' <<<"$nodeset")"
    printf 'nodeset_generation=%s\n' "$(jq -r '.metadata.generation' <<<"$nodeset")"
    printf 'gpu_3c59_uuid=%s\n' "$(jq -r '.metadata.annotations["pireus.sounio.dev/gpu-3c59-uuid"] // "missing"' <<<"$lease")"
    printf 'gpu_8e54_uuid=%s\n' "$(jq -r '.metadata.annotations["pireus.sounio.dev/gpu-8e54-uuid"] // "missing"' <<<"$lease")"
    printf 'nvidia_driver=%s\n' "$(jq -r '.metadata.annotations["pireus.sounio.dev/nvidia-driver"] // "missing"' <<<"$lease")"
    printf 'slurm_concurrent_attempt=%s\n' "$SLURM_JOB_STATE"
    printf 'second_kubernetes_holder_status=%s\n' "$SECOND_STATUS"
    printf 'second_kubernetes_holder_output=%s\n' "$(tr '\n' ' ' <<<"$SECOND_OUTPUT")"
    printf 'admission_negatives=%s\n' "$ADMISSION_RESULTS"
    printf 'model_download=NOT_RUN\n'
    printf 'litellm_change=NOT_RUN\n'
    printf 'pireus_semantics_change=NOT_RUN\n'
  } > "$EVIDENCE"
}

run_live_gate() {
  check_exclusions
  if ! kubectl -n "$(policy_value namespace)" get lease "$(policy_value lease_name)" >/dev/null 2>&1; then
    fail 'arbiter is not installed; run the separately reviewed installer gate first'
  fi
  verify_slurm_owned
  verify_admission_negatives

  SOUNIO_SPARK_PAIR_HOLDER=live-holder-primary \
    SOUNIO_SPARK_PAIR_RECEIPT_DIR="${SOUNIO_SPARK_PAIR_RECEIPT_DIR:-$HOME/.local/state/pireus-spark-pair/receipts}" \
    "$ARBITER" hold 20 >"$HOLDER_LOG" 2>&1 &
  HOLD_PID=$!
  wait_lease_state K8S_OWNED

  set +e
  SECOND_OUTPUT="$(SOUNIO_SPARK_PAIR_HOLDER=live-holder-secondary \
    SOUNIO_SPARK_PAIR_RECEIPT_DIR="${SOUNIO_SPARK_PAIR_RECEIPT_DIR:-$HOME/.local/state/pireus-spark-pair/receipts}" \
    "$ARBITER" hold 1 2>&1)"
  SECOND_STATUS=$?
  set -e
  [[ $SECOND_STATUS -eq 42 ]] || fail "second Kubernetes holder was not denied: $SECOND_OUTPUT"
  [[ "$SECOND_OUTPUT" == *'Lease acquisition refused'* || "$SECOND_OUTPUT" == *'foreign holder'* ]] || \
    fail "second holder failed for an unrelated reason: $SECOND_OUTPUT"

  SLURM_JOB="$(slurm_exec sbatch --parsable --job-name=pireus-negative-concurrency \
    --nodelist="$(policy_value node_0_slurm),$(policy_value node_1_slurm)" \
    --nodes=1 --gres=gpu:1 --time=00:01:00 --wrap='/bin/sleep 30')"
  SLURM_JOB="${SLURM_JOB%%;*}"
  sleep 2
  SLURM_JOB_STATE="$(slurm_exec squeue -h -j "$SLURM_JOB" -o '%T:%R')"
  [[ "$SLURM_JOB_STATE" == PENDING:*DRAINED* || "$SLURM_JOB_STATE" == PENDING:*drain* || "$SLURM_JOB_STATE" == PENDING:*ReqNodeNotAvail* ]] || \
    fail "Slurm concurrency probe was not fenced: $SLURM_JOB_STATE"
  slurm_exec scancel "$SLURM_JOB" >/dev/null
  SLURM_JOB=''

  wait "$HOLD_PID" || fail "primary holder failed: $(sed -n '1,200p' "$HOLDER_LOG")"
  HOLD_PID=''
  verify_slurm_owned
  verify_admission_negatives
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
