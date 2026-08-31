#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
POLICY="$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1"
FREEZE="$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1"
ARBITER="$ROOT_DIR/scripts/dev/spark_pair_arbiter.sh"
SELFTEST="$ROOT_DIR/scripts/ci/spark_pair_arbiter_selftest.sh"
MODE="${1:---check}"
DRAIN_STARTED=0

fail() {
  printf 'install-spark-pair-arbiter: FAIL: %s\n' "$*" >&2
  exit 42
}

policy_value() {
  local file="$1" key="$2" count value
  count="$(sed -n "s/^${key}=//p" "$file" | wc -l | tr -d ' ')"
  [[ "$count" == 1 ]] || fail "policy key missing or duplicated: $key"
  value="$(sed -n "s/^${key}=//p" "$file")"
  [[ -n "$value" ]] || fail "empty policy key: $key"
  printf '%s\n' "$value"
}

slurm_exec() {
  kubectl -n "$(policy_value "$POLICY" slurm_login_namespace)" exec \
    "deploy/$(policy_value "$POLICY" slurm_login_deployment)" -- "$@"
}

preflight() {
  local node_0 node_1 nodeset plugin_0 plugin_1 jobs steps slurm_nodes reservations workloads
  local slurmd_pods node pod mem_available_mb slurm_line slurm_free_mb slurm_count=0
  "$ARBITER" verify >/dev/null || fail 'frozen arbiter verification failed'
  [[ "$(policy_value "$POLICY" allow_model_download)" == false ]] || fail 'model download must remain disabled'
  [[ "$(policy_value "$POLICY" allow_litellm_change)" == false ]] || fail 'LiteLLM changes must remain disabled'
  [[ "$(policy_value "$POLICY" allow_pireus_semantic_change)" == false ]] || fail 'Pireus semantic changes must remain disabled'

  node_0="$(kubectl get node "$(policy_value "$POLICY" node_0_k8s)" -o json)"
  node_1="$(kubectl get node "$(policy_value "$POLICY" node_1_k8s)" -o json)"
  nodeset="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o json)"
  plugin_0="$(kubectl -n "$(policy_value "$POLICY" device_plugin_0_namespace)" get daemonset \
    "$(policy_value "$POLICY" device_plugin_0_name)" -o json)"
  plugin_1="$(kubectl -n "$(policy_value "$POLICY" device_plugin_1_namespace)" get daemonset \
    "$(policy_value "$POLICY" device_plugin_1_name)" -o json)"

  [[ "$(jq -r '.metadata.uid' <<<"$node_0")" == "$(policy_value "$POLICY" node_0_uid)" ]] || fail 'node_0 UID drifted'
  [[ "$(jq -r '.metadata.uid' <<<"$node_1")" == "$(policy_value "$POLICY" node_1_uid)" ]] || fail 'node_1 UID drifted'
  [[ "$(jq -r '.metadata.uid' <<<"$nodeset")" == "$(policy_value "$POLICY" nodeset_uid)" ]] || fail 'NodeSet UID drifted'
  [[ "$(jq -r '.spec.scalingMode' <<<"$nodeset")" == DaemonSet ]] || fail 'NodeSet is not in DaemonSet mode'
  [[ "$(jq -r '.status.readyReplicas' <<<"$nodeset")" == 2 ]] || fail 'the two Slurm workers are not ready'
  [[ "$(jq -r '.metadata.uid' <<<"$plugin_0")" == "$(policy_value "$POLICY" device_plugin_0_uid)" ]] || fail '3c59 device plugin UID drifted'
  [[ "$(jq -r '.metadata.uid' <<<"$plugin_1")" == "$(policy_value "$POLICY" device_plugin_1_uid)" ]] || fail '8e54 device plugin UID drifted'

  jobs="$(slurm_exec squeue -h -w "$(policy_value "$POLICY" node_0_slurm),$(policy_value "$POLICY" node_1_slurm)")"
  [[ -z "$jobs" ]] || fail 'Spark Slurm queue is not empty'
  steps="$(slurm_exec squeue --steps -h -w "$(policy_value "$POLICY" node_0_slurm),$(policy_value "$POLICY" node_1_slurm)")"
  [[ -z "$steps" ]] || fail 'Spark Slurm steps are not empty'
  slurm_nodes="$(slurm_exec scontrol show node \
    "$(policy_value "$POLICY" node_0_slurm),$(policy_value "$POLICY" node_1_slurm)" -o)"
  [[ "$(grep -c 'CPUAlloc=0' <<<"$slurm_nodes")" == 2 ]] || fail 'Spark Slurm CPU allocations are nonzero'
  [[ "$(grep -c 'AllocMem=0' <<<"$slurm_nodes")" == 2 ]] || fail 'Spark Slurm memory allocations are nonzero'
  while IFS= read -r slurm_line; do
    slurm_free_mb="$(sed -n 's/.* FreeMem=\([0-9][0-9]*\).*/\1/p' <<<"$slurm_line")"
    [[ "$slurm_free_mb" =~ ^[0-9]+$ ]] || fail 'Slurm FreeMem observation is malformed'
    (( slurm_free_mb >= $(policy_value "$POLICY" minimum_free_memory_mb) )) || \
      fail "Slurm FreeMem below safety floor: ${slurm_free_mb}MiB"
    slurm_count=$((slurm_count + 1))
  done <<<"$slurm_nodes"
  [[ $slurm_count -eq 2 ]] || fail 'Slurm did not return exactly two Spark node records'

  slurmd_pods="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get pods \
    -l 'app.kubernetes.io/name=slurmd,app.kubernetes.io/instance=slurm-pilot-worker-spark' -o json)"
  for node in "$(policy_value "$POLICY" node_0_k8s)" "$(policy_value "$POLICY" node_1_k8s)"; do
    pod="$(jq -r --arg node "$node" '[.items[] | select(.spec.nodeName == $node and .status.phase == "Running") | .metadata.name] | if length == 1 then .[0] else "" end' <<<"$slurmd_pods")"
    [[ -n "$pod" ]] || fail "expected exactly one running slurmd Pod on $node"
    timeout "$(policy_value "$POLICY" kubelet_exec_timeout_seconds)" \
      kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" exec "$pod" -- /bin/true >/dev/null || \
      fail "kubelet exec path unavailable for $node"
    mem_available_mb="$(timeout "$(policy_value "$POLICY" kubelet_exec_timeout_seconds)" \
      kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" exec "$pod" -- \
      awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)"
    [[ "$mem_available_mb" =~ ^[0-9]+$ ]] || fail "MemAvailable observation malformed for $node"
    (( mem_available_mb >= $(policy_value "$POLICY" minimum_free_memory_mb) )) || \
      fail "MemAvailable below safety floor on $node: ${mem_available_mb}MiB"
  done

  reservations="$(kubectl -n "$(policy_value "$POLICY" namespace)" get pods \
    -l 'pireus.sounio.dev/spark-pair-reservation=true' -o json)"
  workloads="$(kubectl -n "$(policy_value "$POLICY" namespace)" get pods \
    -l "$(policy_value "$POLICY" workload_label)=true" -o json)"
  [[ "$(jq '.items | length' <<<"$reservations")" == 0 ]] || fail 'reservation pods already exist'
  [[ "$(jq '.items | length' <<<"$workloads")" == 0 ]] || fail 'Spark-pair workloads already exist'
}

on_failure() {
  local status=$?
  trap - EXIT INT TERM
  if [[ $status -ne 0 && $DRAIN_STARTED -eq 1 ]]; then
    printf 'install-spark-pair-arbiter: RECOVERY_REQUIRED; both Slurm nodes remain drained\n' >&2
  fi
  exit "$status"
}

apply_fence() {
  local holder
  holder="bootstrap-$(hostname)-$(date -u +%Y%m%dT%H%M%S)"
  DRAIN_STARTED=1
  SOUNIO_SPARK_PAIR_HOLDER="$holder" \
    SOUNIO_SPARK_PAIR_RECEIPT_DIR="${SOUNIO_SPARK_PAIR_RECEIPT_DIR:-$HOME/.local/state/pireus-spark-pair/receipts}" \
    "$ARBITER" bootstrap-init
  DRAIN_STARTED=0
}

main() {
  [[ "$MODE" == --check || "$MODE" == --apply ]] || fail 'usage: install_spark_pair_arbiter.sh --check|--apply'
  preflight
  if kubectl -n "$(policy_value "$POLICY" namespace)" get lease \
    "$(policy_value "$POLICY" lease_name)" >/dev/null 2>&1; then
    fail 'arbiter Lease already exists; use the arbiter recovery path'
  fi
  if [[ "$MODE" == --check ]]; then
    printf 'SPARK_PAIR_INSTALL_CHECK_PASS nodes=2 scaling=DaemonSet queue=empty semantics=frozen\n'
    return 0
  fi
  "$SELFTEST"
  trap on_failure EXIT INT TERM
  apply_fence
  trap - EXIT INT TERM
  printf 'SPARK_PAIR_INSTALL_PASS owner=SLURM_OWNED gpu_requests=2 lease=%s\n' \
    "$(policy_value "$POLICY" lease_name)"
}

main "$@"
