#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
POLICY=''
FREEZE=''

fail() {
  printf 'spark-pair-k8s-backend: FAIL: %s\n' "$*" >&2
  exit 42
}

policy_value() {
  local file="$1" key="$2" count value
  [[ -r "$file" ]] || fail "missing policy file: $file"
  count="$(sed -n "s/^${key}=//p" "$file" | wc -l | tr -d ' ')"
  [[ "$count" == 1 ]] || fail "policy key is missing or duplicated: $key"
  value="$(sed -n "s/^${key}=//p" "$file")"
  [[ -n "$value" ]] || fail "empty policy value: $key"
  printf '%s\n' "$value"
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

repo_root() {
  printf '%s\n' "$ROOT_DIR"
}

frozen_file_path() {
  local source_key="$1"
  printf '%s/%s\n' "$(repo_root)" "$(policy_value "$FREEZE" "$source_key")"
}

verify_frozen_file() {
  local source_key="$1" hash_key="$2" expected actual path
  path="$(frozen_file_path "$source_key")"
  expected="$(policy_value "$FREEZE" "$hash_key")"
  [[ -r "$path" ]] || fail "frozen file missing: $path"
  actual="$(sha256_file "$path")"
  [[ "$actual" == "$expected" ]] || fail "frozen file drifted: $source_key"
}

verify_frozen_material() {
  [[ "$(policy_value "$FREEZE" status)" == SEMANTICS_FROZEN ]] || fail 'semantics are not frozen'
  verify_frozen_file material_policy_source material_policy_sha256
  verify_frozen_file material_backend_source material_backend_sha256
  verify_frozen_file admission_manifest_source admission_manifest_sha256
  [[ "$(sha256_file "$POLICY")" == "$(policy_value "$FREEZE" material_policy_sha256)" ]] || fail 'selected policy is not frozen'
  [[ "$(sha256_file "${BASH_SOURCE[0]}")" == "$(policy_value "$FREEZE" material_backend_sha256)" ]] || fail 'running backend is not frozen'
}

receipt_value() {
  local receipt="$1" key="$2" count value
  [[ -r "$receipt" && -r "$receipt.sha256" ]] || fail 'decision receipt or digest is missing'
  count="$(sed -n "s/^${key}=//p" "$receipt" | wc -l | tr -d ' ')"
  [[ "$count" == 1 ]] || fail "receipt key missing or duplicated: $key"
  value="$(sed -n "s/^${key}=//p" "$receipt")"
  [[ -n "$value" ]] || fail "empty receipt key: $key"
  printf '%s\n' "$value"
}

frame_field() {
  local frame="$1" wanted="$2" token key value found=''
  for token in $frame; do
    key="${token%%=*}"
    value="${token#*=}"
    if [[ "$key" == "$wanted" ]]; then
      [[ -z "$found" ]] || fail "duplicated evidence field: $wanted"
      found="$value"
    fi
  done
  [[ -n "$found" ]] || fail "missing evidence field: $wanted"
  printf '%s\n' "$found"
}

verify_receipt() {
  local receipt="$1" allowed_actions="$2" epoch="$3" digest action receipt_from receipt_to expected_from expected_to
  digest="$(sed -n '1p' "$receipt.sha256")"
  [[ "$digest" == "$(sha256_file "$receipt")" ]] || fail 'decision receipt digest mismatch'
  [[ "$(receipt_value "$receipt" sounio_source_sha256)" == "$(policy_value "$FREEZE" authority_sha256)" ]] || fail 'receipt Sounio hash mismatch'
  [[ "$(receipt_value "$receipt" semantics_freeze_sha256)" == "$(sha256_file "$FREEZE")" ]] || fail 'receipt freeze hash mismatch'
  [[ "$(receipt_value "$receipt" receipt_emitter_language)" == Bash ]] || fail 'receipt emitter is not the material bridge'
  [[ "$(receipt_value "$receipt" receipt_emitter_role)" == MATERIAL_BRIDGE ]] || fail 'receipt emitter role mismatch'
  [[ "$(receipt_value "$receipt" decision_producer_language)" == Sounio ]] || fail 'decision producer is not Sounio'
  [[ "$(receipt_value "$receipt" decision_language_role)" == SEMANTIC_AUTHORITY ]] || fail 'decision role is not semantic authority'
  [[ "$(receipt_value "$receipt" material_policy_sha256)" == "$(policy_value "$FREEZE" material_policy_sha256)" ]] || fail 'receipt material policy hash mismatch'
  [[ "$(receipt_value "$receipt" material_backend_sha256)" == "$(policy_value "$FREEZE" material_backend_sha256)" ]] || fail 'receipt material backend hash mismatch'
  [[ "$(receipt_value "$receipt" admission_manifest_sha256)" == "$(policy_value "$FREEZE" admission_manifest_sha256)" ]] || fail 'receipt admission manifest hash mismatch'
  [[ "$(receipt_value "$receipt" epoch)" == "$epoch" ]] || fail 'receipt epoch does not match Lease epoch'
  [[ "$(receipt_value "$receipt" result)" == SOUNIO_SPARK_PAIR_ALLOW* ]] || fail 'receipt does not contain Sounio ALLOW'
  action="$(receipt_value "$receipt" action_code)"
  case " $allowed_actions " in
    *" $action "*) ;;
    *) fail "receipt action $action cannot authorize this effect" ;;
  esac
  receipt_from="$(receipt_value "$receipt" from_state)"
  receipt_to="$(receipt_value "$receipt" expected_to_state)"
  case "$action" in
    1) expected_from=UNINITIALIZED; expected_to=SLURM_OWNED ;;
    2) expected_from=SLURM_OWNED; expected_to=DRAINING_SLURM ;;
    3) expected_from=DRAINING_SLURM; expected_to=SLURM_QUIESCENT ;;
    4) expected_from=SLURM_QUIESCENT; expected_to=DETACHING_SLURMD ;;
    5) expected_from=DETACHING_SLURMD; expected_to=K8S_RESERVING ;;
    6) expected_from=K8S_RESERVING; expected_to=K8S_OWNED ;;
    7) expected_from=K8S_OWNED; expected_to=K8S_OWNED ;;
    8) expected_from=K8S_OWNED; expected_to=K8S_RELEASING ;;
    9) expected_from=K8S_RELEASING; expected_to=VERIFYING_GPU_CLEAN ;;
    10) expected_from=VERIFYING_GPU_CLEAN; expected_to=SLURM_RESTORING ;;
    11) expected_from="$receipt_from"; expected_to=RECOVERY_REQUIRED ;;
    12) expected_from=SLURM_RESTORING; expected_to=SLURM_OWNED ;;
    13) expected_from=RECOVERY_REQUIRED; expected_to=SLURM_OWNED ;;
    14) expected_from="$receipt_from"; expected_to="$receipt_from" ;;
    15|16|17|18|19|20|21|22) expected_from=RECOVERY_REQUIRED; expected_to=RECOVERY_REQUIRED ;;
    23|24|25|26|27|28) expected_from=UNINITIALIZED; expected_to=UNINITIALIZED ;;
    *) fail "receipt action $action has no material contract" ;;
  esac
  [[ "$receipt_from" == "$expected_from" && "$receipt_to" == "$expected_to" ]] || \
    fail 'receipt transition binding mismatch'
}

slurm_states_drained() {
  local nodes="$1" token state base flag count=0 has_drain
  local -a parts
  while IFS= read -r token; do
    state="${token#State=}"
    IFS='+' read -r base flag <<<"$state"
    [[ "$base" == IDLE ]] || return 1
    has_drain=0
    IFS='+' read -ra parts <<<"$state"
    for flag in "${parts[@]:1}"; do
      [[ "$flag" == DRAIN ]] && has_drain=1
      [[ "$flag" != DRAINING && "$flag" != COMPLETING && "$flag" != DOWN && "$flag" != FAIL ]] || return 1
    done
    [[ $has_drain -eq 1 ]] || return 1
    count=$((count + 1))
  done < <(tr ' ' '\n' <<<"$nodes" | sed -n '/^State=/p')
  [[ $count -eq 2 ]]
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

require_lease_context() {
  local holder="$1" epoch="$2" allowed_states="$3" lease state
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  [[ "$(jq -r '.spec.holderIdentity // ""' <<<"$lease")" == "$holder" ]] || fail 'material effect holder mismatch'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$epoch" ]] || fail 'material effect epoch mismatch'
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  case " $allowed_states " in
    *" $state "*) ;;
    *) fail "material effect is not allowed in Lease state $state" ;;
  esac
  lease_is_live "$lease" || fail 'Lease expired before material effect'
  printf '%s\n' "$state"
}

guard_mutation() {
  local kind="$1" holder="$2" epoch="$3" receipt="$4" state actions=''
  state="$(require_lease_context "$holder" "$epoch" 'UNINITIALIZED DRAINING_SLURM DETACHING_SLURMD K8S_RESERVING K8S_OWNED K8S_RELEASING SLURM_RESTORING RECOVERY_REQUIRED')"
  case "$kind:$state" in
    drain:UNINITIALIZED) actions=23 ;;
    fence:UNINITIALIZED) actions=24 ;;
    bootstrap-slurmd:UNINITIALIZED) actions=25 ;;
    resume:UNINITIALIZED) actions=26 ;;
    drain:DRAINING_SLURM) actions=2 ;;
    drain:RECOVERY_REQUIRED) actions=22 ;;
    detach:DETACHING_SLURMD) actions=4 ;;
    detach:RECOVERY_REQUIRED) actions=20 ;;
    reserve:K8S_RESERVING) actions=5 ;;
    reserve:RECOVERY_REQUIRED) actions=21 ;;
    stop:K8S_RELEASING) actions=8 ;;
    stop:RECOVERY_REQUIRED) actions=15 ;;
    probe:K8S_RELEASING) actions=8 ;;
    probe:RECOVERY_REQUIRED) actions=16 ;;
    delete:SLURM_RESTORING) actions=10 ;;
    delete:RECOVERY_REQUIRED) actions='15 17' ;;
    restore:SLURM_RESTORING) actions=10 ;;
    restore:RECOVERY_REQUIRED) actions=18 ;;
    resume:SLURM_RESTORING) actions=10 ;;
    resume:RECOVERY_REQUIRED) actions=19 ;;
    *) fail "effect $kind is not admitted from $state" ;;
  esac
  verify_receipt "$receipt" "$actions" "$epoch"
}

arg_value() {
  local wanted="$1"
  shift
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "$wanted" ]]; then
      [[ $# -ge 2 ]] || fail "missing value for $wanted"
      printf '%s\n' "$2"
      return 0
    fi
    shift
  done
  fail "required argument missing: $wanted"
}

kube_namespace() { policy_value "$POLICY" namespace; }
lease_name() { policy_value "$POLICY" lease_name; }
node0() { policy_value "$POLICY" node_0_k8s; }
node1() { policy_value "$POLICY" node_1_k8s; }
slurm0() { policy_value "$POLICY" node_0_slurm; }
slurm1() { policy_value "$POLICY" node_1_slurm; }

lease_json() {
  kubectl -n "$(kube_namespace)" get lease "$(lease_name)" -o json
}

admission_config_json() {
  kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" admission_configmap)" -o json
}

sync_admission_projection() {
  local lease="${1:-}" config state epoch holder source_hash freeze_hash updated
  [[ -n "$lease" ]] || lease="$(lease_json)"
  config="$(admission_config_json)"
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  holder="$(jq -r '.spec.holderIdentity // ""' <<<"$lease")"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  updated="$(jq --arg state "$state" --arg epoch "$epoch" --arg holder "$holder" \
    --arg source "$source_hash" --arg freeze "$freeze_hash" '
      .data.state = $state |
      .data.epoch = $epoch |
      .data.holder = $holder |
      .data.allowWorkload = "false" |
      .data.sounioSourceSha256 = $source |
      .data.semanticsFreezeSha256 = $freeze
    ' <<<"$config")"
  kubectl -n "$(kube_namespace)" replace -f - <<<"$updated" >/dev/null
}

admission_fail_closed() {
  local lease policy binding binding_policy binding_binding config state epoch holder source_hash freeze_hash
  lease="$1"
  policy="$(kubectl get validatingadmissionpolicy "$(policy_value "$POLICY" admission_policy)" -o json 2>/dev/null)" || return 1
  binding="$(kubectl get validatingadmissionpolicybinding "$(policy_value "$POLICY" admission_binding)" -o json 2>/dev/null)" || return 1
  binding_policy="$(kubectl get validatingadmissionpolicy "$(policy_value "$POLICY" admission_binding_policy)" -o json 2>/dev/null)" || return 1
  binding_binding="$(kubectl get validatingadmissionpolicybinding "$(policy_value "$POLICY" admission_binding_binding)" -o json 2>/dev/null)" || return 1
  config="$(admission_config_json 2>/dev/null)" || return 1
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  holder="$(jq -r '.spec.holderIdentity // ""' <<<"$lease")"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  jq -e '
    .spec.failurePolicy == "Fail" and
    .spec.paramKind.apiVersion == "v1" and
    .spec.paramKind.kind == "ConfigMap" and
    (.spec.validations | length) >= 1 and
    (.status.observedGeneration == .metadata.generation) and
    ((.status.typeChecking.expressionWarnings // []) | length) == 0
  ' <<<"$policy" >/dev/null || return 1
  jq -e --arg name "$(policy_value "$POLICY" admission_configmap)" --arg ns "$(kube_namespace)" '
    .spec.paramRef.name == $name and .spec.paramRef.namespace == $ns and
    .spec.paramRef.parameterNotFoundAction == "Deny" and
    (.spec.validationActions == ["Deny"])
  ' <<<"$binding" >/dev/null || return 1
  jq -e '
    .spec.failurePolicy == "Fail" and
    (.spec.validations | length) == 1 and
    (.status.observedGeneration == .metadata.generation) and
    ((.status.typeChecking.expressionWarnings // []) | length) == 0
  ' <<<"$binding_policy" >/dev/null || return 1
  jq -e --arg policy "$(policy_value "$POLICY" admission_binding_policy)" '
    .spec.policyName == $policy and .spec.validationActions == ["Deny"]
  ' <<<"$binding_binding" >/dev/null || return 1
  jq -e --arg state "$state" --arg epoch "$epoch" --arg holder "$holder" \
    --arg source "$source_hash" --arg freeze "$freeze_hash" '
      .data.state == $state and .data.epoch == $epoch and .data.holder == $holder and
      .data.allowWorkload == "false" and .data.sounioSourceSha256 == $source and
      .data.semanticsFreezeSha256 == $freeze
    ' <<<"$config" >/dev/null
}

pair_taint_exact() {
  local node_0="$1" node_1="$2" key value effect
  key="$(policy_value "$POLICY" spark_taint_key)"
  value="$(policy_value "$POLICY" spark_taint_value)"
  effect="$(policy_value "$POLICY" spark_taint_effect)"
  jq -e --arg key "$key" --arg value "$value" --arg effect "$effect" '
    ([.spec.taints[]? | select(.key == $key)] | length) == 1 and
    any(.spec.taints[]?; .key == $key and .value == $value and .effect == $effect)
  ' <<<"$node_0" >/dev/null &&
  jq -e --arg key "$key" --arg value "$value" --arg effect "$effect" '
    ([.spec.taints[]? | select(.key == $key)] | length) == 1 and
    any(.spec.taints[]?; .key == $key and .value == $value and .effect == $effect)
  ' <<<"$node_1" >/dev/null
}

unexpected_gpu_consumers_zero() {
  local pods="$1" epoch="$2" holder="$3"
  jq -e --arg n0 "$(node0)" --arg n1 "$(node1)" --arg epoch "$epoch" --arg holder "$holder" '
    [.items[] | select(.spec.nodeName == $n0 or .spec.nodeName == $n1) | select(
      (.metadata.namespace == "slurm-pilot" and
       .metadata.labels["app.kubernetes.io/name"] == "slurmd" and
       .metadata.labels["app.kubernetes.io/instance"] == "slurm-pilot-worker-spark") or
      (.metadata.namespace == "beagle" and
       .metadata.labels["pireus.sounio.dev/spark-pair-reservation"] == "true" and
       .metadata.labels["pireus.sounio.dev/spark-pair-epoch"] == $epoch and
       .metadata.annotations["pireus.sounio.dev/spark-pair-holder"] == $holder) or
      (.metadata.namespace == "kube-system") or
      (.metadata.namespace == "ceph-csi-cephfs") or
      (.metadata.namespace == "ceph-csi-rbd") or
      (.metadata.namespace == "nvidia-network-operator") or
      (.metadata.namespace == "darwin-observability-system")
    ) | not] | length == 0
  ' <<<"$pods" >/dev/null
}

lease_is_live() {
  local json="$1"
  jq -e '(.spec.renewTime | fromdateiso8601) + .spec.leaseDurationSeconds > now' \
    <<<"$json" >/dev/null
}

verify_lease_freeze_binding() {
  local lease="$1" source freeze
  source="$(jq -r --arg key "$(policy_value "$POLICY" source_hash_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")"
  freeze="$(jq -r --arg key "$(policy_value "$POLICY" freeze_hash_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")"
  [[ "$source" == "$(policy_value "$FREEZE" authority_sha256)" ]] || \
    fail 'Lease Sounio source hash differs from the active freeze'
  [[ "$freeze" == "$(sha256_file "$FREEZE")" ]] || \
    fail 'Lease semantics hash differs from the active freeze'
}

verify_bootstrap_journal_binding() {
  local journal="$1" source freeze
  source="$(jq -r '.data.sounioSourceSha256 // ""' <<<"$journal")"
  freeze="$(jq -r '.data.semanticsFreezeSha256 // ""' <<<"$journal")"
  [[ "$source" == "$(policy_value "$FREEZE" authority_sha256)" ]] || \
    fail 'bootstrap journal Sounio source hash differs from the active freeze'
  [[ "$freeze" == "$(sha256_file "$FREEZE")" ]] || \
    fail 'bootstrap journal semantics hash differs from the active freeze'
}

replace_lease() {
  kubectl -n "$(kube_namespace)" replace -f - >/dev/null
}

slurm_exec() {
  kubectl -n "$(policy_value "$POLICY" slurm_login_namespace)" exec \
    "deploy/$(policy_value "$POLICY" slurm_login_deployment)" -- "$@"
}

slurmd_pods_json() {
  local selector
  selector="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o jsonpath='{.status.selector}')"
  [[ -n "$selector" ]] || fail 'NodeSet status selector is empty'
  kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get pods -l "$selector" -o json
}

reservation_pods_json() {
  kubectl -n "$(kube_namespace)" get pods \
    -l 'pireus.sounio.dev/spark-pair-reservation=true' -o json
}

workload_pods_json() {
  kubectl -n "$(kube_namespace)" get pods \
    -l "$(policy_value "$POLICY" workload_label)=true" -o json
}

bit_add() {
  local current="$1" power="$2" truth="$3"
  if [[ "$truth" == 1 ]]; then
    printf '%s\n' "$((current + power))"
  else
    printf '%s\n' "$current"
  fi
}

prebootstrap_facts() {
  local holder node_0 node_1 nodeset slurm_nodes slurm_jobs slurm_steps all_pods
  local authority_mask=1 slurm_mask=0 k8s_mask=0 truth=0
  holder="$(arg_value --holder "$@")"
  if kubectl -n "$(kube_namespace)" get lease "$(lease_name)" >/dev/null 2>&1; then
    fail 'arbiter Lease already exists'
  fi
  if kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" bootstrap_journal)" >/dev/null 2>&1; then
    fail 'bootstrap journal already exists without a Lease'
  fi
  node_0="$(kubectl get node "$(node0)" -o json)"
  node_1="$(kubectl get node "$(node1)" -o json)"
  nodeset="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o json)"
  slurm_nodes="$(slurm_exec scontrol show node "$(slurm0),$(slurm1)" -o)"
  slurm_jobs="$(slurm_exec squeue -h -w "$(slurm0),$(slurm1)")"
  slurm_steps="$(slurm_exec squeue --steps -h -w "$(slurm0),$(slurm1)")"
  all_pods="$(kubectl get pods -A -o json)"

  truth=0
  if [[ "$(jq -r '.metadata.uid' <<<"$node_0")" == "$(policy_value "$POLICY" node_0_uid)" &&
        "$(jq -r '.metadata.uid' <<<"$node_1")" == "$(policy_value "$POLICY" node_1_uid)" &&
        "$(jq -r '.metadata.uid' <<<"$nodeset")" == "$(policy_value "$POLICY" nodeset_uid)" ]]; then
    truth=1
  fi
  authority_mask="$(bit_add "$authority_mask" 32 "$truth")"
  authority_mask="$(bit_add "$authority_mask" 64 1)"
  truth=0
  slurm_exec scontrol ping | grep -q 'is UP' && truth=1
  slurm_mask="$(bit_add "$slurm_mask" 2 "$truth")"
  truth=0
  [[ -z "$slurm_jobs" && -z "$slurm_steps" ]] && truth=1
  slurm_mask="$(bit_add "$slurm_mask" 8 "$truth")"
  truth=0
  if [[ "$(grep -c 'CPUAlloc=0' <<<"$slurm_nodes")" == 2 &&
        "$(grep -c 'AllocMem=0' <<<"$slurm_nodes")" == 2 &&
        "$(grep -c 'AllocTRES= ' <<<"$slurm_nodes")" == 2 ]]; then
    truth=1
  fi
  slurm_mask="$(bit_add "$slurm_mask" 16 "$truth")"
  truth=0
  unexpected_gpu_consumers_zero "$all_pods" 1 "$holder" && truth=1
  k8s_mask="$(bit_add "$k8s_mask" 512 "$truth")"
  k8s_mask="$(bit_add "$k8s_mask" 256 1)"
  printf 'state=UNINITIALIZED epoch=1 observed_epoch=1 authority_mask=%s slurm_mask=%s k8s_mask=%s\n' \
    "$authority_mask" "$slurm_mask" "$k8s_mask"
}

facts() {
  local holder lease nodeset node_0 node_1 plugin_0 plugin_1 plugin_pods slurmd_pods reservations workloads all_pods slurm_nodes slurm_jobs slurm_steps
  local state epoch observed_epoch authority_mask=1 slurm_mask=0 k8s_mask=0 truth current_generation lease_generation
  holder="$(arg_value --holder "$@")"
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  nodeset="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o json)"
  node_0="$(kubectl get node "$(node0)" -o json)"
  node_1="$(kubectl get node "$(node1)" -o json)"
  plugin_0="$(kubectl -n "$(policy_value "$POLICY" device_plugin_0_namespace)" get daemonset \
    "$(policy_value "$POLICY" device_plugin_0_name)" -o json)"
  plugin_1="$(kubectl -n "$(policy_value "$POLICY" device_plugin_1_namespace)" get daemonset \
    "$(policy_value "$POLICY" device_plugin_1_name)" -o json)"
  plugin_pods="$(kubectl -n kube-system get pods -o json)"
  slurmd_pods="$(slurmd_pods_json)"
  reservations="$(reservation_pods_json)"
  workloads="$(workload_pods_json)"
  all_pods="$(kubectl get pods -A -o json)"
  slurm_nodes="$(slurm_exec scontrol show node "$(slurm0),$(slurm1)" -o)"
  slurm_jobs="$(slurm_exec squeue -h -w "$(slurm0),$(slurm1)")"
  slurm_steps="$(slurm_exec squeue --steps -h -w "$(slurm0),$(slurm1)")"

  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")"
  epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" \
    '.metadata.annotations[$key] // "0"' <<<"$lease")"
  observed_epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" \
    '.metadata.annotations[$key] // "0"' <<<"$node_0")"
  if [[ "$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" \
    '.metadata.annotations[$key] // "0"' <<<"$node_1")" != "$observed_epoch" ]]; then
    observed_epoch=0
  fi
  [[ "$epoch" =~ ^[1-9][0-9]*$ ]] || fail 'Lease epoch is missing or invalid'

  truth=0
  [[ "$(jq -r '.spec.holderIdentity // ""' <<<"$lease")" == "$holder" ]] && truth=1
  authority_mask="$(bit_add "$authority_mask" 8 "$truth")"
  truth=0
  lease_is_live "$lease" && truth=1
  authority_mask="$(bit_add "$authority_mask" 16 "$truth")"
  truth=0
  if [[ "$(jq -r '.metadata.uid' <<<"$node_0")" == "$(policy_value "$POLICY" node_0_uid)" &&
        "$(jq -r '.metadata.uid' <<<"$node_1")" == "$(policy_value "$POLICY" node_1_uid)" &&
        "$(jq -r '.metadata.uid' <<<"$nodeset")" == "$(policy_value "$POLICY" nodeset_uid)" ]]; then
    truth=1
  fi
  authority_mask="$(bit_add "$authority_mask" 32 "$truth")"
  current_generation="$(jq -r '.metadata.generation' <<<"$nodeset")"
  lease_generation="$(jq -r --arg key "$(policy_value "$POLICY" nodeset_generation_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")"
  truth=0
  [[ "$current_generation" == "$lease_generation" ]] && truth=1
  authority_mask="$(bit_add "$authority_mask" 64 "$truth")"

  truth=0
  if [[ "$(jq -r '.metadata.uid' <<<"$plugin_0")" == "$(policy_value "$POLICY" device_plugin_0_uid)" &&
        "$(jq -r '.metadata.uid' <<<"$plugin_1")" == "$(policy_value "$POLICY" device_plugin_1_uid)" &&
        "$(jq -r '.spec.template.spec.containers[0].image' <<<"$plugin_0")" == "$(policy_value "$POLICY" device_plugin_image)" &&
        "$(jq -r '.spec.template.spec.containers[0].image' <<<"$plugin_1")" == "$(policy_value "$POLICY" device_plugin_image)" &&
        "$(jq --arg n0 "$(node0)" --arg n1 "$(node1)" '[.items[] | select(
          (.spec.nodeName == $n0 or .spec.nodeName == $n1) and
          any(.spec.containers[]?; .image | contains("nvidia/k8s-device-plugin"))
        )] | length' <<<"$plugin_pods")" == 2 &&
        "$(jq -r --arg n0 "$(node0)" --arg n1 "$(node1)" '[.items[] | select(
          (.spec.nodeName == $n0 or .spec.nodeName == $n1) and
          any(.spec.containers[]?; .image | contains("nvidia/k8s-device-plugin"))
        ) | .metadata.ownerReferences[0].uid] | sort | join(",")' <<<"$plugin_pods")" == \
          "$(printf '%s\n%s\n' "$(policy_value "$POLICY" device_plugin_0_uid)" \
            "$(policy_value "$POLICY" device_plugin_1_uid)" | sort | paste -sd, -)" &&
        "$(jq '[.spec.template.spec.containers[0].env[]? | select(.name == "CONFIG_FILE" or .name == "MPS_ROOT")] | length' <<<"$plugin_0")" == 0 &&
        "$(jq '[.spec.template.spec.containers[0].env[]? | select(.name == "CONFIG_FILE" or .name == "MPS_ROOT")] | length' <<<"$plugin_1")" == 0 &&
        "$(jq --arg key "$(policy_value "$POLICY" spark_taint_key)" --arg value "$(policy_value "$POLICY" spark_taint_value)" \
          --arg effect "$(policy_value "$POLICY" spark_taint_effect)" '[.spec.template.spec.tolerations[]? |
          select(.key == $key and .value == $value and .effect == $effect)] | length' <<<"$plugin_0")" == 1 &&
        "$(jq --arg key "$(policy_value "$POLICY" spark_taint_key)" --arg value "$(policy_value "$POLICY" spark_taint_value)" \
          --arg effect "$(policy_value "$POLICY" spark_taint_effect)" '[.spec.template.spec.tolerations[]? |
          select(.key == $key and .value == $value and .effect == $effect)] | length' <<<"$plugin_1")" == 1 ]]; then
    truth=1
  fi
  authority_mask="$(bit_add "$authority_mask" 128 "$truth")"

  truth=0
  admission_fail_closed "$lease" && truth=1
  authority_mask="$(bit_add "$authority_mask" 256 "$truth")"
  truth=0
  pair_taint_exact "$node_0" "$node_1" && truth=1
  authority_mask="$(bit_add "$authority_mask" 512 "$truth")"

  truth=0
  if [[ "$(jq -r '.spec.slurmd.resources.requests["nvidia.com/gpu"] // "0"' <<<"$nodeset")" == 1 &&
        "$(jq -r '.spec.slurmd.resources.limits["nvidia.com/gpu"] // "0"' <<<"$nodeset")" == 1 ]]; then
    truth=1
  fi
  slurm_mask="$(bit_add "$slurm_mask" 1 "$truth")"
  truth=0
  slurm_exec scontrol ping | grep -q 'is UP' && truth=1
  slurm_mask="$(bit_add "$slurm_mask" 2 "$truth")"
  truth=0
  slurm_states_drained "$slurm_nodes" && truth=1
  slurm_mask="$(bit_add "$slurm_mask" 4 "$truth")"
  truth=0
  [[ -z "$slurm_jobs" && -z "$slurm_steps" ]] && truth=1
  slurm_mask="$(bit_add "$slurm_mask" 8 "$truth")"
  truth=0
  if [[ "$(grep -c 'CPUAlloc=0' <<<"$slurm_nodes")" == 2 &&
        "$(grep -c 'AllocMem=0' <<<"$slurm_nodes")" == 2 &&
        "$(grep -c 'AllocTRES= ' <<<"$slurm_nodes")" == 2 ]]; then
    truth=1
  fi
  slurm_mask="$(bit_add "$slurm_mask" 16 "$truth")"
  truth=0
  [[ "$(jq '.items | length' <<<"$slurmd_pods")" == 0 ]] && truth=1
  slurm_mask="$(bit_add "$slurm_mask" 32 "$truth")"
  truth=0
  if jq -e --arg n0 "$(node0)" --arg n1 "$(node1)" '
      (.items | length) == 2 and
      ([.items[].spec.nodeName] | sort) == ([$n0, $n1] | sort) and
      all(.items[];
        .status.phase == "Running" and
        .spec.containers[0].resources.requests["nvidia.com/gpu"] == "1" and
        .spec.containers[0].resources.limits["nvidia.com/gpu"] == "1")
    ' <<<"$slurmd_pods" >/dev/null; then
    truth=1
  fi
  slurm_mask="$(bit_add "$slurm_mask" 64 "$truth")"
  truth=0
  if [[ "$state" == UNINITIALIZED || "$state" == SLURM_OWNED || "$state" == SLURM_RESTORING || "$state" == RECOVERY_REQUIRED ]]; then
    if slurm_states_resumed "$slurm_nodes"; then truth=1; fi
  fi
  slurm_mask="$(bit_add "$slurm_mask" 128 "$truth")"

  truth=0
  if [[ "$(jq -r '.status.capacity["nvidia.com/gpu"] // "0"' <<<"$node_0")" == 1 &&
        "$(jq -r '.status.capacity["nvidia.com/gpu"] // "0"' <<<"$node_1")" == 1 ]]; then
    truth=1
  fi
  k8s_mask="$(bit_add "$k8s_mask" 1 "$truth")"
  truth=0
  [[ "$(jq '.items | length' <<<"$reservations")" == 2 ]] && truth=1
  k8s_mask="$(bit_add "$k8s_mask" 2 "$truth")"
  truth=0
  [[ "$(jq '.items | length' <<<"$reservations")" == 0 ]] && truth=1
  k8s_mask="$(bit_add "$k8s_mask" 256 "$truth")"
  truth=0
  if jq -e --arg n0 "$(node0)" --arg n1 "$(node1)" \
    '([.items[].spec.nodeName] | sort) == ([$n0, $n1] | sort)' <<<"$reservations" >/dev/null; then truth=1; fi
  k8s_mask="$(bit_add "$k8s_mask" 4 "$truth")"
  truth=0
  if jq -e --arg key "$(policy_value "$POLICY" epoch_label)" --arg epoch "$epoch" \
    'all(.items[]; .metadata.labels[$key] == $epoch)' <<<"$reservations" >/dev/null; then truth=1; fi
  k8s_mask="$(bit_add "$k8s_mask" 8 "$truth")"
  truth=0
  if [[ "$(jq -r --arg key 'pireus.sounio.dev/nvml-3c59-epoch' '.metadata.annotations[$key] // ""' <<<"$lease")" == "$epoch" &&
        "$(jq -r --arg key 'pireus.sounio.dev/nvml-8e54-epoch' '.metadata.annotations[$key] // ""' <<<"$lease")" == "$epoch" ]]; then
    truth=1
  fi
  k8s_mask="$(bit_add "$k8s_mask" 16 "$truth")"
  truth=0
  [[ "$(jq '.items | length' <<<"$workloads")" == 0 ]] && truth=1
  k8s_mask="$(bit_add "$k8s_mask" 32 "$truth")"
  truth=0
  if jq -e --arg key "$(policy_value "$POLICY" epoch_label)" --arg epoch "$epoch" \
    '[.items[] | select(.metadata.labels[$key] != $epoch)] | length == 0' <<<"$workloads" >/dev/null; then truth=1; fi
  k8s_mask="$(bit_add "$k8s_mask" 64 "$truth")"
  truth=0
  lease_is_live "$lease" && truth=1
  k8s_mask="$(bit_add "$k8s_mask" 128 "$truth")"
  truth=0
  unexpected_gpu_consumers_zero "$all_pods" "$epoch" \
    "$(jq -r '.spec.holderIdentity // ""' <<<"$lease")" && truth=1
  k8s_mask="$(bit_add "$k8s_mask" 512 "$truth")"

  printf 'state=%s epoch=%s observed_epoch=%s authority_mask=%s slurm_mask=%s k8s_mask=%s\n' \
    "$state" "$epoch" "$observed_epoch" "$authority_mask" "$slurm_mask" "$k8s_mask"
}

lease_acquire() {
  local holder lease state epoch duration now live_holder nodeset_generation updated
  holder="$(arg_value --holder "$@")"
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  [[ "$state" == SLURM_OWNED ]] || fail "Lease state is $state, not SLURM_OWNED"
  live_holder="$(jq -r '.spec.holderIdentity // ""' <<<"$lease")"
  if lease_is_live "$lease" && [[ "$live_holder" != "$holder" && "$live_holder" != slurm-owned ]]; then
    fail "Lease is live under foreign holder $live_holder"
  fi
  epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // "0"' <<<"$lease")"
  [[ "$epoch" =~ ^[0-9]+$ ]] || fail 'stored Lease epoch is invalid'
  epoch=$((epoch + 1))
  duration="$(policy_value "$POLICY" lease_duration_seconds)"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nodeset_generation="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o jsonpath='{.metadata.generation}')"
  updated="$(jq --arg holder "$holder" --arg now "$now" --arg epoch "$epoch" \
    --arg state SLURM_OWNED --arg generation "$nodeset_generation" --argjson duration "$duration" \
    --arg state_key "$(policy_value "$POLICY" state_annotation)" \
    --arg epoch_key "$(policy_value "$POLICY" epoch_annotation)" \
    --arg generation_key "$(policy_value "$POLICY" nodeset_generation_annotation)" '
      .spec.holderIdentity = $holder |
      .spec.acquireTime = $now |
      .spec.renewTime = $now |
      .spec.leaseDurationSeconds = $duration |
      .spec.leaseTransitions = ((.spec.leaseTransitions // 0) + 1) |
      .metadata.annotations[$state_key] = $state |
      .metadata.annotations[$epoch_key] = $epoch |
      .metadata.annotations[$generation_key] = $generation
    ' <<<"$lease")"
  replace_lease <<<"$updated"
  kubectl annotate nodes "$(node0)" "$(node1)" \
    "$(policy_value "$POLICY" epoch_annotation)=$epoch" --overwrite >/dev/null
  sync_admission_projection "$updated"
  printf 'epoch=%s state=SLURM_OWNED\n' "$epoch"
}

lease_transition() {
  local holder epoch from to receipt lease now updated action
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  from="$(arg_value --from "$@")"
  to="$(arg_value --to "$@")"
  receipt="$(arg_value --receipt "$@")"
  case "$from:$to" in
    UNINITIALIZED:SLURM_OWNED) action=1 ;;
    SLURM_OWNED:DRAINING_SLURM) action=2 ;;
    DRAINING_SLURM:SLURM_QUIESCENT) action=3 ;;
    SLURM_QUIESCENT:DETACHING_SLURMD) action=4 ;;
    DETACHING_SLURMD:K8S_RESERVING) action=5 ;;
    K8S_RESERVING:K8S_OWNED) action=6 ;;
    K8S_OWNED:K8S_RELEASING) action=8 ;;
    K8S_RELEASING:VERIFYING_GPU_CLEAN) action=9 ;;
    VERIFYING_GPU_CLEAN:SLURM_RESTORING) action=10 ;;
    SLURM_RESTORING:SLURM_OWNED) action=12 ;;
    RECOVERY_REQUIRED:SLURM_OWNED) action=13 ;;
    *:RECOVERY_REQUIRED) action=11 ;;
    *) fail "unsupported Lease transition $from -> $to" ;;
  esac
  verify_receipt "$receipt" "$action" "$epoch"
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  [[ "$(jq -r '.spec.holderIdentity' <<<"$lease")" == "$holder" ]] || fail 'Lease holder changed'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key]' <<<"$lease")" == "$epoch" ]] || fail 'Lease epoch changed'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key]' <<<"$lease")" == "$from" ]] || fail 'Lease state changed'
  if [[ "$action" != 11 ]]; then
    lease_is_live "$lease" || fail 'Lease expired before transition CAS'
  fi
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  updated="$(jq --arg now "$now" --arg to "$to" --arg key "$(policy_value "$POLICY" state_annotation)" '
    .spec.renewTime = $now |
    .metadata.annotations[$key] = $to |
    if $to == "SLURM_OWNED" then .spec.holderIdentity = "slurm-owned" else . end
  ' <<<"$lease")"
  replace_lease <<<"$updated"
  if [[ "$action" == 11 ]]; then
    kubectl annotate nodes "$(node0)" "$(node1)" \
      "$(policy_value "$POLICY" epoch_annotation)=$epoch" --overwrite >/dev/null
  fi
  sync_admission_projection "$updated"
}

lease_recovery_acquire() {
  local holder epoch from receipt lease live_holder stored_epoch state next_epoch duration now updated
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  from="$(arg_value --from "$@")"
  receipt="$(arg_value --receipt "$@")"
  verify_receipt "$receipt" 11 "$epoch"
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  live_holder="$(jq -r '.spec.holderIdentity // ""' <<<"$lease")"
  if lease_is_live "$lease" && [[ "$live_holder" != "$holder" ]]; then
    fail "cannot recover a live Lease held by $live_holder"
  fi
  stored_epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // "0"' <<<"$lease")"
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  [[ "$stored_epoch" == "$epoch" ]] || fail 'Lease epoch changed before recovery CAS'
  [[ "$state" == "$from" ]] || fail 'Lease state changed before recovery CAS'
  [[ "$epoch" =~ ^[1-9][0-9]*$ ]] || fail 'stored recovery epoch is invalid'
  next_epoch=$((epoch + 1))
  duration="$(policy_value "$POLICY" lease_duration_seconds)"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  updated="$(jq --arg holder "$holder" --arg now "$now" --arg epoch "$next_epoch" --argjson duration "$duration" \
    --arg epoch_key "$(policy_value "$POLICY" epoch_annotation)" \
    --arg state_key "$(policy_value "$POLICY" state_annotation)" '
      .spec.holderIdentity = $holder |
      .spec.acquireTime = $now |
      .spec.renewTime = $now |
      .spec.leaseDurationSeconds = $duration |
      .spec.leaseTransitions = ((.spec.leaseTransitions // 0) + 1) |
      .metadata.annotations[$epoch_key] = $epoch |
      .metadata.annotations[$state_key] = "RECOVERY_REQUIRED"
    ' <<<"$lease")"
  replace_lease <<<"$updated"
  kubectl annotate nodes "$(node0)" "$(node1)" \
    "$(policy_value "$POLICY" epoch_annotation)=$next_epoch" --overwrite >/dev/null
  sync_admission_projection "$updated"
  printf 'epoch=%s state=RECOVERY_REQUIRED\n' "$next_epoch"
}

lease_bootstrap_recovery_acquire() {
  local holder epoch receipt lease live_holder stored_epoch state next_epoch duration now generation updated
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  verify_receipt "$receipt" 27 "$epoch"
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  if kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" bootstrap_journal)" >/dev/null 2>&1; then
    verify_bootstrap_journal_binding "$(kubectl -n "$(kube_namespace)" get configmap \
      "$(policy_value "$POLICY" bootstrap_journal)" -o json)"
  fi
  live_holder="$(jq -r '.spec.holderIdentity // ""' <<<"$lease")"
  if lease_is_live "$lease" && [[ "$live_holder" != "$holder" ]]; then
    fail "cannot recover a live bootstrap Lease held by $live_holder"
  fi
  stored_epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // "0"' <<<"$lease")"
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  [[ "$stored_epoch" == "$epoch" ]] || fail 'Lease epoch changed before bootstrap recovery CAS'
  [[ "$state" == UNINITIALIZED ]] || fail 'bootstrap recovery requires UNINITIALIZED Lease state'
  [[ "$epoch" =~ ^[1-9][0-9]*$ ]] || fail 'stored bootstrap epoch is invalid'
  next_epoch=$((epoch + 1))
  duration="$(policy_value "$POLICY" lease_duration_seconds)"
  generation="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o jsonpath='{.metadata.generation}')"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  updated="$(jq --arg holder "$holder" --arg now "$now" --arg epoch "$next_epoch" \
    --arg generation "$generation" --argjson duration "$duration" \
    --arg epoch_key "$(policy_value "$POLICY" epoch_annotation)" \
    --arg state_key "$(policy_value "$POLICY" state_annotation)" \
    --arg generation_key "$(policy_value "$POLICY" nodeset_generation_annotation)" '
      .spec.holderIdentity = $holder |
      .spec.acquireTime = $now |
      .spec.renewTime = $now |
      .spec.leaseDurationSeconds = $duration |
      .spec.leaseTransitions = ((.spec.leaseTransitions // 0) + 1) |
      .metadata.annotations[$epoch_key] = $epoch |
      .metadata.annotations[$state_key] = "UNINITIALIZED" |
      .metadata.annotations[$generation_key] = $generation
    ' <<<"$lease")"
  replace_lease <<<"$updated"
  kubectl annotate nodes "$(node0)" "$(node1)" \
    "$(policy_value "$POLICY" epoch_annotation)=$next_epoch" --overwrite >/dev/null
  if kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" admission_configmap)" >/dev/null 2>&1; then
    sync_admission_projection "$updated"
  fi
  ensure_bootstrap_journal BOOTSTRAP_TAKEOVER "$holder" "$receipt"
  printf 'epoch=%s state=UNINITIALIZED\n' "$next_epoch"
}

lease_renew() {
  local holder epoch receipt lease now updated
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  [[ "$(jq -r '.spec.holderIdentity' <<<"$lease")" == "$holder" ]] || fail 'Lease holder changed'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key]' <<<"$lease")" == "$epoch" ]] || fail 'Lease epoch changed'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key]' <<<"$lease")" == K8S_OWNED ]] || fail 'heartbeat outside K8S_OWNED'
  lease_is_live "$lease" || fail 'Lease expired before heartbeat'
  verify_receipt "$receipt" 7 "$epoch"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  updated="$(jq --arg now "$now" '.spec.renewTime = $now' <<<"$lease")"
  replace_lease <<<"$updated"
}

wait_for() {
  local description="$1" probe="$2" holder="${3:-}" epoch="${4:-}" deadline next_renew interval
  deadline=$((SECONDS + $(policy_value "$POLICY" operation_timeout_seconds)))
  interval="$(policy_value "$POLICY" heartbeat_seconds)"
  next_renew=$((SECONDS + interval))
  until "$probe"; do
    (( SECONDS < deadline )) || fail "timeout waiting for $description"
    if [[ -n "$holder" && $SECONDS -ge $next_renew ]]; then
      renew_lease_material "$holder" "$epoch"
      next_renew=$((SECONDS + interval))
    fi
    sleep 2
  done
}

renew_lease_material() {
  local holder="$1" epoch="$2" lease state now updated
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  [[ "$(jq -r '.spec.holderIdentity // ""' <<<"$lease")" == "$holder" ]] || fail 'material keepalive holder mismatch'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$epoch" ]] || fail 'material keepalive epoch mismatch'
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  [[ "$state" != SLURM_OWNED ]] || fail 'material keepalive outside an active transition'
  lease_is_live "$lease" || fail 'Lease expired before material keepalive'
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  updated="$(jq --arg now "$now" '.spec.renewTime = $now' <<<"$lease")"
  replace_lease <<<"$updated"
}

slurm_drained() {
  local nodes jobs steps
  nodes="$(slurm_exec scontrol show node "$(slurm0),$(slurm1)" -o)" || return 1
  jobs="$(slurm_exec squeue -h -w "$(slurm0),$(slurm1)")" || return 1
  steps="$(slurm_exec squeue --steps -h -w "$(slurm0),$(slurm1)")" || return 1
  slurm_states_drained "$nodes" && [[ -z "$jobs" && -z "$steps" ]]
}

drain_slurm() {
  local holder epoch receipt key value effect
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation drain "$holder" "$epoch" "$receipt"
  key="$(policy_value "$POLICY" spark_taint_key)"
  value="$(policy_value "$POLICY" spark_taint_value)"
  effect="$(policy_value "$POLICY" spark_taint_effect)"
  kubectl taint nodes "$(node0)" "$(node1)" "$key=$value:$effect" --overwrite >/dev/null
  slurm_exec scontrol update NodeName="$(slurm0),$(slurm1)" State=DRAIN Reason="pireus-epoch-$epoch" >/dev/null
  wait_for 'both Slurm nodes to drain' slurm_drained "$holder" "$epoch"
}

slurmd_absent() {
  [[ "$(slurmd_pods_json | jq '.items | length')" == 0 ]]
}

detach_slurmd() {
  local holder epoch receipt key
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation detach "$holder" "$epoch" "$receipt"
  key="$(policy_value "$POLICY" slurmd_selector_key)"
  kubectl label nodes "$(node0)" "$(node1)" "${key}-" >/dev/null
  wait_for 'both slurmd pods to disappear' slurmd_absent "$holder" "$epoch"
}

reservation_name() {
  printf '%s-%s\n' "$(policy_value "$POLICY" reservation_prefix)" "$1"
}

create_reservation_pod() {
  local node="$1" suffix="$2" epoch="$3" holder="$4" namespace image taint_key taint_value taint_effect
  namespace="$(kube_namespace)"
  image="$(policy_value "$POLICY" reservation_image)"
  taint_key="$(policy_value "$POLICY" spark_taint_key)"
  taint_value="$(policy_value "$POLICY" spark_taint_value)"
  taint_effect="$(policy_value "$POLICY" spark_taint_effect)"
  kubectl -n "$namespace" apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $(reservation_name "$suffix")
  labels:
    pireus.sounio.dev/spark-pair-reservation: "true"
    $(policy_value "$POLICY" epoch_label): "$epoch"
  annotations:
    $(policy_value "$POLICY" holder_annotation): "$holder"
spec:
  serviceAccountName: $(policy_value "$POLICY" reservation_service_account)
  automountServiceAccountToken: false
  restartPolicy: Never
  hostPID: true
  nodeSelector:
    kubernetes.io/hostname: "$node"
  tolerations:
    - key: "$taint_key"
      operator: Equal
      value: "$taint_value"
      effect: "$taint_effect"
    - key: sounio.dev/arch
      operator: Exists
      effect: NoSchedule
  containers:
    - name: reservation
      image: "$image"
      securityContext:
        privileged: true
      resources:
        requests:
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
      command: [/bin/bash, -lc]
      args:
        - |
          set -euo pipefail
          uuid="\$(chroot /host /usr/bin/nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)"
          processes="\$(chroot /host /usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
          test -z "\${processes//[[:space:]]/}"
          pmon="\$(chroot /host /usr/bin/nvidia-smi pmon -c 1)"
          ! awk 'NF >= 3 && \$2 ~ /^[0-9]+$/ { found=1 } END { exit found ? 0 : 1 }' <<<"\$pmon"
          ! chroot /host /usr/bin/pgrep -f '[n]vidia-cuda-mps' >/dev/null 2>&1
          driver="\$(chroot /host /usr/bin/nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
          product="\$(chroot /host /usr/bin/nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ' ' '_')"
          memory="\$(chroot /host /usr/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)"
          if [[ "\$memory" == '[N/A]' ]]; then memory=UNAVAILABLE_UNIFIED; fi
          utilization="\$(chroot /host /usr/bin/nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1)"
          printf 'PIREUS_NVML_CLEAN node=%s epoch=%s uuid=%s product=%s driver=%s memory_observation=%s utilization_pct=%s\n' \
            "$node" "$epoch" "\$uuid" "\$product" "\$driver" "\$memory" "\$utilization"
          touch /tmp/nvml-clean
          exec sleep infinity
      readinessProbe:
        exec:
          command: [/usr/bin/test, -f, /tmp/nvml-clean]
        periodSeconds: 2
        failureThreshold: 90
      volumeMounts:
        - name: host-root
          mountPath: /host
          readOnly: true
  volumes:
    - name: host-root
      hostPath:
        path: /
        type: Directory
EOF
}

reservations_ready() {
  reservation_pods_json | jq -e --arg n0 "$(node0)" --arg n1 "$(node1)" '
    (.items | length) == 2 and
    ([.items[].spec.nodeName] | sort) == ([$n0, $n1] | sort) and
    all(.items[]; any(.status.conditions[]?; .type == "Ready" and .status == "True"))
  ' >/dev/null
}

record_nvml_values() {
  local epoch="$1" holder="$2" evidence0="$3" evidence1="$4" lease state now hash0 hash1 updated uuid0 uuid1 product0 product1 driver0 driver1 memory0 memory1 utilization0 utilization1
  [[ "$evidence0" == "PIREUS_NVML_CLEAN node=$(node0) epoch=$epoch uuid=GPU-"* ]] || fail '3c59 NVML receipt missing'
  [[ "$evidence1" == "PIREUS_NVML_CLEAN node=$(node1) epoch=$epoch uuid=GPU-"* ]] || fail '8e54 NVML receipt missing'
  hash0="$(printf '%s' "$evidence0" | sha256sum | cut -d ' ' -f 1)"
  hash1="$(printf '%s' "$evidence1" | sha256sum | cut -d ' ' -f 1)"
  uuid0="$(frame_field "$evidence0" uuid)"
  uuid1="$(frame_field "$evidence1" uuid)"
  product0="$(frame_field "$evidence0" product)"
  product1="$(frame_field "$evidence1" product)"
  driver0="$(frame_field "$evidence0" driver)"
  driver1="$(frame_field "$evidence1" driver)"
  memory0="$(frame_field "$evidence0" memory_observation)"
  memory1="$(frame_field "$evidence1" memory_observation)"
  utilization0="$(frame_field "$evidence0" utilization_pct)"
  utilization1="$(frame_field "$evidence1" utilization_pct)"
  [[ "$uuid0" == "$(policy_value "$POLICY" node_0_gpu_uuid)" && \
     "$uuid1" == "$(policy_value "$POLICY" node_1_gpu_uuid)" ]] || fail 'GPU UUID evidence is not the canonical Spark pair'
  [[ "$product0" == "$(policy_value "$POLICY" gpu_product)" && "$product1" == "$product0" ]] || fail 'GPU product evidence is not canonical GB10'
  [[ "$driver0" == "$(policy_value "$POLICY" gpu_driver)" && "$driver1" == "$driver0" ]] || fail 'Spark driver versions differ from policy'
  [[ "$memory0" == "$(policy_value "$POLICY" gpu_memory_observation)" && "$memory1" == "$memory0" ]] || fail 'unified-memory evidence differs from policy'
  [[ "$utilization0" =~ ^[0-9]+$ && "$utilization1" =~ ^[0-9]+$ ]] || fail 'GPU utilization evidence is malformed'
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  [[ "$(jq -r '.spec.holderIdentity' <<<"$lease")" == "$holder" ]] || fail 'Lease holder changed while recording NVML receipts'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$epoch" ]] || \
    fail 'Lease epoch changed while recording NVML receipts'
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  [[ "$state" == K8S_RESERVING || "$state" == K8S_RELEASING || "$state" == RECOVERY_REQUIRED ]] || \
    fail 'NVML receipts are outside an admitted reservation or clean-probe state'
  lease_is_live "$lease" || fail 'Lease expired before recording NVML receipts'
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  updated="$(jq --arg epoch "$epoch" --arg hash0 "$hash0" --arg hash1 "$hash1" --arg now "$now" \
    --arg uuid0 "$uuid0" --arg uuid1 "$uuid1" --arg product "$product0" --arg driver "$driver0" \
    --arg memory0 "$memory0" --arg memory1 "$memory1" \
    --arg utilization0 "$utilization0" --arg utilization1 "$utilization1" '
    .spec.renewTime = $now |
    .metadata.annotations["pireus.sounio.dev/nvml-3c59-epoch"] = $epoch |
    .metadata.annotations["pireus.sounio.dev/nvml-3c59-sha256"] = $hash0 |
    .metadata.annotations["pireus.sounio.dev/gpu-3c59-uuid"] = $uuid0 |
    .metadata.annotations["pireus.sounio.dev/gpu-3c59-memory-observation"] = $memory0 |
    .metadata.annotations["pireus.sounio.dev/gpu-3c59-utilization-pct"] = $utilization0 |
    .metadata.annotations["pireus.sounio.dev/nvml-8e54-epoch"] = $epoch |
    .metadata.annotations["pireus.sounio.dev/nvml-8e54-sha256"] = $hash1 |
    .metadata.annotations["pireus.sounio.dev/gpu-8e54-uuid"] = $uuid1 |
    .metadata.annotations["pireus.sounio.dev/gpu-8e54-memory-observation"] = $memory1 |
    .metadata.annotations["pireus.sounio.dev/gpu-8e54-utilization-pct"] = $utilization1 |
    .metadata.annotations["pireus.sounio.dev/gpu-product"] = $product |
    .metadata.annotations["pireus.sounio.dev/nvidia-driver"] = $driver
  ' <<<"$lease")"
  replace_lease <<<"$updated"
}

record_nvml_receipts() {
  local epoch="$1" holder="$2" log0 log1
  log0="$(kubectl -n "$(kube_namespace)" logs "$(reservation_name 3c59)")"
  log1="$(kubectl -n "$(kube_namespace)" logs "$(reservation_name 8e54)")"
  record_nvml_values "$epoch" "$holder" "$log0" "$log1"
}

create_reservations() {
  local holder epoch receipt
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation reserve "$holder" "$epoch" "$receipt"
  create_reservation_pod "$(node0)" 3c59 "$epoch" "$holder"
  create_reservation_pod "$(node1)" 8e54 "$epoch" "$holder"
  wait_for 'two exact-node GPU reservations' reservations_ready "$holder" "$epoch"
  record_nvml_receipts "$epoch" "$holder"
}

stop_workloads() {
  local holder epoch receipt pods
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation stop "$holder" "$epoch" "$receipt"
  pods="$(workload_pods_json | jq -r '.items[].metadata.name')"
  if [[ -n "$pods" ]]; then
    kubectl -n "$(kube_namespace)" delete pod $pods --wait=false >/dev/null
  fi
  wait_for "epoch $epoch workloads to stop" workloads_absent "$holder" "$epoch"
}

workloads_absent() {
  [[ "$(workload_pods_json | jq '.items | length')" == 0 ]]
}

probe_clean() {
  local epoch holder receipt pod node actual_node output evidence0='' evidence1=''
  epoch="$(arg_value --epoch "$@")"
  holder="$(arg_value --holder "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation probe "$holder" "$epoch" "$receipt"
  [[ "$(reservation_pods_json | jq '.items | length')" == 2 ]] || fail 'fresh clean probe requires both reservations'
  for node in 3c59 8e54; do
    pod="$(reservation_name "$node")"
    if [[ "$node" == 3c59 ]]; then actual_node="$(node0)"; else actual_node="$(node1)"; fi
    output="$(timeout "$(policy_value "$POLICY" heartbeat_seconds)" kubectl -n "$(kube_namespace)" exec "$pod" -- /bin/bash -lc \
      "set -euo pipefail
       processes=\"\$(chroot /host /usr/bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader)\"
       test -z \"\${processes//[[:space:]]/}\"
       pmon=\"\$(chroot /host /usr/bin/nvidia-smi pmon -c 1)\"
       ! awk 'NF >= 3 && \\\$2 ~ /^[0-9]+\\$/ { found=1 } END { exit found ? 0 : 1 }' <<<\"\$pmon\"
       ! chroot /host /usr/bin/pgrep -f '[n]vidia-cuda-mps' >/dev/null 2>&1
       uuid=\"\$(chroot /host /usr/bin/nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)\"
       product=\"\$(chroot /host /usr/bin/nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ' ' '_')\"
       driver=\"\$(chroot /host /usr/bin/nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)\"
       memory=\"\$(chroot /host /usr/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)\"
       if [[ \"\$memory\" == '[N/A]' ]]; then memory=UNAVAILABLE_UNIFIED; fi
       utilization=\"\$(chroot /host /usr/bin/nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1)\"
       printf 'PIREUS_NVML_CLEAN node=%s epoch=%s uuid=%s product=%s driver=%s memory_observation=%s utilization_pct=%s\\n' \
         '$actual_node' '$epoch' \"\$uuid\" \"\$product\" \"\$driver\" \"\$memory\" \"\$utilization\"" \
      )"
    if [[ "$node" == 3c59 ]]; then
      evidence0="$output"
    else
      evidence1="$output"
    fi
  done
  record_nvml_values "$epoch" "$holder" "$evidence0" "$evidence1"
}

delete_reservations() {
  local holder epoch receipt count
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation delete "$holder" "$epoch" "$receipt"
  count="$(reservation_pods_json | jq '.items | length')"
  if [[ "$count" != 0 ]]; then
    kubectl -n "$(kube_namespace)" delete pods \
      -l 'pireus.sounio.dev/spark-pair-reservation=true' --wait=false >/dev/null
  fi
  wait_for "epoch $epoch reservations to terminate" reservations_absent "$holder" "$epoch"
}

reservations_absent() {
  [[ "$(reservation_pods_json | jq '.items | length')" == 0 ]]
}

slurmd_bound() {
  slurmd_pods_json | jq -e --arg n0 "$(node0)" --arg n1 "$(node1)" '
    (.items | length) == 2 and
    ([.items[].spec.nodeName] | sort) == ([$n0, $n1] | sort) and
    all(.items[];
      .status.phase == "Running" and
      any(.status.conditions[]?; .type == "Ready" and .status == "True") and
      .spec.containers[0].resources.requests["nvidia.com/gpu"] == "1" and
      .spec.containers[0].resources.limits["nvidia.com/gpu"] == "1")
  ' >/dev/null
}

restore_slurmd() {
  local holder epoch receipt key value
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation restore "$holder" "$epoch" "$receipt"
  key="$(policy_value "$POLICY" slurmd_selector_key)"
  value="$(policy_value "$POLICY" slurmd_selector_value)"
  kubectl label nodes "$(node0)" "$(node1)" "$key=$value" --overwrite >/dev/null
  wait_for 'both GPU-bound slurmd pods' slurmd_bound "$holder" "$epoch"
}

slurm_resumed() {
  local nodes jobs steps
  nodes="$(slurm_exec scontrol show node "$(slurm0),$(slurm1)" -o)" || return 1
  jobs="$(slurm_exec squeue -h -w "$(slurm0),$(slurm1)")" || return 1
  steps="$(slurm_exec squeue --steps -h -w "$(slurm0),$(slurm1)")" || return 1
  slurm_states_resumed "$nodes" &&
    [[ "$(grep -c 'CPUAlloc=0' <<<"$nodes")" == 2 && -z "$jobs" && -z "$steps" ]]
}

verify_bootstrap_lease_receipt_context() {
  local holder="$1" receipt="$2" lease receipt_epoch expected_epoch action
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  lease_is_live "$lease" || fail 'Lease expired before bootstrap journal update'
  [[ "$(jq -r '.spec.holderIdentity // ""' <<<"$lease")" == "$holder" ]] || \
    fail 'Lease holder changed before bootstrap journal update'
  receipt_epoch="$(receipt_value "$receipt" epoch)"
  action="$(receipt_value "$receipt" action_code)"
  expected_epoch="$receipt_epoch"
  [[ "$action" != 27 ]] || expected_epoch=$((receipt_epoch + 1))
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")" == "$expected_epoch" ]] || \
    fail 'Lease epoch changed before bootstrap journal update'
}

bootstrap_journal_step() {
  local step="$1" holder="$2" receipt="$3" journal updated now receipt_hash
  journal="$(kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" bootstrap_journal)" -o json)"
  verify_bootstrap_journal_binding "$journal"
  verify_bootstrap_lease_receipt_context "$holder" "$receipt"
  receipt_hash="$(sha256_file "$receipt")"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  updated="$(jq --arg step "$step" --arg now "$now" --arg holder "$holder" --arg receipt "$receipt_hash" '
    .data.step = $step | .data.updatedUtc = $now | .data.holder = $holder |
    .data.lastReceiptSha256 = $receipt
  ' <<<"$journal")"
  kubectl -n "$(kube_namespace)" replace -f - <<<"$updated" >/dev/null
}

ensure_bootstrap_journal() {
  local step="$1" holder="$2" receipt="$3" now receipt_hash source_hash freeze_hash
  verify_bootstrap_lease_receipt_context "$holder" "$receipt"
  if kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" bootstrap_journal)" >/dev/null 2>&1; then
    bootstrap_journal_step "$step" "$holder" "$receipt"
    return 0
  fi
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  receipt_hash="$(sha256_file "$receipt")"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  kubectl -n "$(kube_namespace)" create configmap "$(policy_value "$POLICY" bootstrap_journal)" \
    --from-literal=step="$step" \
    --from-literal=holder="$holder" \
    --from-literal=createdUtc="$now" \
    --from-literal=updatedUtc="$now" \
    --from-literal=lastReceiptSha256="$receipt_hash" \
    --from-literal=sounioSourceSha256="$source_hash" \
    --from-literal=semanticsFreezeSha256="$freeze_hash" >/dev/null
}

admission_current() {
  admission_fail_closed "$(lease_json)"
}

bootstrap_gpu_admission_denied() {
  local output status
  set +e
  output="$(kubectl create --dry-run=server -f - 2>&1 <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: pireus-bootstrap-gpu-deny-probe
  namespace: default
spec:
  restartPolicy: Never
  containers:
    - name: probe
      image: $(policy_value "$POLICY" reservation_image)
      resources:
        limits:
          nvidia.com/gpu: "1"
EOF
)"
  status=$?
  set -e
  [[ $status -ne 0 && ( "$output" == *'Spark GPU Pods require the current Pireus Lease epoch'* || \
    "$output" == *'pireus-spark-pair-fence'* ) ]]
}

install_fence() {
  local holder epoch receipt manifest plugin_name plugin patched key value effect pods
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation fence "$holder" "$epoch" "$receipt"
  manifest="$(repo_root)/$(policy_value "$POLICY" admission_manifest)"
  kubectl apply --server-side --field-manager=pireus-spark-pair -f "$manifest" >/dev/null
  sync_admission_projection
  wait_for 'fail-closed Spark admission projection' admission_current "$holder" "$epoch"
  bootstrap_gpu_admission_denied || fail 'generic GPU Pod was not denied during UNINITIALIZED bootstrap'

  key="$(policy_value "$POLICY" spark_taint_key)"
  value="$(policy_value "$POLICY" spark_taint_value)"
  effect="$(policy_value "$POLICY" spark_taint_effect)"
  kubectl taint nodes "$(node0)" "$(node1)" "$key=$value:$effect" --overwrite >/dev/null
  for plugin_name in "$(policy_value "$POLICY" device_plugin_0_name)" \
    "$(policy_value "$POLICY" device_plugin_1_name)"; do
    plugin="$(kubectl -n kube-system get daemonset "$plugin_name" -o json)"
    patched="$(jq --arg key "$key" --arg value "$value" --arg effect "$effect" '
      if any(.spec.template.spec.tolerations[]?;
        .key == $key and .value == $value and .effect == $effect)
      then .
      else .spec.template.spec.tolerations += [{
        "key": $key, "operator": "Equal", "value": $value, "effect": $effect
      }]
      end
    ' <<<"$plugin")"
    kubectl -n kube-system replace -f - <<<"$patched" >/dev/null
    kubectl -n kube-system rollout status daemonset/"$plugin_name" \
      --timeout="$(policy_value "$POLICY" operation_timeout_seconds)s" >/dev/null
    renew_lease_material "$holder" "$epoch"
  done
  pods="$(kubectl get pods -A -o json)"
  unexpected_gpu_consumers_zero "$pods" "$epoch" "$holder" || \
    fail 'unexpected Pod remains on the fenced Spark pair'
  bootstrap_journal_step FENCE_INSTALLED "$holder" "$receipt"
}

wait_nodeset_observed() {
  local generation observed
  generation="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o jsonpath='{.metadata.generation}')"
  observed="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o jsonpath='{.status.observedGeneration}')"
  [[ "$generation" == "$observed" ]]
}

refresh_nodeset_generation() {
  local holder="$1" epoch="$2" lease generation now updated
  lease="$(lease_json)"
  verify_lease_freeze_binding "$lease"
  [[ "$(jq -r '.spec.holderIdentity // ""' <<<"$lease")" == "$holder" ]] || fail 'NodeSet generation refresh holder mismatch'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$epoch" ]] || fail 'NodeSet generation refresh epoch mismatch'
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == UNINITIALIZED ]] || fail 'NodeSet generation refresh outside bootstrap'
  lease_is_live "$lease" || fail 'Lease expired before NodeSet generation refresh'
  generation="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o jsonpath='{.metadata.generation}')"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  updated="$(jq --arg generation "$generation" --arg now "$now" \
    --arg key "$(policy_value "$POLICY" nodeset_generation_annotation)" '
      .metadata.annotations[$key] = $generation |
      .spec.renewTime = $now
    ' <<<"$lease")"
  replace_lease <<<"$updated"
  sync_admission_projection "$updated"
}

install_gpu_bound_slurmd() {
  local holder epoch receipt nodeset patched selector_key selector_value taint_key taint_value taint_effect gpu
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation bootstrap-slurmd "$holder" "$epoch" "$receipt"
  selector_key="$(policy_value "$POLICY" slurmd_selector_key)"
  selector_value="$(policy_value "$POLICY" slurmd_selector_value)"
  taint_key="$(policy_value "$POLICY" spark_taint_key)"
  taint_value="$(policy_value "$POLICY" spark_taint_value)"
  taint_effect="$(policy_value "$POLICY" spark_taint_effect)"
  gpu="$(policy_value "$POLICY" slurmd_gpu_resource)"
  nodeset="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o json)"
  patched="$(jq --arg selector_key "$selector_key" --arg selector_value "$selector_value" \
    --arg taint_key "$taint_key" --arg taint_value "$taint_value" --arg taint_effect "$taint_effect" \
    --arg gpu "$gpu" '
      .spec.template.spec.nodeSelector[$selector_key] = $selector_value |
      .spec.slurmd.resources.requests[$gpu] = "1" |
      .spec.slurmd.resources.limits[$gpu] = "1" |
      if any(.spec.template.spec.tolerations[]?;
        .key == $taint_key and .value == $taint_value and .effect == $taint_effect)
      then .
      else .spec.template.spec.tolerations += [{
        "key": $taint_key, "operator": "Equal", "value": $taint_value, "effect": $taint_effect
      }]
      end
    ' <<<"$nodeset")"
  kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" replace -f - <<<"$patched" >/dev/null
  wait_for 'NodeSet controller observation' wait_nodeset_observed "$holder" "$epoch"
  refresh_nodeset_generation "$holder" "$epoch"
  wait_for 'legacy non-GPU-accounted slurmd pods to terminate' slurmd_absent "$holder" "$epoch"
  kubectl label nodes "$(node0)" "$(node1)" "$selector_key=$selector_value" --overwrite >/dev/null
  wait_for 'both GPU-bound slurmd pods' slurmd_bound "$holder" "$epoch"
  bootstrap_journal_step SLURMD_GPU_BOUND "$holder" "$receipt"
}

resume_slurm() {
  local holder epoch receipt bootstrap
  bootstrap=''
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation resume "$holder" "$epoch" "$receipt"
  slurm_exec scontrol update NodeName="$(slurm0),$(slurm1)" State=RESUME >/dev/null
  wait_for 'both Slurm nodes to resume idle' slurm_resumed "$holder" "$epoch"
  if [[ "$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" \
    '.metadata.annotations[$key]' <<<"$(lease_json)")" == UNINITIALIZED ]]; then
    bootstrap_journal_step SLURM_RESUMED "$holder" "$receipt"
  fi
}

bootstrap_lease() {
  local namespace name duration generation source_hash freeze_hash now holder epoch receipt
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  [[ "$epoch" == 1 ]] || fail 'initial bootstrap epoch must be one'
  verify_receipt "$receipt" 28 "$epoch"
  namespace="$(kube_namespace)"
  name="$(lease_name)"
  kubectl -n "$namespace" get lease "$name" >/dev/null 2>&1 && fail 'arbiter Lease already exists'
  kubectl -n "$namespace" get configmap "$(policy_value "$POLICY" bootstrap_journal)" >/dev/null 2>&1 && \
    fail 'bootstrap journal already exists'
  duration="$(policy_value "$POLICY" lease_duration_seconds)"
  generation="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o jsonpath='{.metadata.generation}')"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256sum "$FREEZE" | cut -d ' ' -f 1)"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  kubectl -n "$namespace" create -f - >/dev/null <<EOF
apiVersion: coordination.k8s.io/v1
kind: Lease
metadata:
  name: $name
  annotations:
    $(policy_value "$POLICY" state_annotation): UNINITIALIZED
    $(policy_value "$POLICY" epoch_annotation): "1"
    $(policy_value "$POLICY" nodeset_generation_annotation): "$generation"
    $(policy_value "$POLICY" source_hash_annotation): "$source_hash"
    $(policy_value "$POLICY" freeze_hash_annotation): "$freeze_hash"
spec:
  holderIdentity: $holder
  leaseDurationSeconds: $duration
  acquireTime: "$now"
  renewTime: "$now"
  leaseTransitions: 1
EOF
  kubectl annotate nodes "$(node0)" "$(node1)" \
    "$(policy_value "$POLICY" epoch_annotation)=1" --overwrite >/dev/null
  ensure_bootstrap_journal LEASE_INITIALIZED "$holder" "$receipt"
  printf 'epoch=1 state=UNINITIALIZED\n'
}

main() {
  [[ "${1:-}" == --policy && $# -ge 5 ]] || fail 'expected --policy FILE --freeze FILE COMMAND'
  POLICY="$2"
  shift 2
  [[ "${1:-}" == --freeze ]] || fail 'expected --freeze FILE'
  FREEZE="$2"
  shift 2
  [[ "$(realpath "$POLICY")" == "$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1" ]] || \
    fail 'real backend requires the canonical material policy'
  [[ "$(realpath "$FREEZE")" == "$ROOT_DIR/tools/cluster/spark_pair_arbiter.freeze.v1" ]] || \
    fail 'real backend requires the canonical semantics freeze'
  verify_frozen_material
  local command="${1:-}"
  shift || true
  case "$command" in
    prebootstrap-facts) prebootstrap_facts "$@" ;;
    facts) facts "$@" ;;
    lease-acquire) lease_acquire "$@" ;;
    lease-recovery-acquire) lease_recovery_acquire "$@" ;;
    lease-bootstrap-recovery-acquire) lease_bootstrap_recovery_acquire "$@" ;;
    lease-transition) lease_transition "$@" ;;
    lease-renew) lease_renew "$@" ;;
    drain-slurm) drain_slurm "$@" ;;
    install-fence) install_fence "$@" ;;
    install-gpu-bound-slurmd) install_gpu_bound_slurmd "$@" ;;
    detach-slurmd) detach_slurmd "$@" ;;
    create-reservations) create_reservations "$@" ;;
    stop-workloads) stop_workloads "$@" ;;
    probe-clean) probe_clean "$@" ;;
    delete-reservations) delete_reservations "$@" ;;
    restore-slurmd) restore_slurmd "$@" ;;
    resume-slurm) resume_slurm "$@" ;;
    bootstrap-lease) bootstrap_lease "$@" ;;
    *) fail "unsupported backend command: $command" ;;
  esac
}

main "$@"
