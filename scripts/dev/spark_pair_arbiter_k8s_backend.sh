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

lease_timestamp() {
  date -u +%Y-%m-%dT%H:%M:%S.000000Z
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
  verify_frozen_file host_fence_manifest_source host_fence_manifest_sha256
  verify_frozen_file device_barrier_source device_barrier_source_sha256
  verify_frozen_file dgx_material_slurm_source dgx_material_slurm_sha256
  verify_frozen_file dgx_material_cuda_source dgx_material_cuda_sha256
  verify_frozen_file dgx_material_header_source dgx_material_header_sha256
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
  [[ "$(receipt_value "$receipt" host_fence_manifest_sha256)" == "$(policy_value "$FREEZE" host_fence_manifest_sha256)" ]] || fail 'receipt host fence manifest hash mismatch'
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
    29) expected_from="$receipt_from"; expected_to="$receipt_from" ;;
    30|31|32) expected_from="$receipt_from"; expected_to="$receipt_from" ;;
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
    [[ "$base" == IDLE || "$base" == DOWN ]] || return 1
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
  state="$(require_lease_context "$holder" "$epoch" 'UNINITIALIZED SLURM_OWNED DRAINING_SLURM SLURM_QUIESCENT DETACHING_SLURMD K8S_RESERVING K8S_OWNED K8S_RELEASING VERIFYING_GPU_CLEAN SLURM_RESTORING RECOVERY_REQUIRED')"
  case "$kind:$state" in
    drain:UNINITIALIZED) actions=23 ;;
    fence:UNINITIALIZED) actions=24 ;;
    host-fence:UNINITIALIZED) actions=29 ;;
    host-fence:RECOVERY_REQUIRED) actions=29 ;;
    host-pair:UNINITIALIZED) actions=30 ;;
    host-pair:DRAINING_SLURM) actions=30 ;;
    host-pair:SLURM_QUIESCENT) actions=30 ;;
    host-pair:K8S_RELEASING) actions=30 ;;
    host-pair:VERIFYING_GPU_CLEAN) actions=30 ;;
    host-pair:RECOVERY_REQUIRED) actions=30 ;;
    host-grant-slurm:UNINITIALIZED) actions=31 ;;
    host-grant-slurm:SLURM_RESTORING) actions=31 ;;
    host-grant-slurm:RECOVERY_REQUIRED) actions=31 ;;
    host-grant-k8s:K8S_RESERVING) actions=32 ;;
    host-grant-k8s:K8S_OWNED) actions=32 ;;
    keepalive:UNINITIALIZED) actions='23 24 25 26 29 30 31' ;;
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
  kubectl get "$(policy_value "$POLICY" admission_parameter_resource)" \
    "$(policy_value "$POLICY" admission_parameter)" -o json
}

parameter_crd_manifest() {
  cat <<EOF
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: $(policy_value "$POLICY" admission_parameter_crd)
spec:
  group: pireus.sounio.dev
  scope: Cluster
  names:
    plural: $(policy_value "$POLICY" admission_parameter_resource)
    singular: pireussparkpairparameter
    kind: PireusSparkPairParameter
  versions:
    - name: v1alpha1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          required: [data]
          properties:
            data:
              type: object
              required: [state, epoch, holder, allowWorkload, sounioSourceSha256, semanticsFreezeSha256, hostFenceDaemonSetUid]
              properties:
                state: {type: string}
                epoch: {type: string}
                holder: {type: string}
                allowWorkload: {type: string}
                sounioSourceSha256: {type: string}
                semanticsFreezeSha256: {type: string}
                hostFenceDaemonSetUid: {type: string}
EOF
}

vap_typechecking_acceptable() {
  jq -e '
    (.status.observedGeneration == .metadata.generation) and
    ((.status.typeChecking.expressionWarnings // []) | length) <= 2 and
    all(.status.typeChecking.expressionWarnings[]?;
      (.fieldRef == "spec.validations[0].expression" or
       .fieldRef == "spec.validations[2].expression") and
      (.warning | type) == "string" and (.warning | length) > 0)
  ' <<<"$1" >/dev/null
}

host_vap_typechecking_acceptable() {
  jq -e '
    (.status.observedGeneration == .metadata.generation) and
    ((.status.typeChecking.expressionWarnings // []) | length) == 0
  ' <<<"$1" >/dev/null
}

sync_admission_projection() {
  local lease="${1:-}" config state epoch holder source_hash freeze_hash updated
  local daemonset daemonset_uid=UNBOUND
  [[ -n "$lease" ]] || lease="$(lease_json)"
  config="$(admission_config_json)"
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  holder="$(jq -r '.spec.holderIdentity // ""' <<<"$lease")"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  if daemonset="$(kubectl -n "$(kube_namespace)" get daemonset \
    "$(policy_value "$POLICY" host_fence_daemonset)" -o json 2>/dev/null)"; then
    daemonset_uid="$(jq -r '.metadata.uid // ""' <<<"$daemonset")"
    [[ "$daemonset_uid" =~ ^[0-9a-f-]{36}$ ]] || return 1
  fi
  updated="$(jq --arg state "$state" --arg epoch "$epoch" --arg holder "$holder" \
    --arg source "$source_hash" --arg freeze "$freeze_hash" --arg uid "$daemonset_uid" '
      .data.state = $state |
      .data.epoch = $epoch |
      .data.holder = $holder |
      .data.allowWorkload = "false" |
      .data.sounioSourceSha256 = $source |
      .data.semanticsFreezeSha256 = $freeze |
      .data.hostFenceDaemonSetUid = $uid
    ' <<<"$config")"
  kubectl replace -f - <<<"$updated" >/dev/null
}

admission_fail_closed() {
  local lease policy binding host_policy host_binding binding_policy binding_binding control_policy control_binding config probe_config parameter_crd expected_parameter_crd state epoch holder source_hash freeze_hash
  local bound_uid daemonset daemonset_uid expected expected_policy expected_binding expected_host_policy expected_host_binding expected_binding_policy expected_binding_binding expected_control_policy expected_control_binding expected_probe_config expected_probe_sha manifest
  lease="$1"
  policy="$(kubectl get validatingadmissionpolicy "$(policy_value "$POLICY" admission_policy)" -o json 2>/dev/null)" || return 1
  binding="$(kubectl get validatingadmissionpolicybinding "$(policy_value "$POLICY" admission_binding)" -o json 2>/dev/null)" || return 1
  host_policy="$(kubectl get validatingadmissionpolicy "$(policy_value "$POLICY" admission_host_policy)" -o json 2>/dev/null)" || return 1
  host_binding="$(kubectl get validatingadmissionpolicybinding "$(policy_value "$POLICY" admission_host_binding)" -o json 2>/dev/null)" || return 1
  binding_policy="$(kubectl get validatingadmissionpolicy "$(policy_value "$POLICY" admission_binding_policy)" -o json 2>/dev/null)" || return 1
  binding_binding="$(kubectl get validatingadmissionpolicybinding "$(policy_value "$POLICY" admission_binding_binding)" -o json 2>/dev/null)" || return 1
  control_policy="$(kubectl get validatingadmissionpolicy "$(policy_value "$POLICY" admission_control_policy)" -o json 2>/dev/null)" || return 1
  control_binding="$(kubectl get validatingadmissionpolicybinding "$(policy_value "$POLICY" admission_control_binding)" -o json 2>/dev/null)" || return 1
  config="$(admission_config_json 2>/dev/null)" || return 1
  probe_config="$(kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" reservation_probe_configmap)" -o json 2>/dev/null)" || return 1
  parameter_crd="$(kubectl get crd "$(policy_value "$POLICY" admission_parameter_crd)" -o json 2>/dev/null)" || return 1
  expected_parameter_crd="$(parameter_crd_manifest | kubectl apply --dry-run=server -f - -o json 2>/dev/null)" || return 1
  manifest="$(repo_root)/$(policy_value "$POLICY" admission_manifest)"
  expected="$(kubectl apply --dry-run=server -f "$manifest" -o json 2>/dev/null)" || return 1
  expected_policy="$(jq -c --arg name "$(policy_value "$POLICY" admission_policy)" \
    '.items[] | select(.kind == "ValidatingAdmissionPolicy" and .metadata.name == $name)' <<<"$expected")"
  expected_binding="$(jq -c --arg name "$(policy_value "$POLICY" admission_binding)" \
    '.items[] | select(.kind == "ValidatingAdmissionPolicyBinding" and .metadata.name == $name)' <<<"$expected")"
  expected_host_policy="$(jq -c --arg name "$(policy_value "$POLICY" admission_host_policy)" \
    '.items[] | select(.kind == "ValidatingAdmissionPolicy" and .metadata.name == $name)' <<<"$expected")"
  expected_host_binding="$(jq -c --arg name "$(policy_value "$POLICY" admission_host_binding)" \
    '.items[] | select(.kind == "ValidatingAdmissionPolicyBinding" and .metadata.name == $name)' <<<"$expected")"
  expected_binding_policy="$(jq -c --arg name "$(policy_value "$POLICY" admission_binding_policy)" \
    '.items[] | select(.kind == "ValidatingAdmissionPolicy" and .metadata.name == $name)' <<<"$expected")"
  expected_binding_binding="$(jq -c --arg name "$(policy_value "$POLICY" admission_binding_binding)" \
    '.items[] | select(.kind == "ValidatingAdmissionPolicyBinding" and .metadata.name == $name)' <<<"$expected")"
  expected_control_policy="$(jq -c --arg name "$(policy_value "$POLICY" admission_control_policy)" \
    '.items[] | select(.kind == "ValidatingAdmissionPolicy" and .metadata.name == $name)' <<<"$expected")"
  expected_control_binding="$(jq -c --arg name "$(policy_value "$POLICY" admission_control_binding)" \
    '.items[] | select(.kind == "ValidatingAdmissionPolicyBinding" and .metadata.name == $name)' <<<"$expected")"
  expected_probe_config="$(jq -c --arg name "$(policy_value "$POLICY" reservation_probe_configmap)" \
    '.items[] | select(.kind == "ConfigMap" and .metadata.name == $name)' <<<"$expected")"
  [[ -n "$expected_policy" && -n "$expected_binding" && -n "$expected_host_policy" &&
      -n "$expected_host_binding" && -n "$expected_binding_policy" &&
      -n "$expected_binding_binding" && -n "$expected_control_policy" &&
      -n "$expected_control_binding" && -n "$expected_probe_config" ]] || return 1
  expected_probe_sha="$(jq -j '.data["reservation-probe.sh"] // ""' \
    <<<"$expected_probe_config" | sha256sum | cut -d ' ' -f 1)"
  [[ "$(policy_value "$POLICY" reservation_probe_configmap)" == \
      "pireus-spark-pair-reservation-probe-${expected_probe_sha:0:12}" ]] || return 1
  [[ "$(jq -S -c '.spec' <<<"$policy")" == "$(jq -S -c '.spec' <<<"$expected_policy")" &&
      "$(jq -S -c '.spec' <<<"$binding")" == "$(jq -S -c '.spec' <<<"$expected_binding")" &&
      "$(jq -S -c '.spec' <<<"$host_policy")" == "$(jq -S -c '.spec' <<<"$expected_host_policy")" &&
      "$(jq -S -c '.spec' <<<"$host_binding")" == "$(jq -S -c '.spec' <<<"$expected_host_binding")" &&
      "$(jq -S -c '.spec' <<<"$binding_policy")" == "$(jq -S -c '.spec' <<<"$expected_binding_policy")" &&
      "$(jq -S -c '.spec' <<<"$binding_binding")" == "$(jq -S -c '.spec' <<<"$expected_binding_binding")" &&
      "$(jq -S -c '.spec' <<<"$control_policy")" == "$(jq -S -c '.spec' <<<"$expected_control_policy")" &&
      "$(jq -S -c '.spec' <<<"$control_binding")" == "$(jq -S -c '.spec' <<<"$expected_control_binding")" ]] || return 1
  [[ "$(jq -S -c '.spec' <<<"$parameter_crd")" == \
      "$(jq -S -c '.spec' <<<"$expected_parameter_crd")" ]] || return 1
  [[ "$(jq -S -c '{immutable,data}' <<<"$probe_config")" == \
      "$(jq -S -c '{immutable,data}' <<<"$expected_probe_config")" ]] || return 1
  state="$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  epoch="$(jq -r --arg key "$(policy_value "$POLICY" epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  holder="$(jq -r '.spec.holderIdentity // ""' <<<"$lease")"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  bound_uid="$(jq -r '.data.hostFenceDaemonSetUid // ""' <<<"$config")"
  if [[ "$bound_uid" == UNBOUND ]]; then
    if kubectl -n "$(kube_namespace)" get daemonset \
      "$(policy_value "$POLICY" host_fence_daemonset)" >/dev/null 2>&1; then
      host_fence_staged || return 1
    fi
  else
    [[ "$bound_uid" =~ ^[0-9a-f-]{36}$ ]] || return 1
    daemonset="$(kubectl -n "$(kube_namespace)" get daemonset \
      "$(policy_value "$POLICY" host_fence_daemonset)" -o json 2>/dev/null)" || return 1
    daemonset_uid="$(jq -r '.metadata.uid // ""' <<<"$daemonset")"
    [[ "$daemonset_uid" == "$bound_uid" ]] || return 1
  fi
  jq -e '
    .spec.failurePolicy == "Fail" and
    .spec.paramKind.apiVersion == "pireus.sounio.dev/v1alpha1" and
    .spec.paramKind.kind == "PireusSparkPairParameter" and
    (.spec.validations | length) >= 1
  ' <<<"$policy" >/dev/null || return 1
  vap_typechecking_acceptable "$policy" || return 1
  jq -e --arg name "$(policy_value "$POLICY" admission_parameter)" '
    .spec.paramRef.name == $name and (.spec.paramRef | has("namespace") | not) and
    .spec.paramRef.parameterNotFoundAction == "Deny" and
    (.spec.validationActions == ["Deny"])
  ' <<<"$binding" >/dev/null || return 1
  jq -e '
    .spec.failurePolicy == "Fail" and
    .spec.paramKind.apiVersion == "pireus.sounio.dev/v1alpha1" and
    .spec.paramKind.kind == "PireusSparkPairParameter" and
    (.spec.validations | length) == 12
  ' <<<"$host_policy" >/dev/null || return 1
  host_vap_typechecking_acceptable "$host_policy" || return 1
  jq -e --arg name "$(policy_value "$POLICY" admission_parameter)" \
    --arg policy "$(policy_value "$POLICY" admission_host_policy)" '
    .spec.policyName == $policy and
    .spec.paramRef.name == $name and (.spec.paramRef | has("namespace") | not) and
    .spec.paramRef.parameterNotFoundAction == "Deny" and
    (.spec.validationActions == ["Deny"])
  ' <<<"$host_binding" >/dev/null || return 1
  jq -e '
    .spec.failurePolicy == "Fail" and
    (.spec.validations | length) == 1
  ' <<<"$binding_policy" >/dev/null || return 1
  jq -e '
    (.status.observedGeneration == .metadata.generation) and
    ((.status.typeChecking.expressionWarnings // []) | length) == 0
  ' <<<"$binding_policy" >/dev/null || return 1
  jq -e --arg policy "$(policy_value "$POLICY" admission_binding_policy)" '
    .spec.policyName == $policy and .spec.validationActions == ["Deny"]
  ' <<<"$binding_binding" >/dev/null || return 1
  jq -e '
    .spec.failurePolicy == "Fail" and
    (.spec.validations | length) == 1
  ' <<<"$control_policy" >/dev/null || return 1
  vap_typechecking_acceptable "$control_policy" || return 1
  jq -e --arg policy "$(policy_value "$POLICY" admission_control_policy)" '
    .spec.policyName == $policy and .spec.validationActions == ["Deny"]
  ' <<<"$control_binding" >/dev/null || return 1
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
  local pods="$1" epoch="$2" holder="$3" host_fence_uid=''
  host_fence_uid="$(admission_config_json 2>/dev/null | jq -r '.data.hostFenceDaemonSetUid // ""')" || \
    host_fence_uid=''
  jq -e --arg n0 "$(node0)" --arg n1 "$(node1)" --arg epoch "$epoch" --arg holder "$holder" \
    --arg host_fence_uid "$host_fence_uid" '
    [.items[] | select(.spec.nodeName == $n0 or .spec.nodeName == $n1) | select((
      (.metadata.namespace == "slurm-pilot" and
       .metadata.labels["app.kubernetes.io/name"] == "slurmd" and
       .metadata.labels["app.kubernetes.io/instance"] == "slurm-pilot-worker-spark") or
      (.metadata.namespace == "beagle" and
       .metadata.labels["pireus.sounio.dev/spark-pair-reservation"] == "true" and
       .metadata.labels["pireus.sounio.dev/spark-pair-epoch"] == $epoch and
       .metadata.annotations["pireus.sounio.dev/spark-pair-holder"] == $holder) or
      (.metadata.namespace == "beagle" and
       .spec.serviceAccountName == "pireus-spark-host-fence" and
       .metadata.labels["pireus.sounio.dev/spark-pair-infrastructure"] == "true" and
       $host_fence_uid != "" and
       (.metadata.ownerReferences | length) == 1 and
       .metadata.ownerReferences[0].apiVersion == "apps/v1" and
       .metadata.ownerReferences[0].kind == "DaemonSet" and
       .metadata.ownerReferences[0].name == "pireus-spark-host-fence" and
       .metadata.ownerReferences[0].uid == $host_fence_uid and
       .metadata.ownerReferences[0].controller == true and
       .metadata.ownerReferences[0].blockOwnerDeletion == true) or
      (.metadata.namespace == "kube-system") or
      (.metadata.namespace == "ceph-csi-cephfs") or
      (.metadata.namespace == "ceph-csi-rbd") or
      (.metadata.namespace == "nvidia-network-operator") or
      (.metadata.namespace == "darwin-observability-system")
    ) | not)] | length == 0
  ' <<<"$pods" >/dev/null
}

lease_is_live() {
  local json="$1"
  jq -e '(.spec.renewTime | sub("\\.[0-9]+Z$"; "Z") | fromdateiso8601) + .spec.leaseDurationSeconds > now' \
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

bootstrap_freeze_migration_context() {
  local lease="$1" journal="$2" slurm_nodes="$3" slurm_jobs="$4" slurm_steps="$5"
  local report0="$6" report1="$7" reservation_count="$8" workload_count="$9"
  local source_hash current_freeze lease_source journal_source lease_freeze journal_freeze
  local old_freeze='' prior_freeze migration_ancestor binding report report_freeze expected_node
  local journal_step expected_mode expected_valid
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  current_freeze="$(sha256_file "$FREEZE")"
  lease_source="$(jq -r --arg key "$(policy_value "$POLICY" source_hash_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")"
  journal_source="$(jq -r '.data.sounioSourceSha256 // ""' <<<"$journal")"
  lease_freeze="$(jq -r --arg key "$(policy_value "$POLICY" freeze_hash_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")"
  journal_freeze="$(jq -r '.data.semanticsFreezeSha256 // ""' <<<"$journal")"
  prior_freeze="$(jq -r '.data.migrationFromFreezeSha256 // ""' <<<"$journal")"
  migration_ancestor="$(policy_value "$FREEZE" bootstrap_migration_ancestor_sha256)"

  [[ "$lease_source" == "$source_hash" && "$journal_source" == "$source_hash" ]] || return 1
  [[ "$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")" == UNINITIALIZED ]] || return 1
  lease_is_live "$lease" && return 1
  journal_step="$(jq -r '.data.step // ""' <<<"$journal")"
  case "$journal_step" in
    FENCE_INSTALLED|HOST_FENCE_INSTALLED|BOOTSTRAP_TAKEOVER)
      expected_mode=FENCED
      expected_valid=0
      ;;
    SLURMD_GPU_BOUND|SLURM_RESUMED)
      expected_mode=SLURM
      expected_valid=1
      ;;
    *) return 1 ;;
  esac
  for binding in "$lease_freeze" "$journal_freeze"; do
    [[ "$binding" =~ ^[0-9a-f]{64}$ ]] || return 1
    if [[ "$binding" != "$current_freeze" ]]; then
      if [[ -z "$old_freeze" ]]; then
        old_freeze="$binding"
      else
        [[ "$binding" == "$old_freeze" ]] || return 1
      fi
    fi
  done
  [[ -n "$old_freeze" ]] || return 1
  [[ -z "$prior_freeze" || "$prior_freeze" =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ "$migration_ancestor" =~ ^[0-9a-f]{64}$ ]] || return 1

  slurm_states_drained "$slurm_nodes" || return 1
  [[ -z "$slurm_jobs" && -z "$slurm_steps" ]] || return 1
  [[ "$(grep -c 'CPUAlloc=0' <<<"$slurm_nodes")" == 2 &&
      "$(grep -c 'AllocMem=0' <<<"$slurm_nodes")" == 2 &&
      "$(grep -c 'AllocTRES= ' <<<"$slurm_nodes")" == 2 ]] || return 1
  [[ "$reservation_count" == 0 && "$workload_count" == 0 ]] || return 1

  for report in "$report0" "$report1"; do
    if [[ "$report" == "$report0" ]]; then expected_node="$(node0)"; else expected_node="$(node1)"; fi
    report_freeze="$(frame_field "$report" freeze_sha256 2>/dev/null || true)"
    [[ "$report_freeze" == "$old_freeze" ||
        ( -n "$prior_freeze" && "$report_freeze" == "$prior_freeze" ) ||
        "$report_freeze" == "$migration_ancestor" ]] || return 1
    [[ "$(frame_field "$report" node 2>/dev/null || true)" == "$expected_node" &&
        "$(frame_field "$report" grant_mode 2>/dev/null || true)" == "$expected_mode" &&
        "$(frame_field "$report" grant_valid 2>/dev/null || true)" == "$expected_valid" &&
        "$(frame_field "$report" source_sha256 2>/dev/null || true)" == "$source_hash" &&
        "$(frame_field "$report" device_barrier 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report" device_barrier_source_sha256 2>/dev/null || true)" == \
          "$(policy_value "$FREEZE" device_barrier_source_sha256)" &&
        "$(frame_field "$report" inventory 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report" protected 2>/dev/null || true)" == 1 ]] || return 1
  done
}

bootstrap_freeze_migration_live_context() {
  local lease="$1" journal="$2" slurm_nodes slurm_jobs slurm_steps report0 report1
  local reservation_count workload_count report_dir
  slurm_nodes="$(slurm_exec scontrol show node "$(slurm0),$(slurm1)" -o)" || return 1
  slurm_jobs="$(slurm_exec squeue -h -w "$(slurm0),$(slurm1)")" || return 1
  slurm_steps="$(slurm_exec squeue --steps -h -w "$(slurm0),$(slurm1)")" || return 1
  report_dir="$(mktemp -d)" || return 1
  if ! host_fence_report_pair_with_host_tmp "$report_dir/report0" "$report_dir/report1"; then
    rm -rf "$report_dir"
    return 1
  fi
  report0="$(<"$report_dir/report0")"
  report1="$(<"$report_dir/report1")"
  rm -rf "$report_dir"
  reservation_count="$(reservation_pods_json | jq '.items | length')" || return 1
  workload_count="$(workload_pods_json | jq '.items | length')" || return 1
  bootstrap_freeze_migration_context "$lease" "$journal" "$slurm_nodes" "$slurm_jobs" \
    "$slurm_steps" "$report0" "$report1" "$reservation_count" "$workload_count"
}

replace_lease() {
  kubectl -n "$(kube_namespace)" replace -f - >/dev/null
}

replace_lease_json() {
  kubectl -n "$(kube_namespace)" replace -f - -o json
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

host_fence_pods_json() {
  kubectl -n "$(kube_namespace)" get pods \
    -l 'app.kubernetes.io/name=pireus-spark-host-fence' -o json
}

device_barrier_config_json() {
  local source
  source="$(repo_root)/$(policy_value "$POLICY" host_device_barrier_source)"
  kubectl -n "$(kube_namespace)" create configmap \
    "$(policy_value "$POLICY" host_device_barrier_configmap)" \
    --from-file="device-barrier.cpp=$source" --dry-run=client -o json | \
    jq '.immutable = true'
}

host_fence_manifest_script() {
  awk '
    /^  host-fence\.sh: \|$/ { in_script=1; next }
    in_script && /^---$/ { exit }
    in_script { sub(/^    /, ""); print }
  ' "$(repo_root)/$(policy_value "$POLICY" host_fence_manifest)"
}

host_fence_runtime_contract_exact() {
  local daemonset="$1"
  jq -e \
    --arg cm "$(policy_value "$POLICY" host_fence_configmap)" \
    --arg barrier_cm "$(policy_value "$POLICY" host_device_barrier_configmap)" '
      (.spec.template.spec.containers | length) == 1 and
      .spec.template.spec.containers[0].securityContext.readOnlyRootFilesystem == true and
      (.spec.template.spec.containers[0].volumeMounts | length) == 4 and
      any(.spec.template.spec.containers[0].volumeMounts[];
        .name == "fence-script" and .mountPath == "/fence" and .readOnly == true) and
      any(.spec.template.spec.containers[0].volumeMounts[];
        .name == "device-barrier-source" and .mountPath == "/barrier" and .readOnly == true) and
      any(.spec.template.spec.containers[0].volumeMounts[];
        .name == "host-root" and .mountPath == "/host") and
      any(.spec.template.spec.containers[0].volumeMounts[];
        .name == "runtime-tmp" and .mountPath == "/tmp" and (.readOnly // false) == false) and
      (.spec.template.spec.volumes | length) == 4 and
      any(.spec.template.spec.volumes[];
        .name == "fence-script" and .configMap.name == $cm and .configMap.defaultMode == 365) and
      any(.spec.template.spec.volumes[];
        .name == "device-barrier-source" and .configMap.name == $barrier_cm and .configMap.defaultMode == 292) and
      any(.spec.template.spec.volumes[];
        .name == "host-root" and .hostPath.path == "/" and .hostPath.type == "Directory") and
      any(.spec.template.spec.volumes[];
        .name == "runtime-tmp" and .emptyDir.sizeLimit == "64Mi")
    ' <<<"$daemonset" >/dev/null
}

host_fence_pair_exact() {
  local daemonset pods config barrier_config expected_barrier expected_barrier_sha live_barrier expected_script expected_script_sha live_script daemonset_uid bound_uid expected expected_daemonset manifest selector_key selector_value
  local require_ready="${3:-1}"
  daemonset="$(kubectl -n "$(kube_namespace)" get daemonset \
    "$(policy_value "$POLICY" host_fence_daemonset)" -o json 2>/dev/null)" || return 1
  config="$(kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" host_fence_configmap)" -o json 2>/dev/null)" || return 1
  barrier_config="$(kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" host_device_barrier_configmap)" -o json 2>/dev/null)" || return 1
  pods="$(host_fence_pods_json 2>/dev/null)" || return 1
  manifest="$(repo_root)/$(policy_value "$POLICY" host_fence_manifest)"
  selector_key="$(policy_value "$POLICY" host_fence_selector_key)"
  selector_value="$(policy_value "$POLICY" host_fence_selector_value)"
  expected="$(kubectl apply --dry-run=server -f "$manifest" -o json 2>/dev/null)" || return 1
  expected_daemonset="$(jq -c --arg key "$selector_key" --arg value "$selector_value" '
    .items[] | select(.kind == "DaemonSet") |
    .spec.template.spec.nodeSelector = {($key): $value}
  ' <<<"$expected")"
  [[ -n "$expected_daemonset" &&
      "$(jq -S -c '.spec' <<<"$daemonset")" == "$(jq -S -c '.spec' <<<"$expected_daemonset")" ]] || return 1
  host_fence_runtime_contract_exact "$daemonset" || return 1
  expected_script="$(awk '
    /^  host-fence\.sh: \|$/ { in_script=1; next }
    in_script && /^---$/ { exit }
    in_script { sub(/^    /, ""); print }
  ' "$(repo_root)/$(policy_value "$POLICY" host_fence_manifest)")"
  expected_script_sha="$(jq -j '.items[] | select(.kind == "ConfigMap") | .data["host-fence.sh"] // ""' \
    <<<"$expected" | sha256sum | cut -d ' ' -f 1)"
  [[ "$(policy_value "$POLICY" host_fence_configmap)" == \
      "pireus-spark-host-fence-${expected_script_sha:0:12}" ]] || return 1
  live_script="$(jq -r '.data["host-fence.sh"] // ""' <<<"$config")"
  [[ -n "$expected_script" && "$live_script" == "$expected_script" ]] || return 1
  expected_barrier="$(cat "$(repo_root)/$(policy_value "$POLICY" host_device_barrier_source)")"
  expected_barrier_sha="$(sha256_file "$(repo_root)/$(policy_value "$POLICY" host_device_barrier_source)")"
  [[ "$(policy_value "$POLICY" host_device_barrier_configmap)" == \
      "pireus-spark-device-barrier-${expected_barrier_sha:0:12}" ]] || return 1
  live_barrier="$(jq -r '.data["device-barrier.cpp"] // ""' <<<"$barrier_config")"
  [[ -n "$expected_barrier" && "$live_barrier" == "$expected_barrier" ]] || return 1
  daemonset_uid="$(jq -r '.metadata.uid // ""' <<<"$daemonset")"
  bound_uid="$(admission_config_json 2>/dev/null | jq -r '.data.hostFenceDaemonSetUid // ""')" || return 1
  [[ "$daemonset_uid" =~ ^[0-9a-f-]{36}$ && "$bound_uid" == "$daemonset_uid" ]] || return 1
  jq -e '
    .immutable == true and
    (.data | keys) == ["host-fence.sh"]
  ' <<<"$config" >/dev/null || return 1
  jq -e '
    .immutable == true and
    (.data | keys) == ["device-barrier.cpp"]
  ' <<<"$barrier_config" >/dev/null || return 1
  jq -e --arg sa "$(policy_value "$POLICY" host_fence_service_account)" \
    --arg image "$(policy_value "$POLICY" host_fence_image)" \
    --arg cm "$(policy_value "$POLICY" host_fence_configmap)" \
    --arg barrier_cm "$(policy_value "$POLICY" host_device_barrier_configmap)" \
    --arg selector_key "$(policy_value "$POLICY" host_fence_selector_key)" \
    --arg selector_value "$(policy_value "$POLICY" host_fence_selector_value)" '
      .spec.selector.matchLabels == {"app.kubernetes.io/name":"pireus-spark-host-fence"} and
      .spec.template.metadata.labels["app.kubernetes.io/name"] == "pireus-spark-host-fence" and
      .spec.template.metadata.labels["pireus.sounio.dev/spark-pair-infrastructure"] == "true" and
      .spec.template.spec.serviceAccountName == $sa and
      .spec.template.spec.automountServiceAccountToken == false and
      .spec.template.spec.hostPID == true and
      .spec.template.spec.restartPolicy == "Always" and
      .spec.template.spec.nodeSelector[$selector_key] == $selector_value and
      (.spec.template.spec.containers | length) == 1 and
      .spec.template.spec.containers[0].name == "host-fence" and
      .spec.template.spec.containers[0].image == $image and
      .spec.template.spec.containers[0].command == ["/bin/bash", "/fence/host-fence.sh", "daemonset-agent"] and
      .spec.template.spec.containers[0].env == [{"name":"NODE_NAME","valueFrom":{"fieldRef":{"apiVersion":"v1","fieldPath":"spec.nodeName"}}}] and
      .spec.template.spec.containers[0].securityContext.privileged == true and
      .spec.template.spec.containers[0].securityContext.readOnlyRootFilesystem == true and
      .spec.template.spec.containers[0].securityContext.allowPrivilegeEscalation == true and
      .spec.template.spec.containers[0].readinessProbe.exec.command == ["/bin/bash", "/fence/host-fence.sh", "report"]
    ' <<<"$daemonset" >/dev/null || return 1
  jq -e --arg n0 "$(node0)" --arg n1 "$(node1)" --arg daemonset_uid "$daemonset_uid" \
    --arg sa "$(policy_value "$POLICY" host_fence_service_account)" \
    --arg image "$(policy_value "$POLICY" host_fence_image)" \
    --arg require_ready "$require_ready" '
      (.items | length) == 2 and
      ([.items[].spec.nodeName] | sort) == ([$n0, $n1] | sort) and
      all(.items[];
        (.metadata.ownerReferences | length) == 1 and
        .metadata.ownerReferences[0].apiVersion == "apps/v1" and
        .metadata.ownerReferences[0].kind == "DaemonSet" and
        .metadata.ownerReferences[0].name == "pireus-spark-host-fence" and
        .metadata.ownerReferences[0].uid == $daemonset_uid and
        .metadata.ownerReferences[0].controller == true and
        .metadata.ownerReferences[0].blockOwnerDeletion == true and
        .spec.serviceAccountName == $sa and
        .spec.automountServiceAccountToken == false and
        .spec.hostPID == true and
        (.spec.containers | length) == 1 and
        .spec.containers[0].image == $image and
        .spec.containers[0].command == ["/bin/bash", "/fence/host-fence.sh", "daemonset-agent"] and
        .spec.containers[0].securityContext.privileged == true and
        .spec.containers[0].securityContext.readOnlyRootFilesystem == true and
        .spec.containers[0].readinessProbe.exec.command == ["/bin/bash", "/fence/host-fence.sh", "report"] and
        ($require_ready != "1" or
          any(.status.conditions[]?; .type == "Ready" and .status == "True")))
    ' <<<"$pods" >/dev/null &&
    jq -e --arg require_ready "$require_ready" '
      .status.desiredNumberScheduled == 2 and
      ($require_ready != "1" or
        (.status.numberReady == 2 and (.status.numberUnavailable // 0) == 0))
    ' <<<"$daemonset" >/dev/null
}

host_fence_exec() {
  local node="$1" pod
  shift
  pod="$(host_fence_pods_json | jq -r --arg node "$node" \
    '.items[] | select(.spec.nodeName == $node) | .metadata.name')"
  [[ -n "$pod" && "$pod" != *$'\n'* ]] || return 1
  kubectl -n "$(kube_namespace)" exec "$pod" -- \
    /bin/bash /fence/host-fence.sh "$@"
}

host_fence_exec_with_host_tmp() {
  local node="$1" pod
  shift
  pod="$(host_fence_pods_json | jq -r --arg node "$node" \
    '.items[] | select(.spec.nodeName == $node) | .metadata.name')"
  [[ -n "$pod" && "$pod" != *$'\n'* ]] || return 1
  kubectl -n "$(kube_namespace)" exec "$pod" -- env TMPDIR=/host/tmp \
    /bin/bash /fence/host-fence.sh "$@"
}

host_fence_install_current_watchdog_via_bridge() {
  local node="$1" source_hash="$2" freeze_hash="$3" barrier_hash="$4"
  local pod script script_sha expected_configmap
  pod="$(host_fence_pods_json | jq -r --arg node "$node" \
    '.items[] | select(.spec.nodeName == $node) | .metadata.name')"
  [[ -n "$pod" && "$pod" != *$'\n'* ]] || return 1
  script="$(host_fence_manifest_script)" || return 1
  [[ -n "$script" ]] || return 1
  script_sha="$(sha256sum <<<"$script" | cut -d ' ' -f 1)"
  expected_configmap="pireus-spark-host-fence-${script_sha:0:12}"
  [[ "$(policy_value "$POLICY" host_fence_configmap)" == "$expected_configmap" ]] || return 1
  kubectl -n "$(kube_namespace)" exec -i "$pod" -- env \
    PIREUS_EXPECTED_SCRIPT_SHA="$script_sha" \
    PIREUS_BIND_SOURCE_SHA="$source_hash" \
    PIREUS_BIND_FREEZE_SHA="$freeze_hash" \
    PIREUS_BIND_BARRIER_SHA="$barrier_hash" \
    /bin/bash -ceu '
      target="/host/tmp/.pireus-host-fence-current.$$"
      trap '\''rm -f "$target"'\'' EXIT
      cat > "$target"
      [[ "$(sha256sum "$target" | cut -d " " -f 1)" == "$PIREUS_EXPECTED_SCRIPT_SHA" ]]
      chmod 0700 "$target"
      TMPDIR=/host/tmp PIREUS_HOST_FENCE_INSTALL_SOURCE="$target" \
        /bin/bash "$target" install-watchdog \
          "$PIREUS_BIND_SOURCE_SHA" "$PIREUS_BIND_FREEZE_SHA" "$PIREUS_BIND_BARRIER_SHA"
    ' <<<"$script"
}

host_fence_install_current_watchdog_pair_via_bridge() {
  local source_hash="$1" freeze_hash="$2" barrier_hash="$3"
  local pid0 pid1 status0 status1
  host_fence_install_current_watchdog_via_bridge "$(node0)" \
    "$source_hash" "$freeze_hash" "$barrier_hash" &
  pid0=$!
  host_fence_install_current_watchdog_via_bridge "$(node1)" \
    "$source_hash" "$freeze_hash" "$barrier_hash" &
  pid1=$!
  if wait "$pid0"; then status0=0; else status0=$?; fi
  if wait "$pid1"; then status1=0; else status1=$?; fi
  [[ "$status0" == 0 && "$status1" == 0 ]]
}

host_fence_report_with_host_tmp() {
  local node="$1" pod
  pod="$(host_fence_pods_json | jq -r --arg node "$node" \
    '.items[] | select(.spec.nodeName == $node) | .metadata.name')"
  [[ -n "$pod" && "$pod" != *$'\n'* ]] || return 1
  kubectl -n "$(kube_namespace)" exec "$pod" -- env \
    TMPDIR=/host/tmp PIREUS_HOST_ROOT=/host \
    PIREUS_HOST_FENCE_INSTALL_SOURCE=/host/usr/local/lib/pireus/spark-pair-host-fence \
    /bin/bash /host/usr/local/lib/pireus/spark-pair-host-fence report
}

host_fence_report_pair_with_host_tmp() {
  local output0="$1" output1="$2" pid0 pid1 status0 status1
  host_fence_report_with_host_tmp "$(node0)" >"$output0" &
  pid0=$!
  host_fence_report_with_host_tmp "$(node1)" >"$output1" &
  pid1=$!
  if wait "$pid0"; then status0=0; else status0=$?; fi
  if wait "$pid1"; then status1=0; else status1=$?; fi
  [[ "$status0" == 0 && "$status1" == 0 ]]
}

host_fence_host_tmp_writable() {
  local node="$1" pod
  pod="$(host_fence_pods_json | jq -r --arg node "$node" \
    '.items[] | select(.spec.nodeName == $node) | .metadata.name')"
  [[ -n "$pod" && "$pod" != *$'\n'* ]] || return 1
  kubectl -n "$(kube_namespace)" exec "$pod" -- /bin/bash -ceu '
    probe="/host/tmp/.pireus-watchdog-tmp.$$"
    : > "$probe"
    rm -f "$probe"
  '
}

host_fence_report() {
  host_fence_exec "$1" report
}

slurm_free_memory_ready() {
  local nodes="$1" minimum count=0 token value
  minimum="$(policy_value "$POLICY" minimum_free_memory_mb)"
  while IFS= read -r token; do
    value="${token#FreeMem=}"
    [[ "$value" =~ ^[0-9]+$ ]] || return 1
    (( value >= minimum )) || return 1
    count=$((count + 1))
  done < <(tr ' ' '\n' <<<"$nodes" | sed -n '/^FreeMem=/p')
  [[ $count -eq 2 ]]
}

host_mask_from_facts() {
  local lease="$1" slurm_nodes="$2" holder="$3" epoch="$4"
  local mask=0 exact=0 report0='' report1='' truth=0 boot0='' boot1='' field_power field power mode0 mode1 source_hash freeze_hash receipt0 receipt1
  local transaction0 transaction1 decision0 decision1 pair0 pair1 lease_uid lease_rv_bound reported0 reported1 watchdog0 watchdog1 fresh
  local prepare0 prepare1 base_lease_rv expected_pair
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  host_fence_pair_exact && exact=1
  mask="$(bit_add "$mask" 1 "$exact")"
  if [[ $exact -eq 1 ]]; then
    report0="$(host_fence_report "$(node0)" 2>/dev/null || true)"
    report1="$(host_fence_report "$(node1)" 2>/dev/null || true)"
  fi
  if [[ -n "$report0" ]]; then boot0="$(frame_field "$report0" boot_id 2>/dev/null || true)"; fi
  if [[ -n "$report1" ]]; then boot1="$(frame_field "$report1" boot_id 2>/dev/null || true)"; fi

  truth=0
  if [[ -n "$boot0" && -n "$boot1" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_boot_0_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$boot0" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_boot_1_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$boot1" ]]; then truth=1; fi
  mask="$(bit_add "$mask" 2 "$truth")"
  truth=0
  lease_uid="$(jq -r '.metadata.uid // ""' <<<"$lease")"
  lease_rv_bound="$(jq -r --arg key "$(policy_value "$POLICY" host_lease_resource_version_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  transaction0="$(frame_field "$report0" transaction_id 2>/dev/null || true)"
  transaction1="$(frame_field "$report1" transaction_id 2>/dev/null || true)"
  decision0="$(frame_field "$report0" decision_receipt_sha256 2>/dev/null || true)"
  decision1="$(frame_field "$report1" decision_receipt_sha256 2>/dev/null || true)"
  pair0="$(frame_field "$report0" pair_digest 2>/dev/null || true)"
  pair1="$(frame_field "$report1" pair_digest 2>/dev/null || true)"
  mode0="$(frame_field "$report0" grant_mode 2>/dev/null || true)"
  mode1="$(frame_field "$report1" grant_mode 2>/dev/null || true)"
  prepare0="$(jq -r --arg key "$(policy_value "$POLICY" host_prepare_0_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  prepare1="$(jq -r --arg key "$(policy_value "$POLICY" host_prepare_1_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  base_lease_rv="$(jq -r --arg key "$(policy_value "$POLICY" host_intent_base_rv_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")"
  expected_pair="$(printf 'transaction_id=%s\nlease_uid=%s\nbase_lease_resource_version=%s\nnode0_prepare=%s\nnode1_prepare=%s\n' \
    "$transaction0" "$lease_uid" "$base_lease_rv" "$prepare0" "$prepare1" | \
    sha256sum | cut -d ' ' -f 1)"
  if [[ "$(frame_field "$report0" grant_epoch 2>/dev/null || true)" == "$epoch" &&
        "$(frame_field "$report1" grant_epoch 2>/dev/null || true)" == "$epoch" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_fence_epoch_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$epoch" &&
        "$(frame_field "$report0" lease_uid 2>/dev/null || true)" == "$lease_uid" &&
        "$(frame_field "$report1" lease_uid 2>/dev/null || true)" == "$lease_uid" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_lease_uid_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$lease_uid" &&
        "$(frame_field "$report0" lease_resource_version 2>/dev/null || true)" == "$lease_rv_bound" &&
        "$(frame_field "$report1" lease_resource_version 2>/dev/null || true)" == "$lease_rv_bound" &&
        "$transaction0" =~ ^[0-9a-f]{64}$ && "$transaction1" == "$transaction0" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_transaction_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$transaction0" &&
        "$decision0" =~ ^[0-9a-f]{64}$ && "$decision1" == "$decision0" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_decision_receipt_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$decision0" &&
        "$pair1" == "$pair0" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_pair_digest_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$pair0" ]]; then
    if [[ "$mode0" == FENCED && "$mode1" == FENCED &&
          "$(frame_field "$report0" grant_valid 2>/dev/null || true)" == 0 &&
          "$(frame_field "$report1" grant_valid 2>/dev/null || true)" == 0 &&
          "$pair0" == none && -z "$prepare0" && -z "$prepare1" && -z "$base_lease_rv" ]]; then
      truth=1
    elif [[ "$mode0" =~ ^(SLURM|K8S)$ && "$mode1" == "$mode0" &&
            "$(frame_field "$report0" grant_valid 2>/dev/null || true)" == 1 &&
            "$(frame_field "$report1" grant_valid 2>/dev/null || true)" == 1 &&
            "$prepare0" =~ ^[0-9a-f]{64}$ && "$prepare1" =~ ^[0-9a-f]{64}$ &&
            "$base_lease_rv" =~ ^[1-9][0-9]*$ && "$pair0" == "$expected_pair" ]]; then
      truth=1
    fi
  fi
  mask="$(bit_add "$mask" 4 "$truth")"
  truth=0
  if [[ "$(frame_field "$report0" grant_owner 2>/dev/null || true)" == "$holder" &&
        "$(frame_field "$report1" grant_owner 2>/dev/null || true)" == "$holder" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_fence_owner_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$holder" ]]; then truth=1; fi
  mask="$(bit_add "$mask" 8 "$truth")"
  truth=0
  if [[ "$exact" == 1 &&
        "$(frame_field "$report0" watchdog 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report1" watchdog 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report0" source_sha256 2>/dev/null || true)" == "$source_hash" &&
        "$(frame_field "$report1" source_sha256 2>/dev/null || true)" == "$source_hash" &&
        "$(frame_field "$report0" freeze_sha256 2>/dev/null || true)" == "$freeze_hash" &&
        "$(frame_field "$report1" freeze_sha256 2>/dev/null || true)" == "$freeze_hash" ]]; then truth=1; fi
  mask="$(bit_add "$mask" 16 "$truth")"
  truth=0
  if [[ "$(frame_field "$report0" watchdog 2>/dev/null || true)" == 1 &&
      "$(frame_field "$report1" watchdog 2>/dev/null || true)" == 1 ]] &&
      admission_fail_closed "$lease" &&
      [[ "$(policy_value "$POLICY" host_runtime_restart_required)" == false ]]; then truth=1; fi
  mask="$(bit_add "$mask" 32 "$truth")"

  for field_power in inventory:64 services:128 restarts:256 docker_claims:512 \
    consumers:1024 cgroups:2048; do
    field="${field_power%%:*}"
    power="${field_power#*:}"
    truth=0
    if [[ "$(frame_field "$report0" "$field" 2>/dev/null || true)" == 1 &&
          "$(frame_field "$report1" "$field" 2>/dev/null || true)" == 1 ]]; then truth=1; fi
    mask="$(bit_add "$mask" "$power" "$truth")"
  done
  truth=0
  if [[ "$(frame_field "$report0" memory 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report1" memory 2>/dev/null || true)" == 1 ]] &&
      slurm_free_memory_ready "$slurm_nodes"; then truth=1; fi
  mask="$(bit_add "$mask" 4096 "$truth")"
  truth=0
  if [[ "$(frame_field "$report0" protected 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report1" protected 2>/dev/null || true)" == 1 ]]; then truth=1; fi
  mask="$(bit_add "$mask" 8192 "$truth")"
  truth=0
  receipt0="$(frame_field "$report0" receipt_sha256 2>/dev/null || true)"
  receipt1="$(frame_field "$report1" receipt_sha256 2>/dev/null || true)"
  if [[ "$report0" == "PIREUS_HOST_FACTS node=$(node0) "* &&
        "$report1" == "PIREUS_HOST_FACTS node=$(node1) "* &&
        "$receipt0" =~ ^[0-9a-f]{64}$ &&
        "$receipt1" =~ ^[0-9a-f]{64}$ &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_receipt_0_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$receipt0" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_receipt_1_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$receipt1" &&
        "$(frame_field "$report0" source_sha256 2>/dev/null || true)" == "$source_hash" &&
        "$(frame_field "$report1" source_sha256 2>/dev/null || true)" == "$source_hash" &&
        "$(frame_field "$report0" freeze_sha256 2>/dev/null || true)" == "$freeze_hash" &&
        "$(frame_field "$report1" freeze_sha256 2>/dev/null || true)" == "$freeze_hash" &&
        "$(frame_field "$report0" reported_monotonic 2>/dev/null || true)" =~ ^[0-9]+$ &&
        "$(frame_field "$report1" reported_monotonic 2>/dev/null || true)" =~ ^[0-9]+$ ]]; then truth=1; fi
  mask="$(bit_add "$mask" 16384 "$truth")"
  truth=0
  reported0="$(frame_field "$report0" reported_monotonic 2>/dev/null || true)"
  reported1="$(frame_field "$report1" reported_monotonic 2>/dev/null || true)"
  watchdog0="$(frame_field "$report0" watchdog_monotonic 2>/dev/null || true)"
  watchdog1="$(frame_field "$report1" watchdog_monotonic 2>/dev/null || true)"
  fresh="$(policy_value "$POLICY" host_watchdog_fresh_seconds)"
  if [[ $exact -eq 1 && "$reported0" =~ ^[0-9]+$ && "$reported1" =~ ^[0-9]+$ &&
        "$watchdog0" =~ ^[0-9]+$ && "$watchdog1" =~ ^[0-9]+$ &&
        $((reported0 - watchdog0)) -ge 0 && $((reported0 - watchdog0)) -le $fresh &&
        $((reported1 - watchdog1)) -ge 0 && $((reported1 - watchdog1)) -le $fresh &&
        "$(frame_field "$report0" watchdog 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report1" watchdog 2>/dev/null || true)" == 1 ]]; then
    mode0="$(frame_field "$report0" grant_mode 2>/dev/null || true)"
    mode1="$(frame_field "$report1" grant_mode 2>/dev/null || true)"
    if [[ "$mode0" == "$mode1" && "$mode0" != K8S ]] ||
       [[ "$mode0" == K8S && "$mode1" == K8S &&
          "$(frame_field "$report0" grant_valid 2>/dev/null || true)" == 1 &&
          "$(frame_field "$report1" grant_valid 2>/dev/null || true)" == 1 ]]; then truth=1; fi
  fi
  mask="$(bit_add "$mask" 32768 "$truth")"
  truth=0
  if [[ "$(frame_field "$report0" device_barrier 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report1" device_barrier 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report0" device_barrier_source_sha256 2>/dev/null || true)" == "$(policy_value "$FREEZE" device_barrier_source_sha256)" &&
        "$(frame_field "$report1" device_barrier_source_sha256 2>/dev/null || true)" == "$(policy_value "$FREEZE" device_barrier_source_sha256)" &&
        "$(frame_field "$report0" device_barrier_binary_sha256 2>/dev/null || true)" =~ ^[0-9a-f]{64}$ &&
        "$(frame_field "$report1" device_barrier_binary_sha256 2>/dev/null || true)" =~ ^[0-9a-f]{64}$ ]]; then truth=1; fi
  mask="$(bit_add "$mask" 65536 "$truth")"
  printf '%s\n' "$mask"
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
  printf 'state=UNINITIALIZED epoch=1 observed_epoch=1 authority_mask=%s slurm_mask=%s k8s_mask=%s host_mask=0\n' \
    "$authority_mask" "$slurm_mask" "$k8s_mask"
}

facts_impl() {
  local binding_mode="$1"
  shift
  local holder lease nodeset node_0 node_1 plugin_0 plugin_1 plugin_pods slurmd_pods reservations workloads all_pods slurm_nodes slurm_jobs slurm_steps
  local state epoch observed_epoch authority_mask=1 slurm_mask=0 k8s_mask=0 host_mask=0 truth current_generation lease_generation
  holder="$(arg_value --holder "$@")"
  lease="$(lease_json)"
  if [[ "$binding_mode" == strict ]]; then
    verify_lease_freeze_binding "$lease"
  else
    [[ "$binding_mode" == migration ]] || fail 'unknown facts binding mode'
  fi
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
  if [[ "$(jq -r '.spec.template.spec.runtimeClassName // ""' <<<"$nodeset")" == "$(policy_value "$POLICY" slurmd_runtime_class)" &&
        "$(jq -r '.spec.slurmd.resources.requests["nvidia.com/gpu"] // "0"' <<<"$nodeset")" == 1 &&
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
        .spec.runtimeClassName == "nvidia" and
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

  host_mask="$(host_mask_from_facts "$lease" "$slurm_nodes" \
    "$(jq -r '.spec.holderIdentity // ""' <<<"$lease")" "$epoch")"
  printf 'state=%s epoch=%s observed_epoch=%s authority_mask=%s slurm_mask=%s k8s_mask=%s host_mask=%s\n' \
    "$state" "$epoch" "$observed_epoch" "$authority_mask" "$slurm_mask" "$k8s_mask" "$host_mask"
}

facts() {
  facts_impl strict "$@"
}

bootstrap_migration_facts() {
  local lease journal
  lease="$(lease_json)"
  journal="$(kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" bootstrap_journal)" -o json)" || \
    fail 'bootstrap migration requires the persisted journal'
  bootstrap_freeze_migration_live_context "$lease" "$journal" || \
    fail 'bootstrap freeze migration context is not fail-closed'
  facts_impl migration "$@"
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
  now="$(lease_timestamp)"
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
  now="$(lease_timestamp)"
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
  now="$(lease_timestamp)"
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
  now="$(lease_timestamp)"
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
  if kubectl get "$(policy_value "$POLICY" admission_parameter_resource)" \
    "$(policy_value "$POLICY" admission_parameter)" >/dev/null 2>&1; then
    sync_admission_projection "$updated"
  fi
  ensure_bootstrap_journal BOOTSTRAP_TAKEOVER "$holder" "$receipt"
  printf 'epoch=%s state=UNINITIALIZED\n' "$next_epoch"
}

bootstrap_migrate_freeze() {
  local holder epoch receipt lease journal current_freeze old_freeze receipt_hash now
  local updated_journal updated_lease
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  verify_receipt "$receipt" 27 "$epoch"
  lease="$(lease_json)"
  journal="$(kubectl -n "$(kube_namespace)" get configmap \
    "$(policy_value "$POLICY" bootstrap_journal)" -o json)" || \
    fail 'bootstrap migration requires the persisted journal'
  bootstrap_freeze_migration_live_context "$lease" "$journal" || \
    fail 'bootstrap freeze migration context changed before CAS'

  current_freeze="$(sha256_file "$FREEZE")"
  old_freeze="$(jq -r --arg key "$(policy_value "$POLICY" freeze_hash_annotation)" \
    '.metadata.annotations[$key] // ""' <<<"$lease")"
  if [[ "$old_freeze" == "$current_freeze" ]]; then
    old_freeze="$(jq -r '.data.semanticsFreezeSha256 // ""' <<<"$journal")"
  fi
  [[ "$old_freeze" =~ ^[0-9a-f]{64}$ && "$old_freeze" != "$current_freeze" ]] || \
    fail 'bootstrap migration old freeze is missing'
  receipt_hash="$(sha256_file "$receipt")"
  now="$(lease_timestamp)"

  updated_journal="$(jq --arg current "$current_freeze" --arg old "$old_freeze" \
    --arg receipt "$receipt_hash" --arg now "$now" '
      .data.semanticsFreezeSha256 = $current |
      .data.updatedUtc = $now |
      .data.lastReceiptSha256 = $receipt |
      .data.migrationFromFreezeSha256 = $old |
      .data.migrationToFreezeSha256 = $current |
      .data.migrationReceiptSha256 = $receipt
    ' <<<"$journal")"
  kubectl -n "$(kube_namespace)" replace -f - <<<"$updated_journal" >/dev/null

  updated_lease="$(jq --arg current "$current_freeze" \
    --arg key "$(policy_value "$POLICY" freeze_hash_annotation)" '
      .metadata.annotations[$key] = $current
    ' <<<"$lease")"
  replace_lease <<<"$updated_lease"
  printf 'epoch=%s state=UNINITIALIZED from_freeze=%s to_freeze=%s\n' \
    "$epoch" "$old_freeze" "$current_freeze"
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
  now="$(lease_timestamp)"
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
  now="$(lease_timestamp)"
  updated="$(jq --arg now "$now" '.spec.renewTime = $now' <<<"$lease")"
  replace_lease <<<"$updated"
}

material_keepalive() {
  local holder epoch receipt
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation keepalive "$holder" "$epoch" "$receipt"
  renew_lease_material "$holder" "$epoch"
}

slurm_drained() {
  local nodes jobs steps
  nodes="$(slurm_exec scontrol show node "$(slurm0),$(slurm1)" -o)" || return 1
  jobs="$(slurm_exec squeue -h -w "$(slurm0),$(slurm1)")" || return 1
  steps="$(slurm_exec squeue --steps -h -w "$(slurm0),$(slurm1)")" || return 1
  slurm_states_drained "$nodes" && [[ -z "$jobs" && -z "$steps" ]]
}

drain_slurm() {
  local holder epoch receipt key value effect state selector_key
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  state="$(guard_mutation drain "$holder" "$epoch" "$receipt")"
  key="$(policy_value "$POLICY" spark_taint_key)"
  value="$(policy_value "$POLICY" spark_taint_value)"
  effect="$(policy_value "$POLICY" spark_taint_effect)"
  kubectl taint nodes "$(node0)" "$(node1)" "$key=$value:$effect" --overwrite >/dev/null
  slurm_exec scontrol update NodeName="$(slurm0),$(slurm1)" State=DRAIN Reason="pireus-epoch-$epoch" >/dev/null
  wait_for 'both Slurm nodes to drain' slurm_drained "$holder" "$epoch"
  if [[ "$state" == UNINITIALIZED ]]; then
    selector_key="$(policy_value "$POLICY" slurmd_selector_key)"
    kubectl label nodes "$(node0)" "$(node1)" "$selector_key-" >/dev/null
    wait_for 'bootstrap slurmd pods to terminate' slurmd_absent "$holder" "$epoch"
  fi
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
      env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        - name: PIREUS_EPOCH
          value: "$epoch"
      securityContext:
        privileged: true
      resources:
        requests:
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
      command: [/bin/bash, /probe/reservation-probe.sh]
      readinessProbe:
        exec:
          command: [/usr/bin/test, -f, /tmp/nvml-clean]
        periodSeconds: 2
        failureThreshold: 90
      volumeMounts:
        - name: reservation-probe
          mountPath: /probe
          readOnly: true
        - name: host-root
          mountPath: /host
          readOnly: true
  volumes:
    - name: reservation-probe
      configMap:
        name: $(policy_value "$POLICY" reservation_probe_configmap)
        defaultMode: 0555
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
  now="$(lease_timestamp)"
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
  # The Sounio controller refreshes the paired K8S host grant while this wait
  # is active. A second Lease writer here would race that durable 2PC loop.
  wait_for 'two exact-node GPU reservations' reservations_ready
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
  now="$(lease_timestamp)"
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
  now="$(lease_timestamp)"
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
  if kubectl -n "$(kube_namespace)" get daemonset \
    "$(policy_value "$POLICY" host_fence_daemonset)" >/dev/null 2>&1; then
    stage_existing_host_fence_for_bootstrap "$holder" "$epoch"
  fi
  manifest="$(repo_root)/$(policy_value "$POLICY" admission_manifest)"
  parameter_crd_manifest | kubectl apply --server-side --field-manager=pireus-spark-pair -f - >/dev/null
  kubectl wait --for=condition=Established \
    "crd/$(policy_value "$POLICY" admission_parameter_crd)" \
    --timeout="$(policy_value "$POLICY" operation_timeout_seconds)s" >/dev/null
  kubectl apply --server-side --force-conflicts \
    --field-manager=pireus-spark-pair -f "$manifest" >/dev/null
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

host_fence_staged() {
  local daemonset pods bootstrap_key bootstrap_value
  bootstrap_key="$(policy_value "$POLICY" host_fence_bootstrap_selector_key)"
  bootstrap_value="$(policy_value "$POLICY" host_fence_bootstrap_selector_value)"
  daemonset="$(kubectl -n "$(kube_namespace)" get daemonset \
    "$(policy_value "$POLICY" host_fence_daemonset)" -o json 2>/dev/null)" || return 1
  pods="$(host_fence_pods_json 2>/dev/null)" || return 1
  jq -e --arg key "$bootstrap_key" --arg value "$bootstrap_value" '
    .spec.template.spec.nodeSelector == {($key): $value} and
    (.status.desiredNumberScheduled // 0) == 0 and
    (.status.currentNumberScheduled // 0) == 0 and
    (.status.numberReady // 0) == 0
  ' <<<"$daemonset" >/dev/null &&
    jq -e '(.items | length) == 0' <<<"$pods" >/dev/null
}

host_fence_legacy_runtime_bridge_exact() {
  local daemonset pods config barrier_config daemonset_uid bound_uid script_sha barrier_sha
  local live_configmap live_barrier_configmap
  local require_ready="${3:-1}"
  daemonset="$(kubectl -n "$(kube_namespace)" get daemonset \
    "$(policy_value "$POLICY" host_fence_daemonset)" -o json 2>/dev/null)" || return 1
  pods="$(host_fence_pods_json 2>/dev/null)" || return 1
  live_configmap="$(jq -r '
    .spec.template.spec.volumes[] |
    select(.name == "fence-script") | .configMap.name // ""
  ' <<<"$daemonset")"
  live_barrier_configmap="$(jq -r '
    .spec.template.spec.volumes[] |
    select(.name == "device-barrier-source") | .configMap.name // ""
  ' <<<"$daemonset")"
  [[ -n "$live_configmap" && -n "$live_barrier_configmap" ]] || return 1
  config="$(kubectl -n "$(kube_namespace)" get configmap \
    "$live_configmap" -o json 2>/dev/null)" || return 1
  barrier_config="$(kubectl -n "$(kube_namespace)" get configmap \
    "$live_barrier_configmap" -o json 2>/dev/null)" || return 1
  daemonset_uid="$(jq -r '.metadata.uid // ""' <<<"$daemonset")"
  bound_uid="$(admission_config_json 2>/dev/null | \
    jq -r '.data.hostFenceDaemonSetUid // ""')" || return 1
  script_sha="$(jq -j '.data["host-fence.sh"] // ""' <<<"$config" | \
    sha256sum | cut -d ' ' -f 1)"
  barrier_sha="$(jq -j '.data["device-barrier.cpp"] // ""' <<<"$barrier_config" | \
    sha256sum | cut -d ' ' -f 1)"
  [[ "$daemonset_uid" =~ ^[0-9a-f-]{36}$ && "$bound_uid" == "$daemonset_uid" &&
      "$live_configmap" == "pireus-spark-host-fence-${script_sha:0:12}" &&
      "$live_barrier_configmap" == \
        "pireus-spark-device-barrier-${barrier_sha:0:12}" ]] || return 1
  jq -e '
    .immutable == true and (.data | keys) == ["host-fence.sh"]
  ' <<<"$config" >/dev/null || return 1
  jq -e '
    .immutable == true and (.data | keys) == ["device-barrier.cpp"]
  ' <<<"$barrier_config" >/dev/null || return 1
  jq -e \
    --arg sa "$(policy_value "$POLICY" host_fence_service_account)" \
    --arg image "$(policy_value "$POLICY" host_fence_image)" \
    --arg cm "$live_configmap" \
    --arg barrier_cm "$live_barrier_configmap" \
    --arg selector_key "$(policy_value "$POLICY" host_fence_selector_key)" \
    --arg selector_value "$(policy_value "$POLICY" host_fence_selector_value)" \
    --arg require_ready "$require_ready" '
      .spec.selector.matchLabels == {"app.kubernetes.io/name":"pireus-spark-host-fence"} and
      .spec.template.metadata.labels["app.kubernetes.io/name"] == "pireus-spark-host-fence" and
      .spec.template.metadata.labels["pireus.sounio.dev/spark-pair-infrastructure"] == "true" and
      .spec.template.spec.serviceAccountName == $sa and
      .spec.template.spec.automountServiceAccountToken == false and
      .spec.template.spec.hostPID == true and
      .spec.template.spec.restartPolicy == "Always" and
      .spec.template.spec.nodeSelector == {($selector_key): $selector_value} and
      (.spec.template.spec.containers | length) == 1 and
      .spec.template.spec.containers[0].name == "host-fence" and
      .spec.template.spec.containers[0].image == $image and
      .spec.template.spec.containers[0].command == ["/bin/bash", "/fence/host-fence.sh", "daemonset-agent"] and
      .spec.template.spec.containers[0].securityContext.privileged == true and
      .spec.template.spec.containers[0].securityContext.readOnlyRootFilesystem == true and
      .spec.template.spec.containers[0].securityContext.allowPrivilegeEscalation == true and
      .spec.template.spec.containers[0].readinessProbe.exec.command == ["/bin/bash", "/fence/host-fence.sh", "report"] and
      ((.spec.template.spec.containers[0].volumeMounts | length) == 3 or
        ((.spec.template.spec.containers[0].volumeMounts | length) == 4 and
          any(.spec.template.spec.containers[0].volumeMounts[];
            .name == "runtime-tmp" and .mountPath == "/tmp" and
            (.readOnly // false) == false))) and
      any(.spec.template.spec.containers[0].volumeMounts[];
        .name == "fence-script" and .mountPath == "/fence" and .readOnly == true) and
      any(.spec.template.spec.containers[0].volumeMounts[];
        .name == "device-barrier-source" and .mountPath == "/barrier" and .readOnly == true) and
      any(.spec.template.spec.containers[0].volumeMounts[];
        .name == "host-root" and .mountPath == "/host") and
      ((.spec.template.spec.volumes | length) == 3 or
        ((.spec.template.spec.volumes | length) == 4 and
          any(.spec.template.spec.volumes[];
            .name == "runtime-tmp" and .emptyDir.sizeLimit == "64Mi"))) and
      any(.spec.template.spec.volumes[];
        .name == "fence-script" and .configMap.name == $cm and .configMap.defaultMode == 365) and
      any(.spec.template.spec.volumes[];
        .name == "device-barrier-source" and .configMap.name == $barrier_cm and .configMap.defaultMode == 292) and
      any(.spec.template.spec.volumes[];
        .name == "host-root" and .hostPath.path == "/" and .hostPath.type == "Directory") and
      (.status.desiredNumberScheduled // 0) == 2 and
      ($require_ready != "1" or
        ((.status.numberReady // 0) == 2 and
          (.status.numberUnavailable // 0) == 0))
    ' <<<"$daemonset" >/dev/null || return 1
  jq -e --arg n0 "$(node0)" --arg n1 "$(node1)" \
    --arg daemonset_uid "$daemonset_uid" \
    --arg sa "$(policy_value "$POLICY" host_fence_service_account)" \
    --arg image "$(policy_value "$POLICY" host_fence_image)" \
    --arg require_ready "$require_ready" '
      (.items | length) == 2 and
      ([.items[].spec.nodeName] | sort) == ([$n0, $n1] | sort) and
      all(.items[];
        (.metadata.ownerReferences | length) == 1 and
        .metadata.ownerReferences[0].apiVersion == "apps/v1" and
        .metadata.ownerReferences[0].kind == "DaemonSet" and
        .metadata.ownerReferences[0].name == "pireus-spark-host-fence" and
        .metadata.ownerReferences[0].uid == $daemonset_uid and
        .metadata.ownerReferences[0].controller == true and
        .metadata.ownerReferences[0].blockOwnerDeletion == true and
        .spec.serviceAccountName == $sa and
        .spec.automountServiceAccountToken == false and
        .spec.hostPID == true and
        (.spec.containers | length) == 1 and
        .spec.containers[0].image == $image and
        .spec.containers[0].command == ["/bin/bash", "/fence/host-fence.sh", "daemonset-agent"] and
        .spec.containers[0].securityContext.privileged == true and
        .spec.containers[0].securityContext.readOnlyRootFilesystem == true and
        ($require_ready != "1" or
          any(.status.conditions[]?; .type == "Ready" and .status == "True")))
    ' <<<"$pods" >/dev/null
}

host_fence_watchdogs_ready_for_staging() {
  local report0 report1 source_hash freeze_hash barrier_hash barrier_binary0 barrier_binary1 report_dir
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  barrier_hash="$(policy_value "$FREEZE" device_barrier_source_sha256)"
  report_dir="$(mktemp -d)" || return 1
  if ! host_fence_report_pair_with_host_tmp "$report_dir/report0" "$report_dir/report1" \
    2>/dev/null; then
    rm -rf "$report_dir"
    return 1
  fi
  report0="$(<"$report_dir/report0")"
  report1="$(<"$report_dir/report1")"
  rm -rf "$report_dir"
  barrier_binary0="$(frame_field "$report0" device_barrier_binary_sha256 2>/dev/null || true)"
  barrier_binary1="$(frame_field "$report1" device_barrier_binary_sha256 2>/dev/null || true)"
  [[ "$barrier_binary0" =~ ^[0-9a-f]{64}$ && "$barrier_binary1" == "$barrier_binary0" ]] || return 1
  local report expected_node
  for report in "$report0" "$report1"; do
    if [[ "$report" == "$report0" ]]; then expected_node="$(node0)"; else expected_node="$(node1)"; fi
    [[ "$(frame_field "$report" node 2>/dev/null || true)" == "$expected_node" &&
        "$(frame_field "$report" grant_mode 2>/dev/null || true)" == FENCED &&
        "$(frame_field "$report" grant_valid 2>/dev/null || true)" == 0 &&
        "$(frame_field "$report" watchdog 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report" source_sha256 2>/dev/null || true)" == "$source_hash" &&
        "$(frame_field "$report" freeze_sha256 2>/dev/null || true)" == "$freeze_hash" &&
        "$(frame_field "$report" device_barrier 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report" device_barrier_source_sha256 2>/dev/null || true)" == "$barrier_hash" &&
        "$(frame_field "$report" inventory 2>/dev/null || true)" == 1 &&
        "$(frame_field "$report" protected 2>/dev/null || true)" == 1 ]] || return 1
  done
}

stage_host_fence_daemonset() {
  local daemonset patched bootstrap_key bootstrap_value
  bootstrap_key="$(policy_value "$POLICY" host_fence_bootstrap_selector_key)"
  bootstrap_value="$(policy_value "$POLICY" host_fence_bootstrap_selector_value)"
  daemonset="$(kubectl -n "$(kube_namespace)" get daemonset \
    "$(policy_value "$POLICY" host_fence_daemonset)" -o json)"
  patched="$(jq --arg key "$bootstrap_key" --arg value "$bootstrap_value" '
    .spec.template.spec.nodeSelector = {($key): $value}
  ' <<<"$daemonset")"
  kubectl -n "$(kube_namespace)" replace -f - <<<"$patched" >/dev/null
}

stage_existing_host_fence_for_bootstrap() {
  local holder="$1" epoch="$2" source_hash freeze_hash barrier_hash
  host_fence_staged && return 0
  if host_fence_pair_exact "$holder" "$epoch" 0; then
    source_hash="$(policy_value "$FREEZE" authority_sha256)"
    freeze_hash="$(sha256_file "$FREEZE")"
    barrier_hash="$(policy_value "$FREEZE" device_barrier_source_sha256)"
    host_fence_install_current_watchdog_pair_via_bridge \
      "$source_hash" "$freeze_hash" "$barrier_hash"
    wait_for 'rebound current Spark host watchdogs' \
      host_fence_watchdogs_ready_for_staging "$holder" "$epoch"
    stage_host_fence_daemonset
    wait_for 'inert current Spark host fence DaemonSet' host_fence_staged \
      "$holder" "$epoch"
    return 0
  fi
  host_fence_legacy_runtime_bridge_exact "$holder" "$epoch" 0 || \
    fail 'existing host fence is not the exact legacy runtime bridge'
  host_fence_host_tmp_writable "$(node0)" && host_fence_host_tmp_writable "$(node1)" || \
    fail 'existing host fence cannot use host /tmp for watchdog repair'
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  barrier_hash="$(policy_value "$FREEZE" device_barrier_source_sha256)"
  host_fence_install_current_watchdog_pair_via_bridge \
    "$source_hash" "$freeze_hash" "$barrier_hash"
  wait_for 'rebound Spark host watchdogs' host_fence_watchdogs_ready_for_staging \
    "$holder" "$epoch"
  stage_host_fence_daemonset
  wait_for 'inert legacy Spark host fence DaemonSet' host_fence_staged "$holder" "$epoch"
}

bind_host_fence_daemonset_uid() {
  local daemonset config uid current updated
  host_fence_staged || fail 'host fence DaemonSet is not inert during UID binding'
  daemonset="$(kubectl -n "$(kube_namespace)" get daemonset \
    "$(policy_value "$POLICY" host_fence_daemonset)" -o json)"
  uid="$(jq -r '.metadata.uid // ""' <<<"$daemonset")"
  [[ "$uid" =~ ^[0-9a-f-]{36}$ ]] || fail 'host fence DaemonSet UID is malformed'
  config="$(admission_config_json)"
  current="$(jq -r '.data.hostFenceDaemonSetUid // ""' <<<"$config")"
  [[ "$current" == UNBOUND || "$current" =~ ^[0-9a-f-]{36}$ ]] || \
    fail 'host fence admission UID parameter is malformed'
  updated="$(jq --arg uid "$uid" '.data.hostFenceDaemonSetUid = $uid' <<<"$config")"
  kubectl -n "$(kube_namespace)" replace -f - <<<"$updated" >/dev/null
  [[ "$(admission_config_json | jq -r '.data.hostFenceDaemonSetUid // ""')" == "$uid" ]] || \
    fail 'host fence admission UID CAS did not persist'
}

activate_host_fence_daemonset() {
  local daemonset patched selector_key selector_value
  selector_key="$(policy_value "$POLICY" host_fence_selector_key)"
  selector_value="$(policy_value "$POLICY" host_fence_selector_value)"
  daemonset="$(kubectl -n "$(kube_namespace)" get daemonset \
    "$(policy_value "$POLICY" host_fence_daemonset)" -o json)"
  patched="$(jq --arg key "$selector_key" --arg value "$selector_value" '
    .spec.template.spec.nodeSelector = {($key): $value}
  ' <<<"$daemonset")"
  kubectl -n "$(kube_namespace)" replace -f - <<<"$patched" >/dev/null
  kubectl label nodes "$(node0)" "$(node1)" \
    "$selector_key=$selector_value" --overwrite >/dev/null
}

install_host_fence() {
  local holder epoch receipt manifest bootstrap_key bootstrap_value report0 report1 boot0 boot1 host_receipt0 host_receipt1 lease now updated source_hash freeze_hash barrier_hash
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation host-fence "$holder" "$epoch" "$receipt"
  [[ "$(policy_value "$POLICY" host_runtime_restart_required)" == false ]] || \
    fail 'host fence installation would require a forbidden runtime restart'
  manifest="$(repo_root)/$(policy_value "$POLICY" host_fence_manifest)"
  bootstrap_key="$(policy_value "$POLICY" host_fence_bootstrap_selector_key)"
  bootstrap_value="$(policy_value "$POLICY" host_fence_bootstrap_selector_value)"
  [[ "$(kubectl get nodes -l "$bootstrap_key=$bootstrap_value" -o json | jq '.items | length')" == 0 ]] || \
    fail 'host fence bootstrap selector unexpectedly matches a live node'
  device_barrier_config_json | kubectl apply --server-side \
    --field-manager=pireus-spark-pair-device-barrier -f - >/dev/null
  kubectl apply --server-side --force-conflicts \
    --field-manager=pireus-spark-pair-host-fence \
    -f "$manifest" >/dev/null
  wait_for 'inert Spark host fence DaemonSet' host_fence_staged "$holder" "$epoch"
  bind_host_fence_daemonset_uid
  admission_fail_closed "$(lease_json)" || \
    fail 'host fence admission did not bind the inert DaemonSet UID'
  activate_host_fence_daemonset
  kubectl -n "$(kube_namespace)" rollout status \
    daemonset/"$(policy_value "$POLICY" host_fence_daemonset)" \
    --timeout="$(policy_value "$POLICY" operation_timeout_seconds)s" >/dev/null
  wait_for 'exact Spark host fence pair' host_fence_pair_exact "$holder" "$epoch"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  barrier_hash="$(policy_value "$FREEZE" device_barrier_source_sha256)"
  host_fence_exec "$(node0)" capture-baseline
  host_fence_exec "$(node1)" capture-baseline
  host_fence_exec "$(node0)" install-watchdog "$source_hash" "$freeze_hash" "$barrier_hash"
  host_fence_exec "$(node1)" install-watchdog "$source_hash" "$freeze_hash" "$barrier_hash"
  report0="$(host_fence_report "$(node0)")"
  report1="$(host_fence_report "$(node1)")"
  [[ "$report0" == "PIREUS_HOST_FACTS node=$(node0) "* &&
      "$report1" == "PIREUS_HOST_FACTS node=$(node1) "* ]] || \
    fail 'host fence did not emit the canonical receipt pair'
  for field in inventory protected; do
    [[ "$(frame_field "$report0" "$field")" == 1 &&
        "$(frame_field "$report1" "$field")" == 1 ]] || \
      fail "host fence installation field $field is not established on both Sparks"
  done
  boot0="$(frame_field "$report0" boot_id)"
  boot1="$(frame_field "$report1" boot_id)"
  host_receipt0="$(frame_field "$report0" receipt_sha256)"
  host_receipt1="$(frame_field "$report1" receipt_sha256)"
  [[ "$boot0" =~ ^[0-9a-f-]{36}$ && "$boot1" =~ ^[0-9a-f-]{36}$ ]] || \
    fail 'host fence boot identity is malformed'
  [[ "$host_receipt0" =~ ^[0-9a-f]{64}$ && "$host_receipt1" =~ ^[0-9a-f]{64}$ ]] || \
    fail 'host fence receipt digest is malformed'
  lease="$(lease_json)"
  require_lease_context "$holder" "$epoch" 'UNINITIALIZED RECOVERY_REQUIRED' >/dev/null
  now="$(lease_timestamp)"
  updated="$(jq --arg boot0 "$boot0" --arg boot1 "$boot1" \
    --arg host_receipt0 "$host_receipt0" --arg host_receipt1 "$host_receipt1" \
    --arg epoch "$epoch" --arg holder "$holder" --arg now "$now" \
    --arg boot0_key "$(policy_value "$POLICY" host_boot_0_annotation)" \
    --arg boot1_key "$(policy_value "$POLICY" host_boot_1_annotation)" \
    --arg receipt0_key "$(policy_value "$POLICY" host_receipt_0_annotation)" \
    --arg receipt1_key "$(policy_value "$POLICY" host_receipt_1_annotation)" \
    --arg epoch_key "$(policy_value "$POLICY" host_fence_epoch_annotation)" \
    --arg owner_key "$(policy_value "$POLICY" host_fence_owner_annotation)" '
      .metadata.annotations[$boot0_key] = $boot0 |
      .metadata.annotations[$boot1_key] = $boot1 |
      .metadata.annotations[$receipt0_key] = $host_receipt0 |
      .metadata.annotations[$receipt1_key] = $host_receipt1 |
      .metadata.annotations[$epoch_key] = $epoch |
      .metadata.annotations[$owner_key] = $holder |
      .spec.renewTime = $now
    ' <<<"$lease")"
  replace_lease <<<"$updated"
  if [[ "$(jq -r --arg key "$(policy_value "$POLICY" state_annotation)" \
    '.metadata.annotations[$key]' <<<"$updated")" == UNINITIALIZED ]]; then
    bootstrap_journal_step HOST_FENCE_INSTALLED "$holder" "$receipt"
  fi
}

bind_host_pair_intent() {
  local holder="$1" epoch="$2" allowed_states="$3" transaction="$4" pair_digest="$5"
  local lease_uid="$6" base_lease_rv="$7" decision_sha="$8" prepare0="$9" prepare1="${10}"
  local lease updated persisted now intent_rv
  lease="$(lease_json)"
  require_lease_context "$holder" "$epoch" "$allowed_states" >/dev/null
  [[ "$(jq -r '.metadata.uid' <<<"$lease")" == "$lease_uid" &&
      "$(jq -r '.metadata.resourceVersion' <<<"$lease")" == "$base_lease_rv" ]] || \
    fail 'Lease changed before durable host pair intent'
  now="$(lease_timestamp)"
  updated="$(jq --arg transaction "$transaction" --arg pair "$pair_digest" \
    --arg decision "$decision_sha" --arg uid "$lease_uid" --arg base_rv "$base_lease_rv" \
    --arg prepare0 "$prepare0" --arg prepare1 "$prepare1" --arg now "$now" \
    --arg transaction_key "$(policy_value "$POLICY" host_transaction_annotation)" \
    --arg pair_key "$(policy_value "$POLICY" host_pair_digest_annotation)" \
    --arg decision_key "$(policy_value "$POLICY" host_decision_receipt_annotation)" \
    --arg uid_key "$(policy_value "$POLICY" host_lease_uid_annotation)" \
    --arg base_rv_key "$(policy_value "$POLICY" host_intent_base_rv_annotation)" \
    --arg prepare0_key "$(policy_value "$POLICY" host_prepare_0_annotation)" \
    --arg prepare1_key "$(policy_value "$POLICY" host_prepare_1_annotation)" \
    --arg receipt0_key "$(policy_value "$POLICY" host_receipt_0_annotation)" \
    --arg receipt1_key "$(policy_value "$POLICY" host_receipt_1_annotation)" \
    --arg bound_rv_key "$(policy_value "$POLICY" host_lease_resource_version_annotation)" '
      .metadata.annotations[$transaction_key] = $transaction |
      .metadata.annotations[$pair_key] = $pair |
      .metadata.annotations[$decision_key] = $decision |
      .metadata.annotations[$uid_key] = $uid |
      .metadata.annotations[$base_rv_key] = $base_rv |
      .metadata.annotations[$prepare0_key] = $prepare0 |
      .metadata.annotations[$prepare1_key] = $prepare1 |
      del(.metadata.annotations[$receipt0_key]) |
      del(.metadata.annotations[$receipt1_key]) |
      del(.metadata.annotations[$bound_rv_key]) |
      .spec.renewTime = $now
    ' <<<"$lease")"
  persisted="$(replace_lease_json <<<"$updated")"
  intent_rv="$(jq -r '.metadata.resourceVersion // ""' <<<"$persisted")"
  [[ "$intent_rv" =~ ^[1-9][0-9]*$ ]] || fail 'durable host pair intent lacks a resourceVersion'
  printf '%s\n' "$intent_rv"
}

bind_host_commit_receipts() {
  local holder="$1" epoch="$2" allowed_states="$3" expected_mode="$4" receipt="$5"
  local transaction="$6" pair_digest="$7" lease_uid="$8" lease_rv="$9"
  local report0="${10}" report1="${11}" decision_sha source_hash freeze_hash barrier_hash lease updated
  local host_receipt0 host_receipt1 boot0 boot1 expected_valid
  decision_sha="$(sha256_file "$receipt")"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  barrier_hash="$(policy_value "$FREEZE" device_barrier_source_sha256)"
  lease="$(lease_json)"
  require_lease_context "$holder" "$epoch" "$allowed_states" >/dev/null
  [[ "$(jq -r '.metadata.uid' <<<"$lease")" == "$lease_uid" &&
      "$(jq -r '.metadata.resourceVersion' <<<"$lease")" == "$lease_rv" ]] || \
    fail 'Lease changed during host transaction'
  if [[ "$expected_mode" != FENCED ]]; then
    [[ "$(jq -r --arg key "$(policy_value "$POLICY" host_transaction_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$transaction" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_pair_digest_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$pair_digest" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_decision_receipt_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$decision_sha" &&
        "$(jq -r --arg key "$(policy_value "$POLICY" host_lease_uid_annotation)" '.metadata.annotations[$key] // ""' <<<"$lease")" == "$lease_uid" ]] || \
      fail 'durable host pair intent changed before receipt binding'
  fi
  case "$expected_mode" in
    FENCED) expected_valid=0 ;;
    SLURM|K8S) expected_valid=1 ;;
    *) fail "invalid committed host mode: $expected_mode" ;;
  esac
  for report in "$report0" "$report1"; do
    [[ "$(frame_field "$report" grant_mode)" == "$expected_mode" &&
        "$(frame_field "$report" grant_epoch)" == "$epoch" &&
        "$(frame_field "$report" grant_owner)" == "$holder" &&
        "$(frame_field "$report" grant_valid)" == "$expected_valid" &&
        "$(frame_field "$report" transaction_id)" == "$transaction" &&
        "$(frame_field "$report" lease_uid)" == "$lease_uid" &&
        "$(frame_field "$report" lease_resource_version)" == "$lease_rv" &&
        "$(frame_field "$report" decision_receipt_sha256)" == "$decision_sha" &&
        "$(frame_field "$report" pair_digest)" == "$pair_digest" &&
        "$(frame_field "$report" source_sha256)" == "$source_hash" &&
        "$(frame_field "$report" freeze_sha256)" == "$freeze_hash" &&
        "$(frame_field "$report" device_barrier)" == 1 &&
        "$(frame_field "$report" device_barrier_source_sha256)" == "$barrier_hash" &&
        "$(frame_field "$report" device_barrier_binary_sha256)" =~ ^[0-9a-f]{64}$ ]] || \
      fail "host report is not bound to the $expected_mode transaction"
  done
  host_receipt0="$(frame_field "$report0" receipt_sha256)"
  host_receipt1="$(frame_field "$report1" receipt_sha256)"
  boot0="$(frame_field "$report0" boot_id)"
  boot1="$(frame_field "$report1" boot_id)"
  [[ "$host_receipt0" =~ ^[0-9a-f]{64}$ && "$host_receipt1" =~ ^[0-9a-f]{64}$ ]] || \
    fail 'committed host receipt digest is malformed'
  updated="$(jq --arg boot0 "$boot0" --arg boot1 "$boot1" \
    --arg receipt0 "$host_receipt0" --arg receipt1 "$host_receipt1" \
    --arg epoch "$epoch" --arg holder "$holder" --arg transaction "$transaction" \
    --arg pair_digest "$pair_digest" --arg decision "$decision_sha" \
    --arg lease_uid "$lease_uid" --arg lease_rv "$lease_rv" \
    --arg boot0_key "$(policy_value "$POLICY" host_boot_0_annotation)" \
    --arg boot1_key "$(policy_value "$POLICY" host_boot_1_annotation)" \
    --arg receipt0_key "$(policy_value "$POLICY" host_receipt_0_annotation)" \
    --arg receipt1_key "$(policy_value "$POLICY" host_receipt_1_annotation)" \
    --arg epoch_key "$(policy_value "$POLICY" host_fence_epoch_annotation)" \
    --arg owner_key "$(policy_value "$POLICY" host_fence_owner_annotation)" \
    --arg transaction_key "$(policy_value "$POLICY" host_transaction_annotation)" \
    --arg pair_key "$(policy_value "$POLICY" host_pair_digest_annotation)" \
    --arg decision_key "$(policy_value "$POLICY" host_decision_receipt_annotation)" \
    --arg uid_key "$(policy_value "$POLICY" host_lease_uid_annotation)" \
    --arg rv_key "$(policy_value "$POLICY" host_lease_resource_version_annotation)" \
    --arg prepare0_key "$(policy_value "$POLICY" host_prepare_0_annotation)" \
    --arg prepare1_key "$(policy_value "$POLICY" host_prepare_1_annotation)" \
    --arg base_rv_key "$(policy_value "$POLICY" host_intent_base_rv_annotation)" \
    --arg expected_mode "$expected_mode" '
      .metadata.annotations[$boot0_key] = $boot0 |
      .metadata.annotations[$boot1_key] = $boot1 |
      .metadata.annotations[$receipt0_key] = $receipt0 |
      .metadata.annotations[$receipt1_key] = $receipt1 |
      .metadata.annotations[$epoch_key] = $epoch |
      .metadata.annotations[$owner_key] = $holder |
      .metadata.annotations[$transaction_key] = $transaction |
      .metadata.annotations[$pair_key] = $pair_digest |
      .metadata.annotations[$decision_key] = $decision |
      .metadata.annotations[$uid_key] = $lease_uid |
      .metadata.annotations[$rv_key] = $lease_rv |
      if $expected_mode == "FENCED" then
        del(.metadata.annotations[$prepare0_key]) |
        del(.metadata.annotations[$prepare1_key]) |
        del(.metadata.annotations[$base_rv_key])
      else . end
    ' <<<"$lease")"
  replace_lease <<<"$updated"
}

fence_host_pair() {
  local holder epoch receipt report0 report1 lease source_hash freeze_hash
  local decision_sha transaction lease_uid lease_rv allowed_states
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation host-pair "$holder" "$epoch" "$receipt"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  decision_sha="$(sha256_file "$receipt")"
  allowed_states='UNINITIALIZED SLURM_OWNED DRAINING_SLURM SLURM_QUIESCENT K8S_RELEASING VERIFYING_GPU_CLEAN RECOVERY_REQUIRED'
  require_lease_context "$holder" "$epoch" "$allowed_states" >/dev/null
  lease="$(lease_json)"
  lease_uid="$(jq -r '.metadata.uid' <<<"$lease")"
  lease_rv="$(jq -r '.metadata.resourceVersion' <<<"$lease")"
  transaction="$(printf 'mode=FENCED\nepoch=%s\nholder=%s\nlease_uid=%s\nlease_rv=%s\ndecision=%s\n' \
    "$epoch" "$holder" "$lease_uid" "$lease_rv" "$decision_sha" | sha256sum | cut -d ' ' -f 1)"
  if ! host_fence_exec "$(node0)" fence "$epoch" "$holder" "$source_hash" "$freeze_hash" \
    "$transaction" "$lease_uid" "$lease_rv" "$decision_sha"; then
    host_fence_exec "$(node1)" fence "$epoch" "$holder" "$source_hash" "$freeze_hash" \
      "$transaction" "$lease_uid" "$lease_rv" "$decision_sha" >/dev/null 2>&1 || true
    fail 'first Spark host fence failed'
  fi
  if ! host_fence_exec "$(node1)" fence "$epoch" "$holder" "$source_hash" "$freeze_hash" \
    "$transaction" "$lease_uid" "$lease_rv" "$decision_sha"; then
    host_fence_exec "$(node0)" fence "$epoch" "$holder" "$source_hash" "$freeze_hash" \
      "$transaction" "$lease_uid" "$lease_rv" "$decision_sha" >/dev/null 2>&1 || true
    fail 'second Spark host fence failed; first host was re-fenced'
  fi
  report0="$(host_fence_report "$(node0)")"
  report1="$(host_fence_report "$(node1)")"
  bind_host_commit_receipts "$holder" "$epoch" "$allowed_states" FENCED "$receipt" \
    "$transaction" none "$lease_uid" "$lease_rv" "$report0" "$report1"
}

refence_host_pair_transaction() {
  local holder="$1" epoch="$2" source_hash="$3" freeze_hash="$4"
  local transaction="$5" lease_uid="$6" lease_rv="$7" decision_sha="$8"
  local report0 report1 failed=0
  host_fence_exec "$(node0)" fence "$epoch" "$holder" "$source_hash" "$freeze_hash" \
    "$transaction" "$lease_uid" "$lease_rv" "$decision_sha" >/dev/null || failed=1
  host_fence_exec "$(node1)" fence "$epoch" "$holder" "$source_hash" "$freeze_hash" \
    "$transaction" "$lease_uid" "$lease_rv" "$decision_sha" >/dev/null || failed=1
  report0="$(host_fence_report "$(node0)" 2>/dev/null || true)"
  report1="$(host_fence_report "$(node1)" 2>/dev/null || true)"
  [[ "$(frame_field "$report0" grant_mode 2>/dev/null || true)" == FENCED &&
      "$(frame_field "$report1" grant_mode 2>/dev/null || true)" == FENCED &&
      "$(frame_field "$report0" grant_valid 2>/dev/null || true)" == 0 &&
      "$(frame_field "$report1" grant_valid 2>/dev/null || true)" == 0 ]] || failed=1
  [[ $failed -eq 0 ]]
}

grant_host_pair() {
  local mode="$1" kind holder epoch receipt report0 report1 source_hash freeze_hash
  local lease lease_uid lease_rv intent_lease_rv decision_sha transaction prepared0 prepared1 prepare_receipt0 prepare_receipt1 pair_digest allowed_states
  shift
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  case "$mode" in
    SLURM) kind=host-grant-slurm ;;
    K8S) kind=host-grant-k8s ;;
    *) fail "unknown host grant mode: $mode" ;;
  esac
  guard_mutation "$kind" "$holder" "$epoch" "$receipt"
  source_hash="$(policy_value "$FREEZE" authority_sha256)"
  freeze_hash="$(sha256_file "$FREEZE")"
  decision_sha="$(sha256_file "$receipt")"
  if [[ "$mode" == SLURM ]]; then
    allowed_states='UNINITIALIZED SLURM_RESTORING RECOVERY_REQUIRED'
  else
    allowed_states='SLURM_QUIESCENT DETACHING_SLURMD K8S_RESERVING K8S_OWNED'
  fi
  require_lease_context "$holder" "$epoch" "$allowed_states" >/dev/null
  lease="$(lease_json)"
  lease_uid="$(jq -r '.metadata.uid' <<<"$lease")"
  lease_rv="$(jq -r '.metadata.resourceVersion' <<<"$lease")"
  transaction="$(printf 'mode=%s\nepoch=%s\nholder=%s\nlease_uid=%s\nlease_rv=%s\ndecision=%s\n' \
    "$mode" "$epoch" "$holder" "$lease_uid" "$lease_rv" "$decision_sha" | sha256sum | cut -d ' ' -f 1)"
  prepared0="$(host_fence_exec "$(node0)" prepare "$mode" "$epoch" "$holder" \
    "$source_hash" "$freeze_hash" "$transaction" "$lease_uid" "$lease_rv" "$decision_sha")" || \
    fail "first Spark rejected $mode prepare"
  if ! prepared1="$(host_fence_exec "$(node1)" prepare "$mode" "$epoch" "$holder" \
    "$source_hash" "$freeze_hash" "$transaction" "$lease_uid" "$lease_rv" "$decision_sha")"; then
    fail "second Spark rejected $mode prepare; no commit was attempted"
  fi
  prepare_receipt0="$(frame_field "$prepared0" prepare_receipt_sha256)"
  prepare_receipt1="$(frame_field "$prepared1" prepare_receipt_sha256)"
  [[ "$prepared0" == "PIREUS_HOST_PREPARED node=$(node0) "* &&
      "$prepared1" == "PIREUS_HOST_PREPARED node=$(node1) "* &&
      "$prepare_receipt0" =~ ^[0-9a-f]{64}$ && "$prepare_receipt1" =~ ^[0-9a-f]{64}$ ]] || \
    fail 'host prepare receipt pair is malformed'
  pair_digest="$(printf 'transaction_id=%s\nlease_uid=%s\nbase_lease_resource_version=%s\nnode0_prepare=%s\nnode1_prepare=%s\n' \
    "$transaction" "$lease_uid" "$lease_rv" "$prepare_receipt0" "$prepare_receipt1" | \
    sha256sum | cut -d ' ' -f 1)"
  intent_lease_rv="$(bind_host_pair_intent "$holder" "$epoch" "$allowed_states" \
    "$transaction" "$pair_digest" "$lease_uid" "$lease_rv" "$decision_sha" \
    "$prepare_receipt0" "$prepare_receipt1")"
  if ! host_fence_exec "$(node0)" commit "$mode" "$epoch" "$holder" "$source_hash" \
    "$freeze_hash" "$transaction" "$lease_uid" "$lease_rv" "$decision_sha" \
    "$prepare_receipt0" "$prepare_receipt1" "$pair_digest" "$intent_lease_rv"; then
    refence_host_pair_transaction "$holder" "$epoch" "$source_hash" "$freeze_hash" \
      "$transaction" "$lease_uid" "$intent_lease_rv" "$decision_sha" || \
      fail "first Spark rejected $mode activation and pair re-fence could not be proven"
    fail "first Spark rejected $mode activation; pair is proven fenced"
  fi
  if ! host_fence_exec "$(node1)" commit "$mode" "$epoch" "$holder" "$source_hash" \
    "$freeze_hash" "$transaction" "$lease_uid" "$lease_rv" "$decision_sha" \
    "$prepare_receipt0" "$prepare_receipt1" "$pair_digest" "$intent_lease_rv"; then
    refence_host_pair_transaction "$holder" "$epoch" "$source_hash" "$freeze_hash" \
      "$transaction" "$lease_uid" "$intent_lease_rv" "$decision_sha" || \
      fail "partial $mode activation and pair re-fence could not be proven"
    fail "partial $mode activation; pair is proven fenced and recovery is required"
  fi
  report0="$(host_fence_report "$(node0)")"
  report1="$(host_fence_report "$(node1)")"
  if [[ "$(frame_field "$report0" grant_mode)" != "$mode" ||
        "$(frame_field "$report1" grant_mode)" != "$mode" ||
        "$(frame_field "$report0" grant_valid)" != 1 ||
        "$(frame_field "$report1" grant_valid)" != 1 ]]; then
    refence_host_pair_transaction "$holder" "$epoch" "$source_hash" "$freeze_hash" \
      "$transaction" "$lease_uid" "$intent_lease_rv" "$decision_sha" || \
      fail "$mode activation verification failed and pair re-fence could not be proven"
    fail "$mode activation verification failed; pair is proven fenced"
  fi
  if ! (bind_host_commit_receipts "$holder" "$epoch" "$allowed_states" "$mode" "$receipt" \
    "$transaction" "$pair_digest" "$lease_uid" "$intent_lease_rv" "$report0" "$report1"); then
    refence_host_pair_transaction "$holder" "$epoch" "$source_hash" "$freeze_hash" \
      "$transaction" "$lease_uid" "$intent_lease_rv" "$decision_sha" || \
      fail "$mode receipt CAS failed and pair re-fence could not be proven"
    fail "$mode receipt CAS failed; pair is proven fenced"
  fi
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
  now="$(lease_timestamp)"
  updated="$(jq --arg generation "$generation" --arg now "$now" \
    --arg key "$(policy_value "$POLICY" nodeset_generation_annotation)" '
      .metadata.annotations[$key] = $generation |
      .spec.renewTime = $now
    ' <<<"$lease")"
  replace_lease <<<"$updated"
  sync_admission_projection "$updated"
}

gpu_bound_nodeset_json() {
  local nodeset="$1" selector_key selector_value taint_key taint_value taint_effect gpu
  selector_key="$(policy_value "$POLICY" slurmd_selector_key)"
  selector_value="$(policy_value "$POLICY" slurmd_selector_value)"
  taint_key="$(policy_value "$POLICY" spark_taint_key)"
  taint_value="$(policy_value "$POLICY" spark_taint_value)"
  taint_effect="$(policy_value "$POLICY" spark_taint_effect)"
  gpu="$(policy_value "$POLICY" slurmd_gpu_resource)"
  jq --arg selector_key "$selector_key" --arg selector_value "$selector_value" \
    --arg taint_key "$taint_key" --arg taint_value "$taint_value" --arg taint_effect "$taint_effect" \
    --arg gpu "$gpu" '
      del(.spec.slurmd.lifecycle) |
      del(.spec.template.spec.initContainers[]?.lifecycle) |
      .spec.template.spec.runtimeClassName = "nvidia" |
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
    ' <<<"$nodeset"
}

install_gpu_bound_slurmd() {
  local holder epoch receipt nodeset patched selector_key selector_value
  holder="$(arg_value --holder "$@")"
  epoch="$(arg_value --epoch "$@")"
  receipt="$(arg_value --receipt "$@")"
  guard_mutation bootstrap-slurmd "$holder" "$epoch" "$receipt"
  selector_key="$(policy_value "$POLICY" slurmd_selector_key)"
  selector_value="$(policy_value "$POLICY" slurmd_selector_value)"
  nodeset="$(kubectl -n "$(policy_value "$POLICY" nodeset_namespace)" get nodeset \
    "$(policy_value "$POLICY" nodeset_name)" -o json)"
  patched="$(gpu_bound_nodeset_json "$nodeset")"
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
  if ! slurm_resumed "$holder" "$epoch"; then
    slurm_exec scontrol update NodeName="$(slurm0),$(slurm1)" State=RESUME >/dev/null
    wait_for 'both Slurm nodes to resume idle' slurm_resumed "$holder" "$epoch"
  fi
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
  now="$(lease_timestamp)"
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
    bootstrap-migration-facts) bootstrap_migration_facts "$@" ;;
    lease-acquire) lease_acquire "$@" ;;
    lease-recovery-acquire) lease_recovery_acquire "$@" ;;
    lease-bootstrap-recovery-acquire) lease_bootstrap_recovery_acquire "$@" ;;
    bootstrap-migrate-freeze) bootstrap_migrate_freeze "$@" ;;
    lease-transition) lease_transition "$@" ;;
    lease-renew) lease_renew "$@" ;;
    material-keepalive) material_keepalive "$@" ;;
    drain-slurm) drain_slurm "$@" ;;
    install-fence) install_fence "$@" ;;
    install-host-fence) install_host_fence "$@" ;;
    fence-host-pair) fence_host_pair "$@" ;;
    grant-host-slurm) grant_host_pair SLURM "$@" ;;
    grant-host-k8s) grant_host_pair K8S "$@" ;;
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

if [[ "${SOUNIO_SPARK_PAIR_BACKEND_LIBRARY_MODE:-0}" == 1 ]]; then
  return 0 2>/dev/null || exit 0
fi

main "$@"
