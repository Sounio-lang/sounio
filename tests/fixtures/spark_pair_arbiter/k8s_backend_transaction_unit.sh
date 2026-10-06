#!/usr/bin/env bash

set -euo pipefail
umask 077

[[ $# -eq 2 ]] || {
  printf 'usage: %s BACKEND WORK_DIR\n' "$0" >&2
  exit 64
}

BACKEND="$1"
WORK_DIR="$2"
mkdir -p "$WORK_DIR"
export SOUNIO_SPARK_PAIR_BACKEND_LIBRARY_MODE=1

ROOT_DIR="$(cd "$(dirname "$BACKEND")/../.." && pwd -P)"
HOST_FENCE_MANIFEST="$ROOT_DIR/tools/cluster/spark_pair_host_fence.yaml"
HOST_FENCE_SCRIPT="$WORK_DIR/host-fence.sh"
LIVE_POLICY="$ROOT_DIR/tools/cluster/spark_pair_arbiter.policy.v1"
LIVE_ADMISSION="$ROOT_DIR/tools/cluster/spark_pair_arbiter_admission.yaml"
awk '
  /^  host-fence\.sh: \|$/ { in_script=1; next }
  in_script && /^---$/ { exit }
  in_script { sub(/^    /, ""); print }
' "$HOST_FENCE_MANIFEST" > "$HOST_FENCE_SCRIPT"
grep -Fq 'source="${PIREUS_HOST_FENCE_INSTALL_SOURCE:-$0}"' "$HOST_FENCE_SCRIPT" || {
  printf 'backend-transaction-unit: watchdog verification ignores the authenticated install source\n' >&2
  exit 1
}
if grep -Fq 'source="$0"' "$HOST_FENCE_SCRIPT"; then
  printf 'backend-transaction-unit: stale mounted script can override the authenticated install source\n' >&2
  exit 1
fi
grep -Fq 'readonly DEVICE_INVENTORY_NAMES=nvidia,drm,dma_heap,nvidia-uvm,nvidia-caps' \
  "$HOST_FENCE_SCRIPT" || {
  printf 'backend-transaction-unit: host inventory is not bound to device names\n' >&2
  exit 1
}
grep -Fq 'readonly DEVICE_BARRIER_NAMES=nvidia-uvm,nvidia-caps' \
  "$HOST_FENCE_SCRIPT" || {
  printf 'backend-transaction-unit: compute deny set is not bound to device names\n' >&2
  exit 1
}
if grep -Eq 'readonly DEVICE_(INVENTORY|BARRIER)_MAJORS=' "$HOST_FENCE_SCRIPT"; then
  printf 'backend-transaction-unit: host fence still hardcodes node-specific device majors\n' >&2
  exit 1
fi
grep -Fq "'\$2 == wanted { print \$1 }' \"\$HOST_ROOT/proc/devices\"" \
  "$HOST_FENCE_SCRIPT" || {
  printf 'backend-transaction-unit: device names are not resolved through host /proc/devices\n' >&2
  exit 1
}
major_fixture="$WORK_DIR/device-major-host"
mkdir -p "$major_fixture/proc"
cat > "$major_fixture/proc/devices" <<'EOF'
Character devices:
195 nvidia
226 drm
247 dma_heap
498 nvidia-uvm
500 nvidia-nvswitch
501 nvidia-nvlink
502 nvidia-caps
503 nvidia-caps-imex
EOF
(
  export NODE_NAME=spark-unit PIREUS_HOST_ROOT="$major_fixture"
  export PIREUS_HOST_FENCE_LIBRARY_MODE=1
  # shellcheck source=/dev/null
  source "$HOST_FENCE_SCRIPT"
  [[ "$(device_inventory_majors)" == 195,226,247,498,502 ]]
  [[ "$(device_barrier_majors)" == 498,502 ]]
  runtime_boot_gate_text kubelet.service | grep -Fq 'Wants=pireus-spark-host-fence-boot.service'
  ! runtime_boot_gate_text kubelet.service | grep -Fq 'Requires='
  runtime_boot_gate_text containerd.service | grep -Fq 'Wants=pireus-spark-host-fence-boot.service'
  runtime_boot_gate_text docker.service | grep -Fq 'Requires=pireus-spark-host-fence-boot.service'
)
sed -i 's/^501 nvidia-nvlink$/500 nvidia-nvlink/; s/^502 nvidia-caps$/501 nvidia-caps/' \
  "$major_fixture/proc/devices"
(
  export NODE_NAME=spark-unit PIREUS_HOST_ROOT="$major_fixture"
  export PIREUS_HOST_FENCE_LIBRARY_MODE=1
  # shellcheck source=/dev/null
  source "$HOST_FENCE_SCRIPT"
  [[ "$(device_inventory_majors)" == 195,226,247,498,501 ]]
  [[ "$(device_barrier_majors)" == 498,501 ]]
)
sed -i 's/^501 nvidia-caps$/498 nvidia-caps/' "$major_fixture/proc/devices"
if (
  export NODE_NAME=spark-unit PIREUS_HOST_ROOT="$major_fixture"
  export PIREUS_HOST_FENCE_LIBRARY_MODE=1
  # shellcheck source=/dev/null
  source "$HOST_FENCE_SCRIPT"
  device_barrier_majors >/dev/null
); then
  printf 'backend-transaction-unit: duplicate resolved major was accepted\n' >&2
  exit 1
fi
expected_host_configmap="$(awk -F= '$1 == "host_fence_configmap" { print $2 }' "$LIVE_POLICY")"
[[ "$(grep -Fc "name: $expected_host_configmap" "$HOST_FENCE_MANIFEST")" == 2 ]] || {
  printf 'backend-transaction-unit: host fence manifest does not bind both content-addressed references\n' >&2
  exit 1
}
[[ "$(grep -Fc "$expected_host_configmap" "$LIVE_ADMISSION")" == 3 ]] || {
  printf 'backend-transaction-unit: admission policy does not bind all host fence references\n' >&2
  exit 1
}
spark_pair_admission_policy="$(awk '
  /^  name: pireus-spark-pair-fence$/ { capture = 1 }
  capture { print }
  capture && /^kind: ValidatingAdmissionPolicyBinding$/ { exit }
' "$LIVE_ADMISSION")"
[[ "$(grep -Fc 'name: isSparkSlurmd' <<<"$spark_pair_admission_policy")" == 1 ]] &&
  grep -Fq "object.metadata.labels['app.kubernetes.io/instance'] == 'slurm-pilot-worker-spark'" \
    <<<"$spark_pair_admission_policy" &&
  grep -Fq 'variables.isSparkSlurmd ||' <<<"$spark_pair_admission_policy" || {
  printf 'backend-transaction-unit: Slurmd identity variable is outside the active Spark pair policy\n' >&2
  exit 1
}
grep -Fq '!has(object.spec.containers[0].lifecycle.postStart)' <<<"$spark_pair_admission_policy" &&
  grep -Fq "scontrol update nodename=\$(hostname) state=down reason='Pod is terminating' && scontrol delete nodename=\$(hostname);" \
    <<<"$spark_pair_admission_policy" || {
  printf 'backend-transaction-unit: admitted Slurmd role is not bound to the operator default preStop\n' >&2
  exit 1
}
grep -Fq '(has(t.key) ?' "$LIVE_ADMISSION" &&
  grep -Fq "t.operator == 'Exists')" "$LIVE_ADMISSION" || {
  printf 'backend-transaction-unit: keyless Exists toleration can raise a CEL evaluation error\n' >&2
  exit 1
}
grep -Fq "(!has(object.spec.nodeName) || object.spec.nodeName == '')" "$LIVE_ADMISSION" || {
  printf 'backend-transaction-unit: a bound non-Spark Pod can be captured by a broad toleration\n' >&2
  exit 1
}
if grep -Fq 'has(v.emptyDir.sizeLimit)' "$LIVE_ADMISSION"; then
  printf 'backend-transaction-unit: emptyDir sizeLimit uses a warning-bearing static optional field\n' >&2
  exit 1
fi
grep -Fq '"$watchdog_usec" == 1min' "$HOST_FENCE_SCRIPT" || {
  printf 'backend-transaction-unit: systemd normalized watchdog duration is rejected\n' >&2
  exit 1
}

# shellcheck source=/dev/null
source "$BACKEND"

POLICY="$LIVE_POLICY"
nodeset_with_lifecycle='{"spec":{"slurmd":{"lifecycle":{"postStart":{"exec":{"command":["/bin/bash","-lc","ldconfig"]}}},"resources":{"requests":{"cpu":"1"},"limits":{"cpu":"2"}}},"template":{"spec":{"nodeSelector":{},"tolerations":[],"initContainers":[{"name":"logfile","lifecycle":{"postStart":{"exec":{"command":["false"]}}}}]}}}}'
patched_nodeset="$(gpu_bound_nodeset_json "$nodeset_with_lifecycle")"
jq -e '
  (.spec.slurmd | has("lifecycle") | not) and
  (.spec.template.spec.initContainers[0] | has("lifecycle") | not) and
  .spec.template.spec.runtimeClassName == "nvidia" and
  .spec.slurmd.resources.requests["nvidia.com/gpu"] == "1" and
  .spec.slurmd.resources.limits["nvidia.com/gpu"] == "1" and
  .spec.template.spec.nodeSelector["pireus.sounio.dev/slurm-enabled"] == "true" and
  any(.spec.template.spec.tolerations[];
    .key == "pireus.sounio.dev/spark-pair" and
    .value == "reserved" and .effect == "NoSchedule")
' <<<"$patched_nodeset" >/dev/null || {
  printf 'backend-transaction-unit: GPU-bound NodeSet is not normalized to the admitted Slurmd role\n' >&2
  exit 1
}

report_bridge_definition="$(declare -f host_fence_report_with_host_tmp)"
grep -Fq '/host/usr/local/lib/pireus/spark-pair-host-fence report' \
  <<<"$report_bridge_definition" || {
  printf 'backend-transaction-unit: legacy bridge does not execute the installed watchdog\n' >&2
  exit 1
}
grep -Fq 'PIREUS_HOST_FENCE_INSTALL_SOURCE=/host/usr/local/lib/pireus/spark-pair-host-fence' \
  <<<"$report_bridge_definition" || {
  printf 'backend-transaction-unit: installed watchdog report can fall back to the legacy mount\n' >&2
  exit 1
}
if grep -Fq 'host_fence_exec_with_host_tmp' <<<"$report_bridge_definition"; then
  printf 'backend-transaction-unit: legacy mounted script can certify its replacement\n' >&2
  exit 1
fi
install_fence_definition="$(declare -f install_fence)"
grep -Fq 'kubectl apply --server-side --force-conflicts' \
  <<<"$install_fence_definition" || {
  printf 'backend-transaction-unit: admission projection cannot adopt legacy field ownership\n' >&2
  exit 1
}
stage_definition="$(declare -f stage_existing_host_fence_for_bootstrap)"
current_line="$(grep -n 'host_fence_pair_exact' <<<"$stage_definition" | head -1 | cut -d: -f1)"
legacy_line="$(grep -n 'host_fence_legacy_runtime_bridge_exact' <<<"$stage_definition" | head -1 | cut -d: -f1)"
[[ -n "$current_line" && -n "$legacy_line" && "$current_line" -lt "$legacy_line" ]] || {
  printf 'backend-transaction-unit: current host fence is not resumable before legacy migration\n' >&2
  exit 1
}
[[ "$(grep -c 'host_fence_install_current_watchdog_pair_via_bridge' <<<"$stage_definition")" == 2 ]] || {
  printf 'backend-transaction-unit: current and legacy staging do not both rebind the watchdog\n' >&2
  exit 1
}
grep -Fq 'host_fence_pair_exact "$holder" "$epoch" 0' <<<"$stage_definition" || {
  printf 'backend-transaction-unit: epoch rollover cannot authenticate an unready current bridge\n' >&2
  exit 1
}
grep -Fq 'host_fence_legacy_runtime_bridge_exact "$holder" "$epoch" 0' \
  <<<"$stage_definition" || {
  printf 'backend-transaction-unit: epoch rollover cannot authenticate an unready legacy bridge\n' >&2
  exit 1
}
legacy_bridge_definition="$(declare -f host_fence_legacy_runtime_bridge_exact)"
grep -Fq '(.spec.template.spec.containers[0].volumeMounts | length) == 4' \
  <<<"$legacy_bridge_definition" &&
  grep -Fq '.name == "runtime-tmp" and .mountPath == "/tmp"' \
    <<<"$legacy_bridge_definition" &&
  grep -Fq '.name == "runtime-tmp" and .emptyDir.sizeLimit == "64Mi"' \
    <<<"$legacy_bridge_definition" || {
  printf 'backend-transaction-unit: four-volume content-addressed legacy bridge is not authenticated\n' >&2
  exit 1
}
pair_exact_definition="$(declare -f host_fence_pair_exact)"
grep -Fq 'local require_ready="${3:-1}"' <<<"$pair_exact_definition" || {
  printf 'backend-transaction-unit: ordinary host facts no longer require Ready agents\n' >&2
  exit 1
}
sync_definition="$(declare -f sync_admission_projection)"
grep -Fq '.data.hostFenceDaemonSetUid = $uid' <<<"$sync_definition" || {
  printf 'backend-transaction-unit: admission projection does not rebind an existing host fence UID\n' >&2
  exit 1
}

POLICY="$WORK_DIR/policy"
FREEZE="$WORK_DIR/freeze"
receipt="$WORK_DIR/decision.receipt"
: > "$POLICY"
: > "$FREEZE"
: > "$receipt"

hex64() {
  local value="$1" index
  for ((index = 0; index < 64; index++)); do
    printf '%s' "$value"
  done
  printf '\n'
}
source_sha="$(hex64 a)"
freeze_sha="$(hex64 b)"
old_freeze_sha="$(hex64 c)"
decision_sha="$(hex64 d)"
prepare0="$(hex64 e)"
prepare1="$(hex64 f)"
barrier_sha="$(hex64 1)"
lease_uid=22222222-2222-4222-8222-222222222222

policy_value() {
  case "$2" in
    authority_sha256) printf '%s\n' "$source_sha" ;;
    state_annotation) printf 'pireus.sounio.dev/state\n' ;;
    source_hash_annotation) printf 'pireus.sounio.dev/sounio-source-sha256\n' ;;
    freeze_hash_annotation) printf 'pireus.sounio.dev/semantics-freeze-sha256\n' ;;
    device_barrier_source_sha256) printf '%s\n' "$barrier_sha" ;;
    bootstrap_migration_ancestor_sha256) printf '%s\n' "$(hex64 3)" ;;
    *) printf 'unit-value\n' ;;
  esac
}
sha256_file() {
  if [[ "$1" == "$FREEZE" ]]; then printf '%s\n' "$freeze_sha"; else printf '%s\n' "$decision_sha"; fi
}
guard_mutation() { return 0; }
require_lease_context() { return 0; }
node0() { printf 'spark-3c59\n'; }
node1() { printf 'spark-8e54\n'; }
admission_config_json() {
  printf '%s\n' '{"data":{"hostFenceDaemonSetUid":"33333333-3333-4333-8333-333333333333"}}'
}

allowed_pods='{"items":[{"metadata":{"namespace":"slurm-pilot","labels":{"app.kubernetes.io/name":"slurmd","app.kubernetes.io/instance":"slurm-pilot-worker-spark"}},"spec":{"nodeName":"spark-3c59"}},{"metadata":{"namespace":"kube-system","labels":{}},"spec":{"nodeName":"spark-8e54"}},{"metadata":{"namespace":"beagle","labels":{"pireus.sounio.dev/spark-pair-infrastructure":"true"},"ownerReferences":[{"apiVersion":"apps/v1","kind":"DaemonSet","name":"pireus-spark-host-fence","uid":"33333333-3333-4333-8333-333333333333","controller":true,"blockOwnerDeletion":true}]},"spec":{"nodeName":"spark-3c59","serviceAccountName":"pireus-spark-host-fence"}}]}'
unexpected_gpu_consumers_zero "$allowed_pods" 7 holder || {
  printf 'backend-transaction-unit: allowed infrastructure was classified as an unexpected consumer\n' >&2
  exit 1
}
forged_host_fence="$(jq '.items += [{"metadata":{"namespace":"beagle","labels":{"pireus.sounio.dev/spark-pair-infrastructure":"true"},"ownerReferences":[{"apiVersion":"apps/v1","kind":"DaemonSet","name":"pireus-spark-host-fence","uid":"44444444-4444-4444-8444-444444444444","controller":true,"blockOwnerDeletion":true}]},"spec":{"nodeName":"spark-8e54","serviceAccountName":"pireus-spark-host-fence"}}]' <<<"$allowed_pods")"
if unexpected_gpu_consumers_zero "$forged_host_fence" 7 holder; then
  printf 'backend-transaction-unit: forged host fence Pod was admitted on the Spark pair\n' >&2
  exit 1
fi
unexpected_pods="$(jq '.items += [{"metadata":{"namespace":"beagle","name":"unexpected","labels":{}},"spec":{"nodeName":"spark-3c59"}}]' <<<"$allowed_pods")"
if unexpected_gpu_consumers_zero "$unexpected_pods" 7 holder; then
  printf 'backend-transaction-unit: unexpected Pod was admitted on the Spark pair\n' >&2
  exit 1
fi
[[ "$(lease_timestamp)" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.000000Z$ ]] || {
  printf 'backend-transaction-unit: Lease timestamp is not Kubernetes MicroTime\n' >&2
  exit 1
}
lease_is_live '{"spec":{"renewTime":"2099-01-01T00:00:00.000000Z","leaseDurationSeconds":30}}' || {
  printf 'backend-transaction-unit: Kubernetes MicroTime Lease was not parsed\n' >&2
  exit 1
}
slurm_states_drained $'NodeName=n0 State=IDLE+DRAIN\nNodeName=n1 State=DOWN+DRAIN+NOT_RESPONDING' || {
  printf 'backend-transaction-unit: DOWN+DRAIN was not classified as fail-closed drained\n' >&2
  exit 1
}
if slurm_states_drained $'NodeName=n0 State=IDLE+DRAIN\nNodeName=n1 State=DOWN+NOT_RESPONDING'; then
  printf 'backend-transaction-unit: DOWN without DRAIN was accepted\n' >&2
  exit 1
fi
if slurm_states_drained $'NodeName=n0 State=ALLOCATED+DRAIN\nNodeName=n1 State=IDLE+DRAIN'; then
  printf 'backend-transaction-unit: allocated drained node was accepted\n' >&2
  exit 1
fi
vap_clean='{"metadata":{"generation":1},"status":{"observedGeneration":1,"typeChecking":{"expressionWarnings":[]}}}'
vap_known_warning='{"metadata":{"generation":1},"status":{"observedGeneration":1,"typeChecking":{"expressionWarnings":[{"fieldRef":"spec.validations[0].expression","warning":"known multi-kind CEL warning"}]}}}'
vap_unknown_warning='{"metadata":{"generation":1},"status":{"observedGeneration":1,"typeChecking":{"expressionWarnings":[{"fieldRef":"spec.matchConditions[0].expression","warning":"unexpected"}]}}}'
vap_typechecking_acceptable "$vap_clean" || {
  printf 'backend-transaction-unit: clean VAP typecheck was rejected\n' >&2
  exit 1
}
vap_typechecking_acceptable "$vap_known_warning" || {
  printf 'backend-transaction-unit: bounded VAP warning was rejected\n' >&2
  exit 1
}
if vap_typechecking_acceptable "$vap_unknown_warning"; then
  printf 'backend-transaction-unit: unbounded VAP warning was accepted\n' >&2
  exit 1
fi
host_vap_warning='{"metadata":{"generation":1},"status":{"observedGeneration":1,"typeChecking":{"expressionWarnings":[{"fieldRef":"spec.validations[5].expression","warning":"undefined field requests"}]}}}'
host_vap_typechecking_acceptable "$vap_clean" || {
  printf 'backend-transaction-unit: clean host VAP typecheck was rejected\n' >&2
  exit 1
}
if host_vap_typechecking_acceptable "$host_vap_warning"; then
  printf 'backend-transaction-unit: warning-bearing host VAP was accepted\n' >&2
  exit 1
fi

host_fence_daemonset='{"spec":{"template":{"spec":{"containers":[{"securityContext":{"readOnlyRootFilesystem":true},"volumeMounts":[{"name":"fence-script","mountPath":"/fence","readOnly":true},{"name":"device-barrier-source","mountPath":"/barrier","readOnly":true},{"name":"host-root","mountPath":"/host"},{"name":"runtime-tmp","mountPath":"/tmp","readOnly":false}]}],"volumes":[{"name":"fence-script","configMap":{"name":"unit-value","defaultMode":365}},{"name":"device-barrier-source","configMap":{"name":"unit-value","defaultMode":292}},{"name":"host-root","hostPath":{"path":"/","type":"Directory"}},{"name":"runtime-tmp","emptyDir":{"sizeLimit":"64Mi"}}]}}}}'
host_fence_runtime_contract_exact "$host_fence_daemonset" || {
  printf 'backend-transaction-unit: exact writable runtime volume was rejected\n' >&2
  exit 1
}
for mutation in missing-tmp-mount wrong-tmp-size readonly-tmp writable-rootfs; do
  case "$mutation" in
    missing-tmp-mount)
      candidate="$(jq 'del(.spec.template.spec.containers[0].volumeMounts[3])' <<<"$host_fence_daemonset")"
      ;;
    wrong-tmp-size)
      candidate="$(jq '.spec.template.spec.volumes[3].emptyDir.sizeLimit = "32Mi"' <<<"$host_fence_daemonset")"
      ;;
    readonly-tmp)
      candidate="$(jq '.spec.template.spec.containers[0].volumeMounts[3].readOnly = true' <<<"$host_fence_daemonset")"
      ;;
    writable-rootfs)
      candidate="$(jq '.spec.template.spec.containers[0].securityContext.readOnlyRootFilesystem = false' <<<"$host_fence_daemonset")"
      ;;
  esac
  if host_fence_runtime_contract_exact "$candidate"; then
    printf 'backend-transaction-unit: %s host fence mutation was accepted\n' "$mutation" >&2
    exit 1
  fi
done

migration_lease="$(jq -n \
  --arg source "$source_sha" --arg freeze "$old_freeze_sha" '
  {
    metadata: {annotations: {
      "pireus.sounio.dev/state": "UNINITIALIZED",
      "pireus.sounio.dev/sounio-source-sha256": $source,
      "pireus.sounio.dev/semantics-freeze-sha256": $freeze
    }},
    spec: {
      renewTime: "2000-01-01T00:00:00.000000Z",
      leaseDurationSeconds: 30
    }
  }
')"
migration_journal="$(jq -n \
  --arg source "$source_sha" --arg freeze "$old_freeze_sha" '
  {data: {
    step: "HOST_FENCE_INSTALLED",
    sounioSourceSha256: $source,
    semanticsFreezeSha256: $freeze
  }}
')"
migration_slurm_nodes=$'NodeName=spark-3c59 CPUAlloc=0 AllocMem=0 AllocTRES= CfgTRES=cpu=20 State=IDLE+DRAIN\nNodeName=spark-8e54 CPUAlloc=0 AllocMem=0 AllocTRES= CfgTRES=cpu=20 State=IDLE+DRAIN'
migration_report0="PIREUS_HOST_FACTS node=spark-3c59 grant_mode=FENCED grant_valid=0 source_sha256=$source_sha freeze_sha256=$old_freeze_sha device_barrier=1 device_barrier_source_sha256=$barrier_sha inventory=1 protected=1"
migration_report1="PIREUS_HOST_FACTS node=spark-8e54 grant_mode=FENCED grant_valid=0 source_sha256=$source_sha freeze_sha256=$old_freeze_sha device_barrier=1 device_barrier_source_sha256=$barrier_sha inventory=1 protected=1"

bootstrap_freeze_migration_context "$migration_lease" "$migration_journal" \
  "$migration_slurm_nodes" '' '' "$migration_report0" "$migration_report1" 0 0 || {
    printf 'backend-transaction-unit: exact fail-closed migration context was rejected\n' >&2
    exit 1
  }
takeover_journal="$(jq '.data.step = "BOOTSTRAP_TAKEOVER"' <<<"$migration_journal")"
bootstrap_freeze_migration_context "$migration_lease" "$takeover_journal" \
  "$migration_slurm_nodes" '' '' "$migration_report0" "$migration_report1" 0 0 || {
    printf 'backend-transaction-unit: reentrant bootstrap takeover migration was rejected\n' >&2
    exit 1
  }
prior_freeze_sha="$(hex64 2)"
chained_takeover_journal="$(jq --arg prior "$prior_freeze_sha" \
  '.data.step = "BOOTSTRAP_TAKEOVER" | .data.migrationFromFreezeSha256 = $prior' \
  <<<"$migration_journal")"
prior_report1="${migration_report1/freeze_sha256=$old_freeze_sha/freeze_sha256=$prior_freeze_sha}"
bootstrap_freeze_migration_context "$migration_lease" "$chained_takeover_journal" \
  "$migration_slurm_nodes" '' '' "$migration_report0" "$prior_report1" 0 0 || {
    printf 'backend-transaction-unit: authenticated partial host binding was rejected\n' >&2
    exit 1
  }
foreign_report1="${migration_report1/freeze_sha256=$old_freeze_sha/freeze_sha256=$prepare0}"
if bootstrap_freeze_migration_context "$migration_lease" "$chained_takeover_journal" \
  "$migration_slurm_nodes" '' '' "$migration_report0" "$foreign_report1" 0 0; then
  printf 'backend-transaction-unit: unauthenticated partial host binding was accepted\n' >&2
  exit 1
fi
ancestor_freeze_sha="$(hex64 3)"
ancestor_report1="${migration_report1/freeze_sha256=$old_freeze_sha/freeze_sha256=$ancestor_freeze_sha}"
bootstrap_freeze_migration_context "$migration_lease" "$chained_takeover_journal" \
  "$migration_slurm_nodes" '' '' "$migration_report0" "$ancestor_report1" 0 0 || {
    printf 'backend-transaction-unit: frozen bootstrap ancestor was rejected\n' >&2
    exit 1
  }

assert_migration_context_denied() {
  local name="$1" lease="$2" journal="$3" slurm_nodes="$4" report0="$5"
  if bootstrap_freeze_migration_context "$lease" "$journal" "$slurm_nodes" '' '' \
    "$report0" "$migration_report1" 0 0; then
    printf 'backend-transaction-unit: %s migration context was accepted\n' "$name" >&2
    exit 1
  fi
}

live_migration_lease="$(jq '
  .spec.renewTime = "2099-01-01T00:00:00.000000Z"
' <<<"$migration_lease")"
assert_migration_context_denied live-lease "$live_migration_lease" \
  "$migration_journal" "$migration_slurm_nodes" "$migration_report0"
resumed_slurm_nodes="${migration_slurm_nodes//IDLE+DRAIN/IDLE}"
assert_migration_context_denied resumed-slurm "$migration_lease" \
  "$migration_journal" "$resumed_slurm_nodes" "$migration_report0"
unfenced_report0="${migration_report0/grant_mode=FENCED/grant_mode=K8S}"
assert_migration_context_denied unfenced-host "$migration_lease" \
  "$migration_journal" "$migration_slurm_nodes" "$unfenced_report0"
mismatched_journal="$(jq --arg freeze "$decision_sha" \
  '.data.semanticsFreezeSha256 = $freeze' <<<"$migration_journal")"
assert_migration_context_denied mismatched-journal "$migration_lease" \
  "$mismatched_journal" "$migration_slurm_nodes" "$migration_report0"

lease_json() {
  printf '{"metadata":{"uid":"%s","resourceVersion":"101"}}\n' "$lease_uid"
}
bind_host_pair_intent() {
  printf 'intent\n' >> "$TRACE"
  printf '102\n'
}
host_state_path() { printf '%s/%s.state\n' "$WORK_DIR" "$1"; }
write_host_state() {
  printf 'grant_mode=%s\ngrant_valid=%s\n' "$2" "$3" > "$(host_state_path "$1")"
}
host_fence_report() {
  local node="$1" state mode valid
  state="$(host_state_path "$node")"
  mode="$(sed -n 's/^grant_mode=//p' "$state")"
  valid="$(sed -n 's/^grant_valid=//p' "$state")"
  printf 'PIREUS_HOST_FACTS node=%s grant_mode=%s grant_valid=%s\n' \
    "$node" "$mode" "$valid"
}
bind_host_commit_receipts() {
  printf 'final-cas\n' >> "$TRACE"
  [[ "$SCENARIO" != cas-conflict ]]
}
host_fence_exec() {
  local node="$1" command="$2"
  printf '%s:%s\n' "$command" "$node" >> "$TRACE"
  case "$command:$node" in
    prepare:spark-3c59)
      printf 'PIREUS_HOST_PREPARED node=spark-3c59 prepare_receipt_sha256=%s\n' "$prepare0"
      ;;
    prepare:spark-8e54)
      printf 'PIREUS_HOST_PREPARED node=spark-8e54 prepare_receipt_sha256=%s\n' "$prepare1"
      ;;
    commit:spark-3c59)
      write_host_state spark-3c59 K8S 1
      [[ "$SCENARIO" != kill-after-commit-1 ]] || kill -KILL "$BASHPID"
      ;;
    commit:spark-8e54)
      write_host_state spark-8e54 K8S 1
      [[ "$SCENARIO" != kill-after-commit-2 ]] || kill -KILL "$BASHPID"
      ;;
    fence:spark-3c59|fence:spark-8e54)
      write_host_state "$node" FENCED 0
      ;;
  esac
}

run_scenario() {
  SCENARIO="$1"
  TRACE="$WORK_DIR/$SCENARIO.trace"
  export SCENARIO TRACE
  : > "$TRACE"
  write_host_state spark-3c59 FENCED 0
  write_host_state spark-8e54 FENCED 0
  set +e
  (grant_host_pair K8S --holder holder --epoch 7 --receipt "$receipt") \
    >/dev/null 2>&1
  status=$?
  set -e
  [[ $status -ne 0 ]] || {
    printf 'backend-transaction-unit: %s unexpectedly succeeded\n' "$SCENARIO" >&2
    exit 1
  }
  intent_line="$(grep -n '^intent$' "$TRACE" | cut -d: -f1)"
  commit0_line="$(grep -n '^commit:spark-3c59$' "$TRACE" | cut -d: -f1)"
  [[ -n "$intent_line" && -n "$commit0_line" && $intent_line -lt $commit0_line ]] || {
    printf 'backend-transaction-unit: %s authorized before durable intent\n' "$SCENARIO" >&2
    exit 1
  }
  case "$SCENARIO" in
    kill-after-commit-1)
      ! grep -Fq 'commit:spark-8e54' "$TRACE"
      ! grep -Fq 'final-cas' "$TRACE"
      [[ "$(sed -n 's/^grant_mode=//p' "$(host_state_path spark-3c59)")" == K8S ]]
      [[ "$(sed -n 's/^grant_mode=//p' "$(host_state_path spark-8e54)")" == FENCED ]]
      refence_host_pair_transaction holder 7 "$source_sha" "$freeze_sha" \
        "$(hex64 c)" "$lease_uid" 102 "$decision_sha"
      ;;
    kill-after-commit-2)
      grep -Fq 'commit:spark-8e54' "$TRACE"
      ! grep -Fq 'final-cas' "$TRACE"
      [[ "$(sed -n 's/^grant_mode=//p' "$(host_state_path spark-3c59)")" == K8S ]]
      [[ "$(sed -n 's/^grant_mode=//p' "$(host_state_path spark-8e54)")" == K8S ]]
      refence_host_pair_transaction holder 7 "$source_sha" "$freeze_sha" \
        "$(hex64 c)" "$lease_uid" 102 "$decision_sha"
      ;;
    cas-conflict)
      grep -Fq 'final-cas' "$TRACE"
      grep -Fq 'fence:spark-3c59' "$TRACE"
      grep -Fq 'fence:spark-8e54' "$TRACE"
      ;;
  esac
  [[ "$(host_fence_report spark-3c59)" == *'grant_mode=FENCED grant_valid=0' ]]
  [[ "$(host_fence_report spark-8e54)" == *'grant_mode=FENCED grant_valid=0' ]]
}

run_scenario kill-after-commit-1
run_scenario kill-after-commit-2
run_scenario cas-conflict

printf 'K8S_BACKEND_TRANSACTION_UNIT_PASS kill_after_commit_1=REFENCED kill_after_commit_2=REFENCED cas_conflict=REFENCED persisted_grants=PROVEN\n'
