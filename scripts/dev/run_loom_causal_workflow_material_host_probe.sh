#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_causal_workflow_material_capsule.sh"
HOST_SELFTEST="$ROOT_DIR/scripts/ci/sounio_loom_causal_workflow_material_host_selftest.sh"

fail() {
  printf 'run-loom-causal-workflow-material-host-probe: REFUSE reason=%s material_execution=false\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s [--capsule ABSOLUTE_PATH] [--expected-manifest-sha256 HEX] [--namespace NAME] [--node NAME] [--image IMAGE] [--run-id ID] [--receipt-output PATH]\n' "$0" >&2
  exit 64
}

sha256_file() { sha256sum "$1" | cut -d ' ' -f 1; }

record_value() {
  local record="$1" key="$2" line value='' count=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == "$key="* ]] || continue
    count=$((count + 1))
    value="${line#*=}"
  done < "$record"
  [[ "$count" == 1 ]] || fail "record field count is invalid: $key=$count"
  printf '%s\n' "$value"
}

CAPSULE=''
EXPECTED_MANIFEST_SHA256=''
NAMESPACE=beagle
NODE=t560-proxmox
IMAGE='192.168.3.207:5003/sounio-lab-beagle-workspace-ssh-20260425-zellij1:stable'
RUN_ID="cm$(date -u +%s)-$$"
RECEIPT_OUTPUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --capsule) CAPSULE="${2:-}"; shift 2 ;;
    --expected-manifest-sha256) EXPECTED_MANIFEST_SHA256="${2:-}"; shift 2 ;;
    --namespace) NAMESPACE="${2:-}"; shift 2 ;;
    --node) NODE="${2:-}"; shift 2 ;;
    --image) IMAGE="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --receipt-output) RECEIPT_OUTPUT="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "$NAMESPACE" =~ ^[a-z0-9.-]+$ && "$NODE" =~ ^[A-Za-z0-9._-]+$ && \
   "$IMAGE" =~ ^[A-Za-z0-9._:/-]+$ && "$RUN_ID" =~ ^[a-z0-9-]{8,48}$ ]] || usage
for input in "$BUILDER" "$HOST_SELFTEST"; do
  [[ -f "$input" && ! -L "$input" && -x "$input" ]] || fail "required probe input is absent, linked, or non-executable: $input"
done
for tool in kubectl sha256sum mktemp timeout tar install; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-causal-workflow-material-host-probe.XXXXXX")"
POD_A="loom-causal-material-a-${RUN_ID}"
POD_B="loom-causal-material-b-${RUN_ID}"
CLEANUP_POD="loom-causal-material-clean-${RUN_ID}"
ACTIVE_POD=''
HOST_ROOT=''

pod_exists() {
  kubectl -n "$NAMESPACE" get pod "$1" >/dev/null 2>&1
}

create_transport_pod() {
  local pod="$1" boundary
  kubectl -n "$NAMESPACE" create -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $pod
  labels:
    app.kubernetes.io/name: sounio-loom-causal-material-transport
    sounio.dev/loom-causal-material-run: $RUN_ID
spec:
  nodeName: $NODE
  hostPID: true
  restartPolicy: Never
  terminationGracePeriodSeconds: 0
  containers:
  - name: transport
    image: $IMAGE
    imagePullPolicy: IfNotPresent
    command: ["/bin/sh", "-c", "sleep 86400"]
    securityContext:
      privileged: true
      allowPrivilegeEscalation: true
      runAsUser: 0
EOF
  kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$pod" --timeout=180s >/dev/null ||
    fail "transport Pod did not become ready: $pod"
  boundary="$(kubectl -n "$NAMESPACE" get pod "$pod" -o jsonpath='{.spec.nodeName}|{.spec.hostPID}|{.spec.containers[0].securityContext.privileged}|{.status.phase}')"
  [[ "$boundary" == "$NODE|true|true|Running" ]] || fail "transport Pod boundary drifted: pod=$pod boundary=$boundary"
  kubectl -n "$NAMESPACE" exec "$pod" -- sh -lc 'command -v nsenter >/dev/null && test "$(id -u)" = 0 && test "$(nsenter -t 1 -m -u -i -n -p -- tr -d "\n" </proc/1/comm)" = systemd' ||
    fail "transport Pod cannot enter host PID1 systemd namespace: $pod"
}

host_exec() {
  local pod="$1"
  shift
  kubectl -n "$NAMESPACE" exec "$pod" -- nsenter -t 1 -m -u -i -n -p -- "$@"
}

host_failure_context() {
  local pod="$1"
  printf 'unit_show={'
  host_exec "$pod" systemctl show "$UNIT" -p Id -p LoadState -p ActiveState -p SubState \
    -p Result -p ExecMainCode -p ExecMainStatus -p MainPID -p InvocationID \
    -p ExecMainStartTimestampMonotonic 2>&1 || true
  printf '} result_log={'
  host_exec "$pod" cat "$HOST_ROOT/result.log" 2>&1 || true
  printf '} journal={'
  host_exec "$pod" journalctl -u "$UNIT" -n 80 --no-pager 2>&1 || true
  printf '} internal_cells={'
  host_exec "$pod" journalctl --since=-10min \
    '--grep=sounio-loom-(operation|causal-material)-cell' -n 160 --no-pager 2>&1 || true
  printf '}'
}

cleanup() {
  local cleanup_transport=''
  if pod_exists "$POD_B"; then
    cleanup_transport="$POD_B"
  elif pod_exists "$POD_A"; then
    cleanup_transport="$POD_A"
  elif [[ -n "$HOST_ROOT" ]]; then
    kubectl -n "$NAMESPACE" create -f - >/dev/null 2>&1 <<EOF || true
apiVersion: v1
kind: Pod
metadata:
  name: $CLEANUP_POD
spec:
  nodeName: $NODE
  hostPID: true
  restartPolicy: Never
  terminationGracePeriodSeconds: 0
  containers:
  - name: transport
    image: $IMAGE
    imagePullPolicy: IfNotPresent
    command: ["/bin/sh", "-c", "sleep 300"]
    securityContext:
      privileged: true
      allowPrivilegeEscalation: true
      runAsUser: 0
EOF
    kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$CLEANUP_POD" --timeout=60s >/dev/null 2>&1 || true
    pod_exists "$CLEANUP_POD" && cleanup_transport="$CLEANUP_POD"
  fi
  if [[ -n "$cleanup_transport" && -n "$HOST_ROOT" ]]; then
    host_exec "$cleanup_transport" systemctl stop "sounio-loom-causal-material-${RUN_ID}.service" >/dev/null 2>&1 || true
    host_exec "$cleanup_transport" systemctl reset-failed "sounio-loom-causal-material-${RUN_ID}.service" >/dev/null 2>&1 || true
    host_exec "$cleanup_transport" rm -rf "$HOST_ROOT" >/dev/null 2>&1 || true
  fi
  pod_exists "$POD_A" && kubectl -n "$NAMESPACE" delete pod "$POD_A" --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
  pod_exists "$POD_B" && kubectl -n "$NAMESPACE" delete pod "$POD_B" --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
  pod_exists "$CLEANUP_POD" && kubectl -n "$NAMESPACE" delete pod "$CLEANUP_POD" --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

if [[ -z "$CAPSULE" ]]; then
  CAPSULE="$WORK/capsule"
  build_output="$($BUILDER --output "$CAPSULE")"
  [[ "$build_output" == 'LOOM_CAUSAL_WORKFLOW_MATERIAL_CAPSULE_BUILD PASS '* ]] || fail "capsule build failed: $build_output"
else
  [[ "$CAPSULE" == /* && -d "$CAPSULE" && ! -L "$CAPSULE" ]] || fail 'provided capsule is absent, linked, or non-absolute'
fi
MANIFEST="$CAPSULE/capsule.manifest.v1"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'capsule manifest is absent or linked'
ACTUAL_MANIFEST_SHA256="$(sha256_file "$MANIFEST")"
[[ -z "$EXPECTED_MANIFEST_SHA256" ]] && EXPECTED_MANIFEST_SHA256="$ACTUAL_MANIFEST_SHA256"
[[ "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ && "$ACTUAL_MANIFEST_SHA256" == "$EXPECTED_MANIFEST_SHA256" ]] ||
  fail 'expected manifest hash differs from local capsule'
HOST_SELFTEST_SHA256="$(sha256_file "$HOST_SELFTEST")"
create_transport_pod "$POD_A"
ACTIVE_POD="$POD_A"
POD_A_UID="$(kubectl -n "$NAMESPACE" get pod "$POD_A" -o jsonpath='{.metadata.uid}')"
[[ "$POD_A_UID" =~ ^[0-9a-f-]{36}$ ]] || fail 'Pod-A UID is non-canonical'

HOST_ROOT="/var/tmp/sounio-loom-causal-workflow-material-${RUN_ID}"
kubectl -n "$NAMESPACE" exec "$POD_A" -- sh -c "umask 077; rm -rf '/proc/1/root$HOST_ROOT'; mkdir -p '/proc/1/root$HOST_ROOT/capsule'; chmod 0711 '/proc/1/root$HOST_ROOT'; chmod 0700 '/proc/1/root$HOST_ROOT/capsule'"
tar -C "$CAPSULE" -cf - . | kubectl -n "$NAMESPACE" exec -i "$POD_A" -- sh -c "tar -xf - -C '/proc/1/root$HOST_ROOT/capsule'; chown -R 0:0 '/proc/1/root$HOST_ROOT/capsule'; chmod 0555 '/proc/1/root$HOST_ROOT/capsule'"
kubectl -n "$NAMESPACE" exec -i "$POD_A" -- sh -c "cat > '/proc/1/root$HOST_ROOT/host-selftest.sh'; chmod 0500 '/proc/1/root$HOST_ROOT/host-selftest.sh'" < "$HOST_SELFTEST"
remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD_A" -- sh -c "sha256sum '/proc/1/root$HOST_ROOT/capsule/capsule.manifest.v1' '/proc/1/root$HOST_ROOT/host-selftest.sh' | cut -d ' ' -f 1")"
mapfile -t hash_lines <<< "$remote_hashes"
[[ "${hash_lines[0]:-}" == "$EXPECTED_MANIFEST_SHA256" && "${hash_lines[1]:-}" == "$HOST_SELFTEST_SHA256" ]] ||
  fail 'transport hash drifted before host namespace entry'

UNIT="sounio-loom-causal-material-${RUN_ID}.service"
host_exec "$POD_A" systemd-run --quiet --no-block --unit="${UNIT%.service}" \
  --property=Type=oneshot --property=RemainAfterExit=yes --property=TimeoutStartSec=480 \
  --property="StandardOutput=append:$HOST_ROOT/result.log" \
  --property="StandardError=append:$HOST_ROOT/result.log" \
  "$HOST_ROOT/host-selftest.sh" --capsule "$HOST_ROOT/capsule" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" --store-root "$HOST_ROOT/store" \
  --phase-marker "$HOST_ROOT/started.v1" ||
  fail "host systemd refused transient unit context=$(host_failure_context "$POD_A")"
for _ in $(seq 1 240); do
  [[ "$(host_exec "$POD_A" cat "$HOST_ROOT/started.v1" 2>/dev/null || true)" == MATERIAL_HOST_UNIT_STARTED ]] && break
  sleep 0.25
done
[[ "$(host_exec "$POD_A" cat "$HOST_ROOT/started.v1" 2>/dev/null || true)" == MATERIAL_HOST_UNIT_STARTED ]] ||
  fail "host-owned transient unit never reached durable started marker context=$(host_failure_context "$POD_A")"
READY_RECORD="$HOST_ROOT/store/pod-loss-ready.record"
for _ in $(seq 1 720); do
  ready_probe="$(host_exec "$POD_A" cat "$READY_RECORD" 2>/dev/null || true)"
  [[ "$ready_probe" == loom-causal-pod-loss-ready-v1$'\n'* ]] && break
  ready_unit_state="$(host_exec "$POD_A" systemctl show "$UNIT" -p ActiveState --value 2>/dev/null || true)"
  [[ "$ready_unit_state" == failed || "$ready_unit_state" == inactive ]] && break
  sleep 0.25
done
[[ "${ready_probe:-}" == loom-causal-pod-loss-ready-v1$'\n'* ]] ||
  fail "host workflow never reached action9037 RUNNING pod-loss synchronization point context=$(host_failure_context "$POD_A")"
READY_SHA256="$(printf '%s\n' "$ready_probe" | sha256sum | cut -d ' ' -f 1)"
READY_GUARDIAN_GENERATION="$(record_value <(printf '%s\n' "$ready_probe") guardian_generation)"
READY_WORKFLOW_ID="$(record_value <(printf '%s\n' "$ready_probe") workflow_id)"
[[ "$READY_GUARDIAN_GENERATION" =~ ^[0-9a-f]{64}$ && "$READY_WORKFLOW_ID" =~ ^[A-Za-z0-9._-]{1,256}$ ]] ||
  fail 'pod-loss readiness record identity is non-canonical'
unit_before_record="$(host_exec "$POD_A" systemctl show "$UNIT" -p Id -p ActiveState -p MainPID -p InvocationID -p ExecMainStartTimestampMonotonic)"
unit_before_id="$(record_value <(printf '%s\n' "$unit_before_record") Id)"
unit_before_state="$(record_value <(printf '%s\n' "$unit_before_record") ActiveState)"
unit_before_pid="$(record_value <(printf '%s\n' "$unit_before_record") MainPID)"
unit_invocation_id="$(record_value <(printf '%s\n' "$unit_before_record") InvocationID)"
unit_exec_start="$(record_value <(printf '%s\n' "$unit_before_record") ExecMainStartTimestampMonotonic)"
[[ "$unit_before_id" == "$UNIT" && "$unit_before_state" == activating &&
   "$unit_before_pid" =~ ^[1-9][0-9]*$ && "$unit_invocation_id" =~ ^[0-9a-f]{32}$ &&
   "$unit_exec_start" =~ ^[1-9][0-9]*$ ]] ||
  fail "host workflow identity was not live at transport-loss point: $unit_before_record"
kubectl -n "$NAMESPACE" delete pod "$POD_A" --wait=true --timeout=120s >/dev/null || fail 'transport Pod-A deletion failed'
[[ "$(kubectl -n "$NAMESPACE" get pod "$POD_A" --ignore-not-found -o name)" == '' ]] || fail 'transport Pod-A remained after deletion'
create_transport_pod "$POD_B"
ACTIVE_POD="$POD_B"
POD_B_UID="$(kubectl -n "$NAMESPACE" get pod "$POD_B" -o jsonpath='{.metadata.uid}')"
[[ "$POD_B_UID" =~ ^[0-9a-f-]{36}$ && "$POD_B_UID" != "$POD_A_UID" ]] || fail 'Pod-B identity does not prove transport replacement'
release_stage="$(mktemp "$WORK/pod-loss-release.XXXXXX")"
cat > "$release_stage" <<EOF
loom-causal-pod-loss-release-v1
guardian_generation=$READY_GUARDIAN_GENERATION
workflow_id=$READY_WORKFLOW_ID
ready_sha256=$READY_SHA256
pod_a_deleted=true
pod_a_uid=$POD_A_UID
pod_b_uid=$POD_B_UID
EOF
kubectl -n "$NAMESPACE" exec -i "$POD_B" -- sh -c "umask 077; cat > '/proc/1/root$HOST_ROOT/pod-loss-release.stage'; chown 0:0 '/proc/1/root$HOST_ROOT/pod-loss-release.stage'; chmod 0400 '/proc/1/root$HOST_ROOT/pod-loss-release.stage'; mv '/proc/1/root$HOST_ROOT/pod-loss-release.stage' '/proc/1/root$HOST_ROOT/store/pod-loss-release.record'" < "$release_stage"
rm -f "$release_stage"
release_metadata="$(host_exec "$POD_B" stat -c '%u:%g:%a' "$HOST_ROOT/store/pod-loss-release.record")"
[[ "$release_metadata" == 0:0:400 ]] || fail "pod-loss release metadata drifted: $release_metadata"
for _ in $(seq 1 240); do
  unit_after_record="$(host_exec "$POD_B" systemctl show "$UNIT" -p Id -p ActiveState -p MainPID -p InvocationID -p ExecMainStartTimestampMonotonic 2>/dev/null || true)"
  unit_after_state="$(record_value <(printf '%s\n' "$unit_after_record") ActiveState 2>/dev/null || true)"
  [[ "$unit_after_state" == active || "$unit_after_state" == failed ]] && break
  sleep 0.25
done
unit_after_id="$(record_value <(printf '%s\n' "${unit_after_record:-}") Id)"
unit_after_invocation_id="$(record_value <(printf '%s\n' "${unit_after_record:-}") InvocationID)"
unit_after_exec_start="$(record_value <(printf '%s\n' "${unit_after_record:-}") ExecMainStartTimestampMonotonic)"
[[ "$unit_after_id" == "$UNIT" && "$unit_after_state" == active &&
   "$unit_after_invocation_id" == "$unit_invocation_id" && "$unit_after_exec_start" == "$unit_exec_start" ]] ||
  fail "same host-owned unit instance did not survive replacement transport: ${unit_after_record:-absent}"
host_output="$(host_exec "$POD_B" cat "$HOST_ROOT/result.log")"
[[ "$host_output" == *'sounio-loom-causal-workflow-material-host-selftest: HOST_MEASUREMENT_PASS '* && "$host_output" == *'LOOM_CAUSAL_WORKFLOW_MATERIAL_HOST PASS '* ]] ||
  fail "host selftest receipt was absent after replacement transport: $host_output"
[[ "$host_output" == *' compile_count=1 ticket_count=1 launch_count=1 '* ]] ||
  fail 'host receipt did not retain exact-once workflow counts'
transport_receipt="LOOM_CAUSAL_WORKFLOW_MATERIAL_HOST_TRANSPORT PASS namespace=$NAMESPACE node=$NODE pod_a=$POD_A pod_a_uid=$POD_A_UID pod_b=$POD_B pod_b_uid=$POD_B_UID transport_pod_deleted=true replacement_transport=true hostguardian_unit=$UNIT hostguardian_invocation_id=$unit_invocation_id hostguardian_exec_start_monotonic=$unit_exec_start hostguardian_unit_survived=true manifest_sha256=$EXPECTED_MANIFEST_SHA256 host_selftest_sha256=$HOST_SELFTEST_SHA256 transport=kubectl+privileged-hostPID+nsenter+systemd capsule_layout=unpacked-directory-v1 shared_checkout=false material_execution=true compile_count=1 ticket_count=1 launch_count=1 pod_loss_synchronized=true transport_pod_loss_measured=true pod_loss_measured=false material_cell_survival_measured=false pod_loss_boundary=MATERIAL_RUNNING_PRE_EXEC production_activation=false parity_open=false claim_ready=false host_output_sha256=$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)"
if [[ -n "$RECEIPT_OUTPUT" ]]; then
  stage="$(mktemp "$(dirname "$RECEIPT_OUTPUT")/.loom-causal-host-receipt.XXXXXX")"
  printf '%s\n%s\n' "$transport_receipt" "$host_output" > "$stage"
  install -m 0644 "$stage" "$RECEIPT_OUTPUT"
  rm -f "$stage"
fi
printf '%s\n%s\n' "$host_output" "$transport_receipt"
