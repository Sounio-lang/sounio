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
[[ -f "${BASH_SOURCE[0]}" && ! -L "${BASH_SOURCE[0]}" ]] || fail 'probe source is absent or linked'
if [[ -z "$CAPSULE" ]]; then
  for input in "$BUILDER" "$HOST_SELFTEST"; do
    [[ -f "$input" && ! -L "$input" && -x "$input" ]] || fail "required capsule build input is absent, linked, or non-executable: $input"
  done
fi
for tool in kubectl sha256sum mktemp timeout tar install; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-causal-workflow-material-host-probe.XXXXXX")"
POD_A="loom-causal-material-a-${RUN_ID}"
POD_B="loom-causal-material-b-${RUN_ID}"
CLEANUP_POD="loom-causal-material-clean-${RUN_ID}"
ACTIVE_POD=''
HOST_ROOT=''
MATERIAL_UNIT=''
NONCE_SCHEMA_VERSION=v1
NONCE_SECRET_SCHEMA="loom-causal-material-barrier-nonce-secret-${NONCE_SCHEMA_VERSION}"

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
  local prior_status=$? cleanup_transport='' outer_state='' material_state=''
  trap - EXIT
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
    if [[ -n "$MATERIAL_UNIT" ]]; then
      host_exec "$cleanup_transport" systemctl stop "$MATERIAL_UNIT" >/dev/null 2>&1 || true
      host_exec "$cleanup_transport" systemctl reset-failed "$MATERIAL_UNIT" >/dev/null 2>&1 || true
    fi
    host_exec "$cleanup_transport" systemctl stop "sounio-loom-causal-material-${RUN_ID}.service" >/dev/null 2>&1 || true
    host_exec "$cleanup_transport" systemctl reset-failed "sounio-loom-causal-material-${RUN_ID}.service" >/dev/null 2>&1 || true
    outer_state="$(host_exec "$cleanup_transport" systemctl is-active "sounio-loom-causal-material-${RUN_ID}.service" 2>/dev/null || true)"
    if [[ -n "$MATERIAL_UNIT" ]]; then
      material_state="$(host_exec "$cleanup_transport" systemctl is-active "$MATERIAL_UNIT" 2>/dev/null || true)"
    fi
    if [[ "$outer_state" != active && "$outer_state" != activating && \
          "$material_state" != active && "$material_state" != activating ]]; then
      host_exec "$cleanup_transport" rm -rf "$HOST_ROOT" >/dev/null 2>&1 || prior_status=70
    else
      printf 'run-loom-causal-workflow-material-host-probe: REFUSE reason=cleanup-could-not-prove-unit-extinction outer_state=%s material_state=%s preserved_host_root=%s\n' \
        "${outer_state:-absent}" "${material_state:-absent}" "$HOST_ROOT" >&2
      prior_status=70
    fi
  elif [[ -n "$HOST_ROOT" ]]; then
    printf 'run-loom-causal-workflow-material-host-probe: REFUSE reason=cleanup-transport-unavailable preserved_host_root=%s\n' \
      "$HOST_ROOT" >&2
    prior_status=70
  fi
  pod_exists "$POD_A" && kubectl -n "$NAMESPACE" delete pod "$POD_A" --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
  pod_exists "$POD_B" && kubectl -n "$NAMESPACE" delete pod "$POD_B" --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
  pod_exists "$CLEANUP_POD" && kubectl -n "$NAMESPACE" delete pod "$CLEANUP_POD" --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
  chmod -R u+rwX "$WORK" 2>/dev/null || true
  rm -rf "$WORK"
  exit "$prior_status"
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
[[ "$(record_value "$MANIFEST" barrier_hold_timeout_milliseconds)" == 600000 && \
   "$(record_value "$MANIFEST" guardian_request_timeout_milliseconds)" == 570000 && \
   "$(record_value "$MANIFEST" broker_timeout_milliseconds)" == 650000 && \
   "$(record_value "$MANIFEST" outer_unit_timeout_milliseconds)" == 720000 ]] ||
  fail 'capsule timing envelope drifted before transport'
PACKAGED_SELFTEST="$CAPSULE/$(record_value "$MANIFEST" host_selftest_path)"
PACKAGED_PROBE="$CAPSULE/$(record_value "$MANIFEST" host_probe_path)"
[[ -x "$PACKAGED_SELFTEST" && ! -L "$PACKAGED_SELFTEST" && -x "$PACKAGED_PROBE" && ! -L "$PACKAGED_PROBE" ]] ||
  fail 'packaged host gates are absent, linked, or non-executable'
HOST_SELFTEST_SHA256="$(sha256_file "$PACKAGED_SELFTEST")"
HOST_PROBE_SHA256="$(sha256_file "$PACKAGED_PROBE")"
[[ "$HOST_SELFTEST_SHA256" == "$(record_value "$MANIFEST" host_selftest_sha256)" && \
   "$HOST_PROBE_SHA256" == "$(record_value "$MANIFEST" host_probe_sha256)" && \
   "$(sha256_file "${BASH_SOURCE[0]}")" == "$HOST_PROBE_SHA256" ]] ||
  fail 'packaged gate or invoking probe hash drifted before transport'
create_transport_pod "$POD_A"
ACTIVE_POD="$POD_A"
POD_A_UID="$(kubectl -n "$NAMESPACE" get pod "$POD_A" -o jsonpath='{.metadata.uid}')"
[[ "$POD_A_UID" =~ ^[0-9a-f-]{36}$ ]] || fail 'Pod-A UID is non-canonical'

HOST_ROOT="/var/lib/sounio-loom-causal-workflow-material-${RUN_ID}"
kubectl -n "$NAMESPACE" exec "$POD_A" -- sh -c "umask 077; test ! -e '/proc/1/root$HOST_ROOT'; mkdir -p '/proc/1/root$HOST_ROOT/capsule'; chmod 0711 '/proc/1/root$HOST_ROOT'; chmod 0700 '/proc/1/root$HOST_ROOT/capsule'" ||
  fail 'host probe root already exists or cannot be created safely'
tar -C "$CAPSULE" -cf - . | kubectl -n "$NAMESPACE" exec -i "$POD_A" -- sh -c "tar -xf - -C '/proc/1/root$HOST_ROOT/capsule'; chown -R 0:0 '/proc/1/root$HOST_ROOT/capsule'; chmod 0555 '/proc/1/root$HOST_ROOT/capsule'"
remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD_A" -- sh -c "sha256sum '/proc/1/root$HOST_ROOT/capsule/capsule.manifest.v1' '/proc/1/root$HOST_ROOT/capsule/${PACKAGED_SELFTEST#"$CAPSULE/"}' '/proc/1/root$HOST_ROOT/capsule/${PACKAGED_PROBE#"$CAPSULE/"}' | cut -d ' ' -f 1")"
mapfile -t hash_lines <<< "$remote_hashes"
[[ "${hash_lines[0]:-}" == "$EXPECTED_MANIFEST_SHA256" && \
   "${hash_lines[1]:-}" == "$HOST_SELFTEST_SHA256" && \
   "${hash_lines[2]:-}" == "$HOST_PROBE_SHA256" ]] ||
  fail 'transport hash drifted before host namespace entry'

UNIT="sounio-loom-causal-material-${RUN_ID}.service"
host_exec "$POD_A" systemd-run --quiet --no-block --unit="${UNIT%.service}" \
  --property=Type=oneshot --property=RemainAfterExit=yes --property=TimeoutStartSec=720 \
  --property="StandardOutput=append:$HOST_ROOT/result.log" \
  --property="StandardError=append:$HOST_ROOT/result.log" \
  "$HOST_ROOT/capsule/${PACKAGED_SELFTEST#"$CAPSULE/"}" --capsule "$HOST_ROOT/capsule" \
  --expected-manifest-sha256 "$EXPECTED_MANIFEST_SHA256" --store-root "$HOST_ROOT/store" \
  --phase-marker "$HOST_ROOT/started.v1" ||
  fail "host systemd refused transient unit context=$(host_failure_context "$POD_A")"
for _ in $(seq 1 240); do
  [[ "$(host_exec "$POD_A" cat "$HOST_ROOT/started.v1" 2>/dev/null || true)" == MATERIAL_HOST_UNIT_STARTED ]] && break
  sleep 0.25
done
[[ "$(host_exec "$POD_A" cat "$HOST_ROOT/started.v1" 2>/dev/null || true)" == MATERIAL_HOST_UNIT_STARTED ]] ||
  fail "host-owned transient unit never reached durable started marker context=$(host_failure_context "$POD_A")"
READY_RECORD="$HOST_ROOT/store/mid-exec-ready.record"
for _ in $(seq 1 720); do
  ready_probe="$(host_exec "$POD_A" cat "$READY_RECORD" 2>/dev/null || true)"
  [[ "$ready_probe" == loom-causal-material-mid-exec-ready-v1$'\n'* ]] && break
  ready_unit_state="$(host_exec "$POD_A" systemctl show "$UNIT" -p ActiveState --value 2>/dev/null || true)"
  [[ "$ready_unit_state" == failed || "$ready_unit_state" == inactive ]] && break
  sleep 0.25
done
[[ "${ready_probe:-}" == loom-causal-material-mid-exec-ready-v1$'\n'* ]] ||
  fail "host workflow never reached authenticated MATERIAL_RUNNING_IN_EXEC context=$(host_failure_context "$POD_A")"
host_exec "$POD_A" sh -c "test ! -e '$HOST_ROOT/store/barrier-nonce.secret' && ! grep -aEq '^barrier_nonce=' '$READY_RECORD'" ||
  fail 'raw barrier nonce escaped HostGuardian memory before Pod-A deletion'
READY_SHA256="$(printf '%s\n' "$ready_probe" | sha256sum | cut -d ' ' -f 1)"
READY_GUARDIAN_GENERATION="$(record_value <(printf '%s\n' "$ready_probe") guardian_generation)"
READY_WORKFLOW_ID="$(record_value <(printf '%s\n' "$ready_probe") workflow_id)"
READY_RUN_GENERATION="$(record_value <(printf '%s\n' "$ready_probe") run_grant_generation)"
READY_NONCE_SHA256="$(record_value <(printf '%s\n' "$ready_probe") barrier_nonce_sha256)"
READY_WITNESS_SHA256="$(record_value <(printf '%s\n' "$ready_probe") running_witness_sha256)"
MATERIAL_UNIT="$(record_value <(printf '%s\n' "$ready_probe") material_unit)"
MATERIAL_INVOCATION_ID="$(record_value <(printf '%s\n' "$ready_probe") material_invocation_id)"
MATERIAL_PID="$(record_value <(printf '%s\n' "$ready_probe") material_pid)"
MATERIAL_START_TICK="$(record_value <(printf '%s\n' "$ready_probe") material_start_tick)"
MATERIAL_CGROUP_SHA256="$(record_value <(printf '%s\n' "$ready_probe") material_cgroup_sha256)"
[[ "$READY_GUARDIAN_GENERATION" =~ ^[0-9a-f]{64}$ && \
   "$READY_RUN_GENERATION" =~ ^[0-9a-f]{64}$ && "$READY_NONCE_SHA256" =~ ^[0-9a-f]{64}$ && \
   "$READY_WITNESS_SHA256" =~ ^[0-9a-f]{64}$ && "$READY_WORKFLOW_ID" =~ ^[A-Za-z0-9._-]{1,256}$ && \
   "$MATERIAL_UNIT" =~ ^[A-Za-z0-9._-]{1,256}$ && "$MATERIAL_INVOCATION_ID" =~ ^[0-9a-f]{32}$ && \
   "$MATERIAL_PID" =~ ^[1-9][0-9]*$ && "$MATERIAL_START_TICK" =~ ^[1-9][0-9]*$ && \
   "$MATERIAL_CGROUP_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'mid-exec readiness record identity is non-canonical'

material_before_record="$(host_exec "$POD_A" systemctl show "$MATERIAL_UNIT" -p Id -p ActiveState -p MainPID -p InvocationID -p ControlGroup)"
material_before_id="$(record_value <(printf '%s\n' "$material_before_record") Id)"
material_before_state="$(record_value <(printf '%s\n' "$material_before_record") ActiveState)"
material_cell_before_pid="$(record_value <(printf '%s\n' "$material_before_record") MainPID)"
material_before_invocation="$(record_value <(printf '%s\n' "$material_before_record") InvocationID)"
material_before_cgroup="$(record_value <(printf '%s\n' "$material_before_record") ControlGroup)"
material_cell_before_stat="$(host_exec "$POD_A" cat "/proc/$material_cell_before_pid/stat")"
material_cell_before_tail="${material_cell_before_stat##*) }"
read -r -a material_cell_before_fields <<< "$material_cell_before_tail"
material_cell_before_start="${material_cell_before_fields[19]:-}"
material_cell_before_cgroup="$(host_exec "$POD_A" cat "/proc/$material_cell_before_pid/cgroup")"
material_cell_before_cgroup_sha256="$(host_exec "$POD_A" sha256sum "/proc/$material_cell_before_pid/cgroup" | cut -d ' ' -f 1)"
material_before_stat="$(host_exec "$POD_A" cat "/proc/$MATERIAL_PID/stat")"
material_before_tail="${material_before_stat##*) }"
read -r -a material_before_fields <<< "$material_before_tail"
material_before_start="${material_before_fields[19]:-}"
material_before_cgroup_sha256="$(host_exec "$POD_A" sha256sum "/proc/$MATERIAL_PID/cgroup" | cut -d ' ' -f 1)"
[[ "$material_before_id" == "$MATERIAL_UNIT" && \
   ( "$material_before_state" == active || "$material_before_state" == activating ) && \
   "$material_cell_before_pid" =~ ^[1-9][0-9]*$ && "$material_cell_before_pid" -gt 1 && \
   "$material_cell_before_pid" != "$MATERIAL_PID" && "$material_cell_before_start" =~ ^[1-9][0-9]*$ && \
   "$material_cell_before_cgroup" == *"$MATERIAL_UNIT"* && \
   "$material_before_invocation" == "$MATERIAL_INVOCATION_ID" && \
   "$material_before_start" == "$MATERIAL_START_TICK" && "$material_before_cgroup" == */"$MATERIAL_UNIT" && \
   "$material_before_cgroup_sha256" == "$MATERIAL_CGROUP_SHA256" ]] ||
  fail "material controller/tracee identity was not live at authenticated EXEC_STARTED: $material_before_record"
host_exec "$POD_A" sh -c "test ! -e '$HOST_ROOT/store/result.record' && test ! -e '$HOST_ROOT/store/attestation.record'" ||
  fail 'material output existed before Pod-A deletion and authorized release'
unit_before_record="$(host_exec "$POD_A" systemctl show "$UNIT" -p Id -p MainPID -p InvocationID -p ExecMainStartTimestampMonotonic)"
unit_invocation_id="$(record_value <(printf '%s\n' "$unit_before_record") InvocationID)"
unit_exec_start="$(record_value <(printf '%s\n' "$unit_before_record") ExecMainStartTimestampMonotonic)"
[[ "$(record_value <(printf '%s\n' "$unit_before_record") Id)" == "$UNIT" && \
   "$unit_invocation_id" =~ ^[0-9a-f]{32}$ && "$unit_exec_start" =~ ^[1-9][0-9]*$ ]] ||
  fail "HostGuardian identity was not live before Pod-A deletion: $unit_before_record"
kubectl -n "$NAMESPACE" delete pod "$POD_A" --wait=true --timeout=120s >/dev/null || fail 'transport Pod-A deletion failed'
[[ "$(kubectl -n "$NAMESPACE" get pod "$POD_A" --ignore-not-found -o name)" == '' ]] || fail 'transport Pod-A remained after deletion'
create_transport_pod "$POD_B"
ACTIVE_POD="$POD_B"
POD_B_UID="$(kubectl -n "$NAMESPACE" get pod "$POD_B" -o jsonpath='{.metadata.uid}')"
[[ "$POD_B_UID" =~ ^[0-9a-f-]{36}$ && "$POD_B_UID" != "$POD_A_UID" ]] || fail 'Pod-B identity does not prove transport replacement'

material_after_record="$(host_exec "$POD_B" systemctl show "$MATERIAL_UNIT" -p Id -p ActiveState -p MainPID -p InvocationID -p ControlGroup)"
material_cell_after_pid="$(record_value <(printf '%s\n' "$material_after_record") MainPID)"
material_after_invocation="$(record_value <(printf '%s\n' "$material_after_record") InvocationID)"
material_cell_after_stat="$(host_exec "$POD_B" cat "/proc/$material_cell_after_pid/stat")"
material_cell_after_tail="${material_cell_after_stat##*) }"
read -r -a material_cell_after_fields <<< "$material_cell_after_tail"
material_cell_after_start="${material_cell_after_fields[19]:-}"
material_cell_after_cgroup_sha256="$(host_exec "$POD_B" sha256sum "/proc/$material_cell_after_pid/cgroup" | cut -d ' ' -f 1)"
material_after_stat="$(host_exec "$POD_B" cat "/proc/$MATERIAL_PID/stat")"
material_after_tail="${material_after_stat##*) }"
read -r -a material_after_fields <<< "$material_after_tail"
material_after_start="${material_after_fields[19]:-}"
material_after_cgroup_sha256="$(host_exec "$POD_B" sha256sum "/proc/$MATERIAL_PID/cgroup" | cut -d ' ' -f 1)"
[[ "$(record_value <(printf '%s\n' "$material_after_record") Id)" == "$MATERIAL_UNIT" && \
   ( "$(record_value <(printf '%s\n' "$material_after_record") ActiveState)" == active || \
     "$(record_value <(printf '%s\n' "$material_after_record") ActiveState)" == activating ) && \
   "$material_cell_after_pid" == "$material_cell_before_pid" && \
   "$material_cell_after_start" == "$material_cell_before_start" && \
   "$material_cell_after_cgroup_sha256" == "$material_cell_before_cgroup_sha256" && \
   "$material_after_invocation" == "$MATERIAL_INVOCATION_ID" && \
   "$material_after_start" == "$MATERIAL_START_TICK" && "$material_after_cgroup_sha256" == "$MATERIAL_CGROUP_SHA256" ]] ||
  fail "material controller/tracee identity did not survive Pod replacement: $material_after_record"

if [[ "${MATERIAL_INVOCATION_ID:0:1}" == 0 ]]; then
  SABOTAGE_INVOCATION_ID="1${MATERIAL_INVOCATION_ID:1}"
else
  SABOTAGE_INVOCATION_ID="0${MATERIAL_INVOCATION_ID:1}"
fi
sabotage_stage="$(mktemp "$WORK/mid-exec-sabotage.XXXXXX")"
cat > "$sabotage_stage" <<EOF
loom-causal-material-mid-exec-sabotage-v1
guardian_generation=$READY_GUARDIAN_GENERATION
workflow_id=$READY_WORKFLOW_ID
ready_sha256=$READY_SHA256
running_witness_sha256=$READY_WITNESS_SHA256
run_grant_generation=$READY_RUN_GENERATION
barrier_nonce_sha256=$READY_NONCE_SHA256
observed_material_unit=$MATERIAL_UNIT
observed_material_invocation_id=$SABOTAGE_INVOCATION_ID
observed_material_pid=$MATERIAL_PID
observed_material_start_tick=$MATERIAL_START_TICK
observed_material_cgroup_sha256=$MATERIAL_CGROUP_SHA256
expected_decision=DENY592
pod_a_deleted=true
pod_a_uid=$POD_A_UID
pod_b_uid=$POD_B_UID
EOF
kubectl -n "$NAMESPACE" exec -i "$POD_B" -- sh -c "umask 077; cat > '/proc/1/root$HOST_ROOT/mid-exec-sabotage.stage'; chown 0:0 '/proc/1/root$HOST_ROOT/mid-exec-sabotage.stage'; chmod 0400 '/proc/1/root$HOST_ROOT/mid-exec-sabotage.stage'; mv '/proc/1/root$HOST_ROOT/mid-exec-sabotage.stage' '/proc/1/root$HOST_ROOT/store/mid-exec-sabotage.record'" < "$sabotage_stage"
rm -f "$sabotage_stage"
for _ in $(seq 1 240); do
  sabotage_refusal="$(host_exec "$POD_B" cat "$HOST_ROOT/store/mid-exec-sabotage-refusal.record" 2>/dev/null || true)"
  [[ "$sabotage_refusal" == loom-causal-material-mid-exec-sabotage-refusal-v1$'\n'* ]] && break
  sleep 0.25
done
[[ "${sabotage_refusal:-}" == *$'\ndecision=DENY592\n'* && \
   "${sabotage_refusal:-}" == *$'\noriginal_cell_held=true\n'* && \
   "${sabotage_refusal:-}" == *$'\nrelease_sent=false\n'* ]] && \
  host_exec "$POD_B" sh -c "test ! -e '$HOST_ROOT/store/result.record' && test ! -e '$HOST_ROOT/store/attestation.record'" ||
  fail "replacement-identity sabotage was not refused before release: ${sabotage_refusal:-absent}"

release_stage="$(mktemp "$WORK/mid-exec-release.XXXXXX")"
cat > "$release_stage" <<EOF
loom-causal-material-mid-exec-release-request-v1
guardian_generation=$READY_GUARDIAN_GENERATION
workflow_id=$READY_WORKFLOW_ID
ready_sha256=$READY_SHA256
running_witness_sha256=$READY_WITNESS_SHA256
run_grant_generation=$READY_RUN_GENERATION
barrier_nonce_sha256=$READY_NONCE_SHA256
material_unit=$MATERIAL_UNIT
material_invocation_id=$MATERIAL_INVOCATION_ID
material_pid=$MATERIAL_PID
material_start_tick=$MATERIAL_START_TICK
material_cgroup_sha256=$MATERIAL_CGROUP_SHA256
pod_a_deleted=true
pod_a_uid=$POD_A_UID
pod_b_uid=$POD_B_UID
replacement_sabotage=DENY592
EOF
kubectl -n "$NAMESPACE" exec -i "$POD_B" -- sh -c "umask 077; cat > '/proc/1/root$HOST_ROOT/mid-exec-release.stage'; chown 0:0 '/proc/1/root$HOST_ROOT/mid-exec-release.stage'; chmod 0400 '/proc/1/root$HOST_ROOT/mid-exec-release.stage'; mv '/proc/1/root$HOST_ROOT/mid-exec-release.stage' '/proc/1/root$HOST_ROOT/store/mid-exec-release-request.record'" < "$release_stage"
rm -f "$release_stage"
release_metadata="$(host_exec "$POD_B" stat -c '%u:%g:%a' "$HOST_ROOT/store/mid-exec-release-request.record")"
[[ "$release_metadata" == 0:0:400 ]] || fail "mid-exec release request metadata drifted: $release_metadata"
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
  fail "same host-owned unit instance did not survive replacement transport: ${unit_after_record:-absent} context=$(host_failure_context "$POD_B")"
host_output="$(host_exec "$POD_B" cat "$HOST_ROOT/result.log")"
[[ "$host_output" == *'sounio-loom-causal-workflow-material-host-selftest: HOST_LOCAL_EXECUTION_PASS '* && "$host_output" == *'LOOM_CAUSAL_WORKFLOW_MATERIAL_HOST PASS '* ]] ||
  fail "host selftest receipt was absent after replacement transport: $host_output"
[[ "$host_output" == *' compile_count=1 ticket_count=1 launch_count=1 result_count=1 attestation_count=1 '* ]] ||
  fail 'host receipt did not retain exact-once workflow counts'
[[ "$host_output" == *' barrier_nonce_source=getrandom barrier_nonce_storage=HostGuardian-memory-only barrier_nonce_secret_persisted=false '* ]] ||
  fail 'host receipt did not prove memory-only kernel-CSPRNG nonce handling'
host_exec "$POD_B" sh -c "test ! -e '$HOST_ROOT/store/barrier-nonce.secret' && ! grep -aEq '^barrier_nonce=[0-9a-f]{64}$' '$HOST_ROOT/result.log' && test -z \"\$(find '$HOST_ROOT/capsule' -type f -exec grep -aFl '$NONCE_SECRET_SCHEMA' {} +)\"" ||
  fail 'raw barrier nonce appeared in capsule, store, or result output'
transport_receipt="LOOM_CAUSAL_WORKFLOW_MATERIAL_HOST_TRANSPORT PASS namespace=$NAMESPACE node=$NODE pod_a=$POD_A pod_a_uid=$POD_A_UID pod_b=$POD_B pod_b_uid=$POD_B_UID transport_pod_deleted=true replacement_transport=true hostguardian_unit=$UNIT hostguardian_invocation_id=$unit_invocation_id hostguardian_exec_start_monotonic=$unit_exec_start hostguardian_unit_survived=true material_unit=$MATERIAL_UNIT material_invocation_id=$MATERIAL_INVOCATION_ID material_cell_pid=$material_cell_before_pid material_cell_start_tick=$material_cell_before_start material_cell_cgroup_sha256=$material_cell_before_cgroup_sha256 material_tracee_pid=$MATERIAL_PID material_tracee_start_tick=$MATERIAL_START_TICK material_tracee_cgroup_sha256=$MATERIAL_CGROUP_SHA256 material_cell_main_pid_distinct_from_tracee=true run_grant_generation=$READY_RUN_GENERATION barrier_nonce_sha256=$READY_NONCE_SHA256 running_witness_sha256=$READY_WITNESS_SHA256 material_cell_survived=true replacement_sabotage=DENY592 release_authority=Sounio manifest_sha256=$EXPECTED_MANIFEST_SHA256 host_selftest_sha256=$HOST_SELFTEST_SHA256 host_probe_sha256=$HOST_PROBE_SHA256 transport=kubectl+privileged-hostPID+nsenter+systemd transport_trust=trusted-privileged-root-observer same_uid_peer_isolation=false hostile_transport_isolation=false capsule_layout=unpacked-directory-v1 shared_checkout=false material_execution=true compile_count=1 ticket_count=1 launch_count=1 result_count=1 attestation_count=1 controller_recovery=false pod_loss_synchronized=true transport_pod_loss_measured=true pod_loss_measured=true material_cell_survival_measured=true pod_loss_boundary=MATERIAL_RUNNING_IN_EXEC production_activation=false parity_open=false claim_ready=false host_output_sha256=$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)"
if [[ -n "$RECEIPT_OUTPUT" ]]; then
  stage="$(mktemp "$(dirname "$RECEIPT_OUTPUT")/.loom-causal-host-receipt.XXXXXX")"
  printf '%s\n%s\n' "$transport_receipt" "$host_output" > "$stage"
  install -m 0644 "$stage" "$RECEIPT_OUTPUT"
  rm -f "$stage"
fi
printf '%s\n%s\n' "$host_output" "$transport_receipt"
