#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_host_durable_lane_capsule.sh"
PROMOTER="$ROOT_DIR/scripts/dev/promote_loom_host_exec_quorum_capsule.sh"
ACTION_BUILDER="$ROOT_DIR/scripts/dev/build_sounio_loom_host_durable_lane_supervisor_fixture.sh"
ACTION_FREEZE="$ROOT_DIR/tools/loom/host_durable_lane_supervisor.freeze.v1"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_host_durable_lane_supervisor_host_selftest.sh"

fail() {
  printf 'run-loom-host-durable-lane-supervisor-probe: REFUSE reason=%s same_physical_reattach=false transport_pod_deleted=false\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s [--namespace NAME] [--node NAME] [--image IMAGE] [--run-id ID] [--receipt-output PATH]\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

record_value() {
  local path="$1" key="$2" line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate field $key in $path"
      found="$value"
    fi
  done < "$path"
  [[ -n "$found" ]] || fail "missing field $key in $path"
  printf '%s\n' "$found"
}

pod_exists() {
  kubectl -n "$NAMESPACE" get pod "$1" >/dev/null 2>&1
}

create_transport_pod() {
  local pod="$1"
  kubectl -n "$NAMESPACE" create -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $pod
  labels:
    app.kubernetes.io/name: sounio-loom-durable-transport
    sounio.dev/loom-durable-run: $RUN_ID
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
  boundary="$(kubectl -n "$NAMESPACE" get pod "$pod" \
    -o jsonpath='{.spec.nodeName}|{.spec.hostPID}|{.spec.containers[0].securityContext.privileged}|{.status.phase}')"
  [[ "$boundary" == "$NODE|true|true|Running" ]] ||
    fail "transport Pod boundary drifted: pod=$pod boundary=$boundary"
  kubectl -n "$NAMESPACE" exec "$pod" -- sh -lc \
    'command -v nsenter >/dev/null && test "$(id -u)" = 0 && test "$(nsenter -t 1 -m -u -i -n -p -- tr -d "\n" </proc/1/comm)" = systemd' ||
    fail "transport Pod cannot enter host systemd namespace: $pod"
}

host_exec() {
  local pod="$1"
  shift
  kubectl -n "$NAMESPACE" exec "$pod" -- nsenter -t 1 -m -u -i -n -p -- "$@"
}

delete_transport_pod() {
  local pod="$1"
  kubectl -n "$NAMESPACE" delete pod "$pod" --wait=true --timeout=120s >/dev/null ||
    fail "transport Pod deletion failed: $pod"
  for _ in $(seq 1 120); do
    pod_exists "$pod" || return 0
    sleep 0.25
  done
  fail "transport Pod still exists after deletion: $pod"
}

fallback_transport_pod() {
  kubectl -n "$NAMESPACE" get pods \
    -l app.kubernetes.io/name=node-ephemeral-governance \
    --field-selector "spec.nodeName=$NODE,status.phase=Running" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null | head -n 1
}

NAMESPACE=beagle
NODE=t560-proxmox
IMAGE='192.168.3.207:5003/sounio-lab-beagle-workspace-ssh-20260425-zellij1:stable'
RUN_ID="d$(date -u +%s)-$$"
RECEIPT_OUTPUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
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
for input in "$BUILDER" "$PROMOTER" "$ACTION_BUILDER" "$ACTION_FREEZE" "$HOST_GATE"; do
  [[ -f "$input" && ! -L "$input" ]] || fail "required probe input is absent or linked: $input"
done
for executable in "$BUILDER" "$PROMOTER" "$ACTION_BUILDER" "$HOST_GATE"; do
  [[ -x "$executable" ]] || fail "required probe input is not executable: $executable"
done
for tool in kubectl sha256sum mktemp timeout install git; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done
[[ -z "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]] ||
  fail 'source worktree must be clean before building the host capsule'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-durable-probe.XXXXXX")"
CAPSULE="$WORK/host-exec-quorum.tar"
ACTION_RUNTIME="$WORK/action-9032"
POD_A="loom-durable-a-${RUN_ID}"
POD_B="loom-durable-b-${RUN_ID}"
HOST_ROOT="/var/tmp/sounio-loom-durable-$RUN_ID"
HOST_ROOT_CREATED=false

cleanup() {
  set +e
  cleanup_pod=''
  if pod_exists "$POD_B"; then
    cleanup_pod="$POD_B"
  elif pod_exists "$POD_A"; then
    cleanup_pod="$POD_A"
  else
    cleanup_pod="$(fallback_transport_pod)"
  fi
  if [[ "$HOST_ROOT_CREATED" == true && -n "$cleanup_pod" ]]; then
    host_exec "$cleanup_pod" "$HOST_ROOT/input/host-gate.sh" \
      --phase cleanup --root "$HOST_ROOT" --run-id "$RUN_ID" >/dev/null 2>&1 ||
      host_exec "$cleanup_pod" /usr/bin/rm -rf "$HOST_ROOT" >/dev/null 2>&1 || true
  fi
  pod_exists "$POD_A" && kubectl -n "$NAMESPACE" delete pod "$POD_A" \
    --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
  pod_exists "$POD_B" && kubectl -n "$NAMESPACE" delete pod "$POD_B" \
    --wait=false --grace-period=0 --force >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

build_output="$($BUILDER --output "$CAPSULE")"
[[ "$build_output" == 'LOOM_HOST_DURABLE_LANE_CAPSULE_BUILD PASS '* ]] ||
  fail "capsule build failed: $build_output"
CAPSULE_SHA256="$(sha256_file "$CAPSULE")"
verify_output="$($PROMOTER --archive "$CAPSULE" --expected-sha256 "$CAPSULE_SHA256" --mode verify)"
[[ "$verify_output" == 'LOOM_HOST_EXEC_QUORUM_CAPSULE_VERIFY PASS '* ]] ||
  fail "local capsule verification failed: $verify_output"

SOUNIO_LOOM_HOST_DURABLE_LANE_OUTPUT="$ACTION_RUNTIME" \
  "$ACTION_BUILDER" >/dev/null
ACTION_RUNTIME_SHA256="$(sha256_file "$ACTION_RUNTIME")"
EXPECTED_ACTION_SHA256="$(record_value "$ACTION_FREEZE" executable_sha256)"
[[ "$ACTION_RUNTIME_SHA256" == "$EXPECTED_ACTION_SHA256" ]] ||
  fail 'source-fresh action-9032 runtime differs from the semantic freeze'
HOST_GATE_SHA256="$(sha256_file "$HOST_GATE")"

create_transport_pod "$POD_A"
POD_A_UID="$(kubectl -n "$NAMESPACE" get pod "$POD_A" -o jsonpath='{.metadata.uid}')"
[[ "$POD_A_UID" =~ ^[0-9a-f-]{36}$ ]] || fail 'Pod-A UID is non-canonical'
kubectl -n "$NAMESPACE" exec "$POD_A" -- sh -c \
  "umask 077; mkdir -p '/proc/1/root$HOST_ROOT/input'; chmod 0700 '/proc/1/root$HOST_ROOT' '/proc/1/root$HOST_ROOT/input'"
HOST_ROOT_CREATED=true
kubectl -n "$NAMESPACE" exec -i "$POD_A" -- sh -c \
  "cat > '/proc/1/root$HOST_ROOT/input/capsule.tar'; chmod 0400 '/proc/1/root$HOST_ROOT/input/capsule.tar'" < "$CAPSULE"
kubectl -n "$NAMESPACE" exec -i "$POD_A" -- sh -c \
  "cat > '/proc/1/root$HOST_ROOT/input/action-9032'; chmod 0500 '/proc/1/root$HOST_ROOT/input/action-9032'" < "$ACTION_RUNTIME"
kubectl -n "$NAMESPACE" exec -i "$POD_A" -- sh -c \
  "cat > '/proc/1/root$HOST_ROOT/input/host-gate.sh'; chmod 0500 '/proc/1/root$HOST_ROOT/input/host-gate.sh'" < "$HOST_GATE"
remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD_A" -- sh -c \
  "sha256sum '/proc/1/root$HOST_ROOT/input/capsule.tar' '/proc/1/root$HOST_ROOT/input/action-9032' '/proc/1/root$HOST_ROOT/input/host-gate.sh' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$CAPSULE_SHA256" && \
   "${remote_hash_lines[1]:-}" == "$ACTION_RUNTIME_SHA256" && \
   "${remote_hash_lines[2]:-}" == "$HOST_GATE_SHA256" ]] ||
  fail 'Pod-A to host transport hashes drifted'

set +e
phase_a_output="$(timeout --signal=TERM --kill-after=10s 240s \
  kubectl -n "$NAMESPACE" exec "$POD_A" -- \
    nsenter -t 1 -m -u -i -n -p -- "$HOST_ROOT/input/host-gate.sh" \
    --phase prepare --root "$HOST_ROOT" --run-id "$RUN_ID" \
    --archive "$HOST_ROOT/input/capsule.tar" --archive-sha256 "$CAPSULE_SHA256" \
    --authority-runtime "$HOST_ROOT/input/action-9032" \
    --authority-runtime-sha256 "$ACTION_RUNTIME_SHA256" 2>&1)"
phase_a_status=$?
set -e
[[ $phase_a_status -eq 0 && "$phase_a_output" == \
  'sounio-loom-host-durable-lane-supervisor-host-selftest: PHASE_A_PASS '* ]] ||
  fail "host phase A failed or timed out status=$phase_a_status output=$phase_a_output"

delete_transport_pod "$POD_A"
[[ "$(kubectl -n "$NAMESPACE" get pod "$POD_A" --ignore-not-found -o name)" == '' ]] ||
  fail 'Pod A still exists before replacement transport is created'

create_transport_pod "$POD_B"
POD_B_UID="$(kubectl -n "$NAMESPACE" get pod "$POD_B" -o jsonpath='{.metadata.uid}')"
[[ "$POD_B_UID" =~ ^[0-9a-f-]{36}$ && "$POD_B_UID" != "$POD_A_UID" ]] ||
  fail 'Pod-B identity does not prove transport replacement'
[[ "$(kubectl -n "$NAMESPACE" get pod "$POD_A" --ignore-not-found -o name)" == '' ]] ||
  fail 'predecessor transport reappeared before host measurement'

set +e
host_output="$(timeout --signal=TERM --kill-after=10s 300s \
  kubectl -n "$NAMESPACE" exec "$POD_B" -- \
    nsenter -t 1 -m -u -i -n -p -- "$HOST_ROOT/input/host-gate.sh" \
    --phase measure --root "$HOST_ROOT" --run-id "$RUN_ID" \
    --transport-a-uid "$POD_A_UID" --transport-b-uid "$POD_B_UID" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 && "$host_output" == \
  'sounio-loom-host-durable-lane-supervisor-host-selftest: HOST_MEASUREMENT_PASS '* && \
  "$host_output" == *' transport_pod_deleted=true '* && \
  "$host_output" == *' same_physical_reattach=true '* && \
  "$host_output" == *' sounio_decision=SAME_PHYSICAL_REATTACH '* && \
  "$host_output" == *' sabotage_decision=DENY526 '* && \
  "$host_output" == *' full_extinction=true '* ]] ||
  fail "host measurement failed or timed out status=$host_status output=$host_output"

transport_receipt="LOOM_HOST_DURABLE_LANE_TRANSPORT PASS namespace=$NAMESPACE node=$NODE pod_a=$POD_A pod_a_uid=$POD_A_UID pod_a_deleted=true pod_b=$POD_B pod_b_uid=$POD_B_UID distinct_transport=true archive_sha256=$CAPSULE_SHA256 action_9032_runtime_sha256=$ACTION_RUNTIME_SHA256 host_gate_sha256=$HOST_GATE_SHA256 host_output_sha256=$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1) semantic_authority=Sounio operational_language=OCaml material_platform=Linux+systemd same_physical_reattach=true kernel_recovered=true causal_sabotage=PASS full_extinction=true tmux_used=false python_executed=false rust_executed=false production_activation=false parity_open=false claim_ready=false"
if [[ -n "$RECEIPT_OUTPUT" ]]; then
  receipt_stage="$(mktemp "$(dirname "$RECEIPT_OUTPUT")/.loom-durable-receipt.XXXXXX")"
  printf '%s\n%s\n%s\n' "$phase_a_output" "$host_output" "$transport_receipt" > "$receipt_stage"
  install -m 0644 "$receipt_stage" "$RECEIPT_OUTPUT"
  rm -f "$receipt_stage"
fi

host_exec "$POD_B" "$HOST_ROOT/input/host-gate.sh" \
  --phase cleanup --root "$HOST_ROOT" --run-id "$RUN_ID" >/dev/null
HOST_ROOT_CREATED=false
delete_transport_pod "$POD_B"
trap - EXIT
rm -rf "$WORK"
printf '%s\n%s\n%s\n' "$phase_a_output" "$host_output" "$transport_receipt"
