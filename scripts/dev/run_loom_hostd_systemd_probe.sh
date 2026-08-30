#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
INSTALLER="$ROOT_DIR/scripts/dev/install_loom_hostd.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_hostd_systemd_host_selftest.sh"
EXEC_CELL_CAPSULE_BUILDER="$ROOT_DIR/scripts/dev/build_loom_host_exec_quorum_capsule.sh"

fail() {
  printf 'run-loom-hostd-systemd-probe: REFUSE reason=%s real_systemd_activation=false full_extinction=false\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s [--namespace NAME] [--node NAME] [--image IMAGE] [--run-id ID] [--receipt-output PATH]\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

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
    app.kubernetes.io/name: sounio-loom-hostd-systemd-transport
    sounio.dev/loom-hostd-systemd-run: $RUN_ID
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
RUN_ID="h$(date -u +%s)-$$"
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

[[ "$NAMESPACE" =~ ^[a-z0-9.-]+$ && "$NODE" =~ ^[A-Za-z0-9._-]+$ &&
   "$IMAGE" =~ ^[A-Za-z0-9._:/-]+$ && "$RUN_ID" =~ ^[a-z0-9-]{8,48}$ ]] || usage
for input in "$INSTALLER" "$HOST_GATE" "$EXEC_CELL_CAPSULE_BUILDER"; do
  [[ -f "$input" && ! -L "$input" && -x "$input" ]] ||
    fail "required probe input is absent, linked, or non-executable: $input"
done
for tool in kubectl sha256sum mktemp timeout install tar git; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done
[[ -z "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]] ||
  fail 'source worktree must be clean before building the host bundle'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-hostd-systemd-probe.XXXXXX")"
STAGE_ROOT="$WORK/stage"
BUNDLE="$WORK/bundle-v1"
ARCHIVE="$WORK/loom-hostd-systemd.tar"
POD_A="loom-hostd-a-${RUN_ID}"
POD_B="loom-hostd-b-${RUN_ID}"
HOST_ROOT="/var/tmp/sounio-loom-hostd-systemd-$RUN_ID"
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
  find "$WORK" -type d -exec chmod u+rwx {} + 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p "$STAGE_ROOT"
EXEC_CELL_CAPSULE="$WORK/loom-host-exec-cell.tar"
bash "$EXEC_CELL_CAPSULE_BUILDER" --output "$EXEC_CELL_CAPSULE" >/dev/null
EXEC_CELL_CAPSULE_SHA256="$(sha256_file "$EXEC_CELL_CAPSULE")"
stage_output="$(bash "$INSTALLER" --install-root "$STAGE_ROOT" --user root \
  --exec-cell-capsule "$EXEC_CELL_CAPSULE" \
  --exec-cell-capsule-sha256 "$EXEC_CELL_CAPSULE_SHA256")"
[[ "$stage_output" == *'exec_cell_bundle_present=true '* &&
   "$stage_output" == *'exec_cell_boot_gate_configured=true '* &&
   "$stage_output" == *'exec_attached=false activated=false '* &&
   "$stage_output" == *'automatic_lineage_resurrection=false'* ]] ||
  fail "local staged install widened activation: $stage_output"
mkdir -p "$BUNDLE"
mv "$STAGE_ROOT" "$BUNDLE/stage"
install -m 0400 "$EXEC_CELL_CAPSULE" "$BUNDLE/exec-cell-capsule.tar"
install -m 0555 "$INSTALLER" "$BUNDLE/install_loom_hostd.sh"
install -m 0555 "$HOST_GATE" "$BUNDLE/host-gate.sh"
cat > "$BUNDLE/bundle-manifest.v1" <<EOF
schema=loom-hostd-systemd-bundle-v1
source_commit=$(git -C "$ROOT_DIR" rev-parse HEAD)
installer_sha256=$(sha256_file "$INSTALLER")
host_gate_sha256=$(sha256_file "$HOST_GATE")
staged_manifest_sha256=$(sha256_file "$BUNDLE/stage/opt/sounio/loom-hostd/manifest.v1")
exec_cell_capsule_sha256=$EXEC_CELL_CAPSULE_SHA256
semantic_authority=Sounio
semantic_actions=9030,9031,9033,9041
operational_language=OCaml
exec_cell_boot_gate_configured=true
exec_cell_boot_gate_test_only=true
exact_fixture_result_attached=false
exec_attached=false
production_activation=false
python_executed=false
rust_executed=false
EOF
chmod 0444 "$BUNDLE/bundle-manifest.v1"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -cf "$ARCHIVE" -C "$WORK" bundle-v1
ARCHIVE_SHA256="$(sha256_file "$ARCHIVE")"
HOST_GATE_SHA256="$(sha256_file "$HOST_GATE")"

create_transport_pod "$POD_A"
POD_A_UID="$(kubectl -n "$NAMESPACE" get pod "$POD_A" -o jsonpath='{.metadata.uid}')"
[[ "$POD_A_UID" =~ ^[0-9a-f-]{36}$ ]] || fail 'Pod-A UID is non-canonical'
kubectl -n "$NAMESPACE" exec "$POD_A" -- sh -c \
  "umask 077; mkdir -p '/proc/1/root$HOST_ROOT/input'; chmod 0700 '/proc/1/root$HOST_ROOT' '/proc/1/root$HOST_ROOT/input'"
HOST_ROOT_CREATED=true
kubectl -n "$NAMESPACE" exec -i "$POD_A" -- sh -c \
  "cat > '/proc/1/root$HOST_ROOT/input/bundle.tar'; chmod 0400 '/proc/1/root$HOST_ROOT/input/bundle.tar'" < "$ARCHIVE"
kubectl -n "$NAMESPACE" exec -i "$POD_A" -- sh -c \
  "cat > '/proc/1/root$HOST_ROOT/input/host-gate.sh'; chmod 0500 '/proc/1/root$HOST_ROOT/input/host-gate.sh'" < "$HOST_GATE"
remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD_A" -- sh -c \
  "sha256sum '/proc/1/root$HOST_ROOT/input/bundle.tar' '/proc/1/root$HOST_ROOT/input/host-gate.sh' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$ARCHIVE_SHA256" &&
   "${remote_hash_lines[1]:-}" == "$HOST_GATE_SHA256" ]] ||
  fail 'Pod-A to host transport hashes drifted'

set +e
phase_a_output="$(timeout --signal=TERM --kill-after=10s 300s \
  kubectl -n "$NAMESPACE" exec "$POD_A" -- \
    nsenter -t 1 -m -u -i -n -p -- "$HOST_ROOT/input/host-gate.sh" \
    --phase prepare --root "$HOST_ROOT" --run-id "$RUN_ID" \
    --archive "$HOST_ROOT/input/bundle.tar" --archive-sha256 "$ARCHIVE_SHA256" 2>&1)"
phase_a_status=$?
set -e
[[ $phase_a_status -eq 0 && "$phase_a_output" == \
  'sounio-loom-hostd-systemd-host-selftest: PHASE_A_PASS '* ]] ||
  fail "host phase A failed or timed out status=$phase_a_status output=$phase_a_output"

delete_transport_pod "$POD_A"
[[ "$(kubectl -n "$NAMESPACE" get pod "$POD_A" --ignore-not-found -o name)" == '' ]] ||
  fail 'Pod A still exists before replacement transport is created'
create_transport_pod "$POD_B"
POD_B_UID="$(kubectl -n "$NAMESPACE" get pod "$POD_B" -o jsonpath='{.metadata.uid}')"
[[ "$POD_B_UID" =~ ^[0-9a-f-]{36}$ && "$POD_B_UID" != "$POD_A_UID" ]] ||
  fail 'Pod-B identity does not prove transport replacement'

set +e
host_output="$(timeout --signal=TERM --kill-after=10s 360s \
  kubectl -n "$NAMESPACE" exec "$POD_B" -- \
    nsenter -t 1 -m -u -i -n -p -- "$HOST_ROOT/input/host-gate.sh" \
    --phase measure --root "$HOST_ROOT" --run-id "$RUN_ID" \
    --transport-a-uid "$POD_A_UID" --transport-b-uid "$POD_B_UID" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 && "$host_output" == \
  'sounio-loom-hostd-systemd-host-selftest: HOST_MEASUREMENT_PASS '* &&
   "$host_output" == *' real_systemd_activation=true '* &&
   "$host_output" == *' exact_fixture_result_attached=true '* &&
   "$host_output" == *' result_returned=true result_presenter=read-only '* &&
   "$host_output" == *' supervisor_restarted=true '* &&
   "$host_output" == *' sabotage_decision=DENY545 '* &&
   "$host_output" == *' full_extinction=true '* ]] ||
  fail "host measurement failed or timed out status=$host_status output=$host_output"

transport_receipt="LOOM_HOSTD_SYSTEMD_TRANSPORT PASS namespace=$NAMESPACE node=$NODE pod_a=$POD_A pod_a_uid=$POD_A_UID pod_a_deleted=true pod_b=$POD_B pod_b_uid=$POD_B_UID distinct_transport=true archive_sha256=$ARCHIVE_SHA256 host_gate_sha256=$HOST_GATE_SHA256 host_output_sha256=$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1) source_commit=$(git -C "$ROOT_DIR" rev-parse HEAD) semantic_authority=Sounio actions=9030,9031,9033,9041 operational_language=OCaml material_platform=Linux+systemd exec_cell_boot_gate=true exec_cell_boot_gate_test_only=true exact_fixture_result_attached=true result_returned=true result_presenter=read-only exec_attached=false real_systemd_activation=true same_physical_recovery=true causal_sabotage=PASS full_extinction=true tmux_used=false python_executed=false rust_executed=false production_activation=canary-only"
if [[ -n "$RECEIPT_OUTPUT" ]]; then
  mkdir -p "$(dirname "$RECEIPT_OUTPUT")"
  receipt_stage="$(mktemp "$(dirname "$RECEIPT_OUTPUT")/.loom-hostd-systemd-receipt.XXXXXX")"
  printf '%s\n%s\n%s\n' "$phase_a_output" "$host_output" "$transport_receipt" > "$receipt_stage"
  install -m 0644 "$receipt_stage" "$RECEIPT_OUTPUT"
  rm -f "$receipt_stage"
fi

host_exec "$POD_B" "$HOST_ROOT/input/host-gate.sh" \
  --phase cleanup --root "$HOST_ROOT" --run-id "$RUN_ID" >/dev/null
HOST_ROOT_CREATED=false
delete_transport_pod "$POD_B"
trap - EXIT
find "$WORK" -type d -exec chmod u+rwx {} + 2>/dev/null || true
rm -rf "$WORK"
printf '%s\n%s\n%s\n' "$phase_a_output" "$host_output" "$transport_receipt"
