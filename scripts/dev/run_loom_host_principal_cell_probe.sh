#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_host_principal_cell.sh"
LOCAL_GATE="$ROOT_DIR/scripts/ci/sounio_loom_host_principal_cell_selftest.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_host_principal_cell_host_gate.sh"

fail() {
  printf 'run-loom-host-principal-cell-probe: REFUSE reason=%s kernel_distinct_principal_candidate=false material_grant=false launch_open=false\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s [--namespace NAME] [--node NAME] [--selector LABELS] [--receipt-output PATH]\n' "$0" >&2
  exit 64
}

NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'
RECEIPT_OUTPUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace)
      [[ $# -ge 2 ]] || usage
      NAMESPACE="$2"
      shift 2
      ;;
    --node)
      [[ $# -ge 2 ]] || usage
      NODE="$2"
      shift 2
      ;;
    --selector)
      [[ $# -ge 2 ]] || usage
      SELECTOR="$2"
      shift 2
      ;;
    --receipt-output)
      [[ $# -ge 2 ]] || usage
      RECEIPT_OUTPUT="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$NAMESPACE" =~ ^[a-z0-9.-]+$ && "$NODE" =~ ^[A-Za-z0-9._-]+$ ]] || fail 'namespace or node name is unsafe'
[[ "$SELECTOR" =~ ^[A-Za-z0-9._,/=-]+$ ]] || fail 'pod selector is unsafe'
for path in "$BUILDER" "$LOCAL_GATE" "$HOST_GATE"; do
  [[ -f "$path" && ! -L "$path" && -x "$path" ]] || fail "required probe input is absent, linked, or non-executable: $path"
done
for tool in kubectl sha256sum mktemp install timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

bash "$LOCAL_GATE" >/dev/null
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-principal-cell-run.XXXXXX")"
BINARY="$WORK/loom-host-principal-cell"
POD=''
REMOTE_BINARY=''
REMOTE_GATE=''
cleanup() {
  if [[ -n "$POD" && -n "$REMOTE_BINARY" && -n "$REMOTE_GATE" ]]; then
    kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
      "rm -f '/proc/1/root$REMOTE_BINARY' '/proc/1/root$REMOTE_GATE'" >/dev/null 2>&1 || true
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

SOUNIO_LOOM_HOST_PRINCIPAL_CELL_OUTPUT="$BINARY" bash "$BUILDER" >/dev/null
BINARY_SHA256="$(sha256sum "$BINARY" | cut -d ' ' -f 1)"
HOST_GATE_SHA256="$(sha256sum "$HOST_GATE" | cut -d ' ' -f 1)"
SOURCE_SHA256="$(sha256sum "$ROOT_DIR/tools/loom/src/loom_host_principal_cell.cpp" | cut -d ' ' -f 1)"
ACTION_MANIFEST_SHA256="$(sha256sum "$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1" | cut -d ' ' -f 1)"
[[ "$ACTION_MANIFEST_SHA256" == 8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051 ]] ||
  fail 'frozen Sounio action 9030 manifest drifted before host transport'

mapfile -t candidate_pods < <(
  kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" \
    --field-selector "spec.nodeName=$NODE,status.phase=Running" -o name
)
[[ ${#candidate_pods[@]} -eq 1 ]] || fail "expected one host transport pod on $NODE; found ${#candidate_pods[@]}"
POD="${candidate_pods[0]#pod/}"
[[ "$POD" =~ ^[a-z0-9.-]+$ ]] || fail 'selected transport pod name is unsafe'
pod_boundary="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.hostPID}|{.spec.containers[0].securityContext.privileged}|{.status.phase}')"
[[ "$pod_boundary" == 'true|true|Running' ]] || fail "transport pod lacks privileged hostPID boundary: $pod_boundary"

kubectl -n "$NAMESPACE" exec "$POD" -- sh -lc \
  'command -v nsenter >/dev/null && test "$(id -u)" = 0 && test "$(nsenter -t 1 -m -u -i -n -p -- tr -d "\n" </proc/1/comm)" = systemd' ||
  fail 'transport pod cannot enter the host systemd namespaces'

REMOTE_BINARY="/usr/local/libexec/sounio/loom-host-principal-cell-$BINARY_SHA256"
REMOTE_GATE="/var/tmp/loom-host-principal-cell-gate-$HOST_GATE_SHA256"
kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
  "install -d -m 0755 -o 0 -g 0 /proc/1/root/usr/local/libexec/sounio; umask 022; cat > '/proc/1/root$REMOTE_BINARY'; chmod 0555 '/proc/1/root$REMOTE_BINARY'" < "$BINARY"
kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
  "umask 077; cat > '/proc/1/root$REMOTE_GATE'; chmod 0500 '/proc/1/root$REMOTE_GATE'" < "$HOST_GATE"

remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "sha256sum '/proc/1/root$REMOTE_BINARY' '/proc/1/root$REMOTE_GATE' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$BINARY_SHA256" ]] || fail 'binary hash drifted before host namespace entry'
[[ "${remote_hash_lines[1]:-}" == "$HOST_GATE_SHA256" ]] || fail 'host gate hash drifted before host namespace entry'

set +e
host_output="$(timeout --signal=TERM --kill-after=5s 150s \
  kubectl -n "$NAMESPACE" exec "$POD" -- nsenter -t 1 -m -u -i -n -p -- \
  "$REMOTE_GATE" --binary "$REMOTE_BINARY" --expected-sha256 "$BINARY_SHA256" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] || fail "host PrincipalCell gate failed or timed out status=$host_status output=$host_output"
[[ "$host_output" == 'sounio-loom-host-principal-cell-host-gate: HOST_MEASUREMENT_PASS '* ]] || fail 'host gate did not return the expected pass receipt'

transport_receipt="LOOM_HOST_PRINCIPAL_CELL_TRANSPORT PASS semantic_authority=Sounio action=9030 material_role=MECHANICAL_TRANSPORT namespace=$NAMESPACE node=$NODE pod=$POD action_manifest_sha256=$ACTION_MANIFEST_SHA256 source_sha256=$SOURCE_SHA256 binary_sha256=$BINARY_SHA256 host_gate_sha256=$HOST_GATE_SHA256 transport=kubectl+hostPID+nsenter host_output_sha256=$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1) kernel_distinct_principal_candidate=true material_grant=false grant_extinction=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false launch_open=false"
if [[ -n "$RECEIPT_OUTPUT" ]]; then
  receipt_stage="$(mktemp "$(dirname "$RECEIPT_OUTPUT")/.loom-principal-cell-receipt.XXXXXX")"
  printf '%s\n%s\n' "$transport_receipt" "$host_output" > "$receipt_stage"
  install -m 0644 "$receipt_stage" "$RECEIPT_OUTPUT"
  rm -f "$receipt_stage"
fi
printf '%s\n%s\n' "$host_output" "$transport_receipt"
