#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_kernel_peer_microhost_v12.sh"
SELFTEST="$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_microhost_v12_selftest.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_microhost_v12_host_gate.sh"
NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'

fail() {
  printf 'run-loom-kernel-peer-microhost-v12-host-probe: REFUSE reason=%s bpf_lsm_microhost=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 claim_ready=false\n' "$*" >&2
  exit 70
}

for path in "$BUILDER" "$SELFTEST" "$HOST_GATE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required microhost component is absent or linked: $path"
done
for tool in kubectl sha256sum mktemp timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-microhost-v12.XXXXXX")"
INITRAMFS="$WORK/microhost.cpio.gz"
cleanup_local() {
  rm -rf "$WORK"
}
trap cleanup_local EXIT

bash "$SELFTEST" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_MICROHOST_V12_OUTPUT="$INITRAMFS" bash "$BUILDER" >/dev/null
INITRAMFS_SHA256="$(sha256sum "$INITRAMFS" | cut -d ' ' -f 1)"
HOST_GATE_SHA256="$(sha256sum "$HOST_GATE" | cut -d ' ' -f 1)"

mapfile -t candidate_pods < <(
  kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" \
    --field-selector "spec.nodeName=$NODE,status.phase=Running" -o name
)
[[ ${#candidate_pods[@]} -eq 1 ]] ||
  fail "expected one host transport pod on $NODE; found ${#candidate_pods[@]}"
POD="${candidate_pods[0]#pod/}"
[[ "$POD" =~ ^[a-z0-9.-]+$ ]] || fail 'selected pod name is unsafe'
pod_boundary="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.hostPID}|{.spec.containers[0].securityContext.privileged}|{.status.phase}')"
[[ "$pod_boundary" == 'true|true|Running' ]] ||
  fail "transport pod lacks privileged hostPID boundary: $pod_boundary"

nonce="$$-$(date +%s%N)"
REMOTE_INITRAMFS="/var/tmp/loom-kernel-peer-microhost-v12-$nonce.cpio.gz"
REMOTE_GATE="/var/tmp/loom-kernel-peer-microhost-v12-gate-$nonce"
cleanup_remote() {
  kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
    "rm -f '/proc/1/root$REMOTE_INITRAMFS' '/proc/1/root$REMOTE_GATE'" \
    >/dev/null 2>&1 || true
}
trap 'cleanup_remote; cleanup_local' EXIT

transfer_file() {
  local source="$1" destination="$2" mode="$3"
  kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
    "umask 077; cat > '/proc/1/root$destination'; chmod '$mode' '/proc/1/root$destination'; chown 0:0 '/proc/1/root$destination'" \
    < "$source"
}
transfer_file "$INITRAMFS" "$REMOTE_INITRAMFS" 0400
transfer_file "$HOST_GATE" "$REMOTE_GATE" 0500
KERNEL_SHA256="$(kubectl -n "$NAMESPACE" exec "$POD" -- nsenter -t 1 -m -u -i -n -p -- \
  /bin/bash -c 'sha256sum "/boot/vmlinuz-$(uname -r)" | cut -d " " -f 1')"
remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "sha256sum '/proc/1/root$REMOTE_INITRAMFS' '/proc/1/root$REMOTE_GATE' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$INITRAMFS_SHA256" &&
   "${remote_hash_lines[1]:-}" == "$HOST_GATE_SHA256" ]] || fail 'host transport hash drifted'

set +e
host_output="$(timeout --signal=TERM --kill-after=10s 90s \
  kubectl -n "$NAMESPACE" exec "$POD" -- \
  nsenter -t 1 -m -u -i -n -p -- \
  /bin/bash "$REMOTE_GATE" --initramfs "$REMOTE_INITRAMFS" \
    --initramfs-sha256 "$INITRAMFS_SHA256" --kernel-sha256 "$KERNEL_SHA256" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] || fail "host microhost gate failed status=$host_status output=$host_output"
summary="$(printf '%s\n' "$host_output" | grep '^sounio-loom-kernel-peer-microhost-v12-host-gate: HOST_MEASUREMENT_PASS ' || true)"
[[ -n "$summary" && "$(printf '%s\n' "$summary" | wc -l)" == 1 ]] ||
  fail 'host microhost summary is absent or duplicated'

printf '%s\n' "$host_output"
printf 'LOOM_KERNEL_PEER_MICROHOST_V12_HOST_TRANSPORT PASS namespace=%s node=%s pod=%s transport=kubectl+hostPID+nsenter initramfs_sha256=%s host_gate_sha256=%s kernel_sha256=%s host_output_sha256=%s semantic_authority=Sounio action=9025 bpf_lsm_microhost=true guest_disk=none guest_network=none material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false\n' \
  "$NAMESPACE" "$NODE" "$POD" "$INITRAMFS_SHA256" "$HOST_GATE_SHA256" \
  "$KERNEL_SHA256" "$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)"
