#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_kernel_peer_bpf_load_v12.sh"
SELFTEST="$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_bpf_load_v12_selftest.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_bpf_load_v12_host_gate.sh"
LOADER_SOURCE="$ROOT_DIR/tools/loom/src/loom_bpf_lsm_loader_v12.cpp"
NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'

fail() {
  printf 'run-loom-kernel-peer-bpf-load-v12-host-probe: REFUSE reason=%s programs_loaded=false pin_survival=false link_extinction=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 claim_ready=false\n' "$*" >&2
  exit 70
}

for path in "$BUILDER" "$SELFTEST" "$HOST_GATE" "$LOADER_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required BPF-load component is absent or linked: $path"
done
for tool in kubectl sha256sum mktemp timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-bpf-load-v12.XXXXXX")"
BASE_INITRAMFS="$WORK/base.cpio.gz"
PACKER="$WORK/loom-newc-packer"
BPF_OBJECT="$WORK/policy.bpf.o"
cleanup_local() {
  rm -rf "$WORK"
}
trap cleanup_local EXIT

bash "$SELFTEST" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_BPF_LOAD_V12_OUTPUT="$BASE_INITRAMFS" \
SOUNIO_LOOM_KERNEL_PEER_BPF_LOAD_V12_PACKER_OUTPUT="$PACKER" \
SOUNIO_LOOM_KERNEL_PEER_BPF_LOAD_V12_BPF_OUTPUT="$BPF_OBJECT" \
  bash "$BUILDER" >/dev/null
BASE_INITRAMFS_SHA256="$(sha256sum "$BASE_INITRAMFS" | cut -d ' ' -f 1)"
PACKER_SHA256="$(sha256sum "$PACKER" | cut -d ' ' -f 1)"
BPF_OBJECT_SHA256="$(sha256sum "$BPF_OBJECT" | cut -d ' ' -f 1)"
LOADER_SOURCE_SHA256="$(sha256sum "$LOADER_SOURCE" | cut -d ' ' -f 1)"
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
REMOTE_BASE="/var/tmp/loom-kernel-peer-bpf-load-v12-base-$nonce.cpio.gz"
REMOTE_PACKER="/var/tmp/loom-kernel-peer-bpf-load-v12-packer-$nonce"
REMOTE_LOADER_SOURCE="/var/tmp/loom-kernel-peer-bpf-load-v12-loader-$nonce.cpp"
REMOTE_GATE="/var/tmp/loom-kernel-peer-bpf-load-v12-gate-$nonce"
cleanup_remote() {
  kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
    "rm -f '/proc/1/root$REMOTE_BASE' '/proc/1/root$REMOTE_PACKER' '/proc/1/root$REMOTE_LOADER_SOURCE' '/proc/1/root$REMOTE_GATE'" \
    >/dev/null 2>&1 || true
}
trap 'cleanup_remote; cleanup_local' EXIT

transfer_file() {
  local source="$1" destination="$2" mode="$3"
  kubectl -n "$NAMESPACE" exec -i "$POD" -- sh -c \
    "umask 077; cat > '/proc/1/root$destination'; chmod '$mode' '/proc/1/root$destination'; chown 0:0 '/proc/1/root$destination'" \
    < "$source"
}
transfer_file "$BASE_INITRAMFS" "$REMOTE_BASE" 0400
transfer_file "$PACKER" "$REMOTE_PACKER" 0500
transfer_file "$LOADER_SOURCE" "$REMOTE_LOADER_SOURCE" 0400
transfer_file "$HOST_GATE" "$REMOTE_GATE" 0500
KERNEL_SHA256="$(kubectl -n "$NAMESPACE" exec "$POD" -- nsenter -t 1 -m -u -i -n -p -- \
  /bin/bash -c 'sha256sum "/boot/vmlinuz-$(uname -r)" | cut -d " " -f 1')"
remote_hashes="$(kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
  "sha256sum '/proc/1/root$REMOTE_BASE' '/proc/1/root$REMOTE_PACKER' '/proc/1/root$REMOTE_LOADER_SOURCE' '/proc/1/root$REMOTE_GATE' | cut -d ' ' -f 1")"
mapfile -t remote_hash_lines <<< "$remote_hashes"
[[ "${remote_hash_lines[0]:-}" == "$BASE_INITRAMFS_SHA256" &&
   "${remote_hash_lines[1]:-}" == "$PACKER_SHA256" &&
   "${remote_hash_lines[2]:-}" == "$LOADER_SOURCE_SHA256" &&
   "${remote_hash_lines[3]:-}" == "$HOST_GATE_SHA256" ]] || fail 'host transport hash drifted'

set +e
host_output="$(timeout --signal=TERM --kill-after=10s 150s \
  kubectl -n "$NAMESPACE" exec "$POD" -- \
  nsenter -t 1 -m -u -i -n -p -- \
  /bin/bash "$REMOTE_GATE" \
    --base-initramfs "$REMOTE_BASE" --base-initramfs-sha256 "$BASE_INITRAMFS_SHA256" \
    --packer "$REMOTE_PACKER" --packer-sha256 "$PACKER_SHA256" \
    --loader-source "$REMOTE_LOADER_SOURCE" --loader-source-sha256 "$LOADER_SOURCE_SHA256" \
    --bpf-object-sha256 "$BPF_OBJECT_SHA256" --kernel-sha256 "$KERNEL_SHA256" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] || fail "host BPF-load gate failed status=$host_status output=$host_output"
summary="$(printf '%s\n' "$host_output" | grep '^sounio-loom-kernel-peer-bpf-load-v12-host-gate: HOST_MEASUREMENT_PASS ' || true)"
[[ -n "$summary" && "$(printf '%s\n' "$summary" | wc -l)" == 1 ]] ||
  fail 'host BPF-load summary is absent or duplicated'

printf '%s\n' "$host_output"
printf 'LOOM_KERNEL_PEER_BPF_LOAD_V12_HOST_TRANSPORT PASS namespace=%s node=%s pod=%s transport=kubectl+hostPID+nsenter base_initramfs_sha256=%s packer_sha256=%s bpf_object_sha256=%s loader_source_sha256=%s host_gate_sha256=%s kernel_sha256=%s host_output_sha256=%s semantic_authority=Sounio action=9025 programs_loaded=3 pin_survival=true link_extinction=true guest_disk=none guest_network=none material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false\n' \
  "$NAMESPACE" "$NODE" "$POD" "$BASE_INITRAMFS_SHA256" "$PACKER_SHA256" \
  "$BPF_OBJECT_SHA256" "$LOADER_SOURCE_SHA256" "$HOST_GATE_SHA256" "$KERNEL_SHA256" \
  "$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)"
