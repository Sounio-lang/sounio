#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_kernel_peer_dumpable_prlimit_falsification_v12.sh"
SELFTEST="$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_dumpable_prlimit_falsification_v12_selftest.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_kernel_peer_dumpable_prlimit_falsification_v12_host_gate.sh"
NAMESPACE=beagle
NODE=t560-proxmox
SELECTOR='app.kubernetes.io/name=node-ephemeral-governance'

fail() {
  printf 'run-loom-kernel-peer-dumpable-prlimit-falsification-v12-host-probe: REFUSE reason=%s observations=0 controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 claim_ready=false\n' "$*" >&2
  exit 70
}
for path in "$BUILDER" "$SELFTEST" "$HOST_GATE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required component is absent or linked: $path"
done
for tool in kubectl sha256sum mktemp timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required transport tool is absent: $tool"
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-dumpable-prlimit-v12.XXXXXX")"
BASE_INITRAMFS="$WORK/base.cpio.gz"
PACKER="$WORK/loom-newc-packer"
cleanup_local() { rm -rf "$WORK"; }
trap cleanup_local EXIT
bash "$SELFTEST" >/dev/null 2>&1
SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_OUTPUT="$BASE_INITRAMFS" \
SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_PACKER_OUTPUT="$PACKER" \
  bash "$BUILDER" >/dev/null 2>&1
BASE_INITRAMFS_SHA256="$(sha256sum "$BASE_INITRAMFS" | cut -d ' ' -f 1)"
PACKER_SHA256="$(sha256sum "$PACKER" | cut -d ' ' -f 1)"
HOST_GATE_SHA256="$(sha256sum "$HOST_GATE" | cut -d ' ' -f 1)"

mapfile -t candidate_pods < <(
  kubectl -n "$NAMESPACE" get pods -l "$SELECTOR" \
    --field-selector "spec.nodeName=$NODE,status.phase=Running" -o name
)
[[ ${#candidate_pods[@]} -eq 1 ]] ||
  fail "expected one host transport pod; found ${#candidate_pods[@]}"
POD="${candidate_pods[0]#pod/}"
[[ "$POD" =~ ^[a-z0-9.-]+$ ]] || fail 'pod name is unsafe'
pod_boundary="$(kubectl -n "$NAMESPACE" get pod "$POD" -o jsonpath='{.spec.hostPID}|{.spec.containers[0].securityContext.privileged}|{.status.phase}')"
[[ "$pod_boundary" == 'true|true|Running' ]] || fail "transport boundary refused: $pod_boundary"

nonce="$$-$(date +%s%N)"
REMOTE_BASE="/var/tmp/loom-kernel-peer-dumpable-prlimit-v12-base-$nonce.cpio.gz"
REMOTE_PACKER="/var/tmp/loom-kernel-peer-dumpable-prlimit-v12-packer-$nonce"
REMOTE_GATE="/var/tmp/loom-kernel-peer-dumpable-prlimit-v12-gate-$nonce"
cleanup_remote() {
  kubectl -n "$NAMESPACE" exec "$POD" -- sh -c \
    "rm -f '/proc/1/root$REMOTE_BASE' '/proc/1/root$REMOTE_PACKER' '/proc/1/root$REMOTE_GATE'" \
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
transfer_file "$HOST_GATE" "$REMOTE_GATE" 0500
KERNEL_SHA256="$(kubectl -n "$NAMESPACE" exec "$POD" -- nsenter -t 1 -m -u -i -n -p -- \
  /bin/bash -c 'sha256sum "/boot/vmlinuz-$(uname -r)" | cut -d " " -f 1')"

set +e
host_output="$(timeout --signal=TERM --kill-after=10s 180s \
  kubectl -n "$NAMESPACE" exec "$POD" -- nsenter -t 1 -m -u -i -n -p -- \
  /bin/bash "$REMOTE_GATE" \
    --base-initramfs "$REMOTE_BASE" --base-initramfs-sha256 "$BASE_INITRAMFS_SHA256" \
    --packer "$REMOTE_PACKER" --packer-sha256 "$PACKER_SHA256" \
    --kernel-sha256 "$KERNEL_SHA256" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] || fail "host falsification gate failed status=$host_status output=$host_output"
summary="$(printf '%s\n' "$host_output" | grep '^sounio-loom-kernel-peer-dumpable-prlimit-falsification-v12-host-gate: HOST_MEASUREMENT_PASS ' || true)"
[[ -n "$summary" && "$(printf '%s\n' "$summary" | wc -l)" == 1 ]] ||
  fail 'host summary absent or duplicated'
MATERIAL_OBSERVED=''
V12_HYPOTHESIS_FALSIFIED=''
for token in $summary; do
  [[ "$token" == material_observed=* ]] && MATERIAL_OBSERVED="${token#*=}"
  [[ "$token" == v12_hypothesis_falsified=* ]] && V12_HYPOTHESIS_FALSIFIED="${token#*=}"
done
[[ "$MATERIAL_OBSERVED" == EFFECT_COMPLETED || "$MATERIAL_OBSERVED" == REFUSED_BEFORE_EFFECT ]] ||
  fail 'host summary carried an inadmissible observation'
[[ "$V12_HYPOTHESIS_FALSIFIED" == true || "$V12_HYPOTHESIS_FALSIFIED" == false ]] ||
  fail 'host summary carried an inadmissible hypothesis decision'

printf '%s\n' "$host_output"
printf 'LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_V12_HOST_TRANSPORT PASS namespace=%s node=%s pod=%s transport=kubectl+hostPID+nsenter base_initramfs_sha256=%s packer_sha256=%s host_gate_sha256=%s kernel_sha256=%s host_output_sha256=%s semantic_authority=Sounio action=9025 operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT material_observed=%s v12_hypothesis_falsified=%s same_four_uids=true target_dumpable=0 attacker_seccomp=0 mediator=absent principal_capability=CAP_SYS_NICE_ONLY typed_witness=true all_epoch_objects_extinct=true controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false next_stage=SOUNIO_V13_GARDEN\n' \
  "$NAMESPACE" "$NODE" "$POD" "$BASE_INITRAMFS_SHA256" "$PACKER_SHA256" \
  "$HOST_GATE_SHA256" "$KERNEL_SHA256" \
  "$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)" \
  "$MATERIAL_OBSERVED" "$V12_HYPOTHESIS_FALSIFIED"
