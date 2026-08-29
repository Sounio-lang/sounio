#!/usr/bin/env bash

set -euo pipefail
umask 077

INITRAMFS=''
INITRAMFS_SHA256=''
KERNEL_SHA256=''

fail() {
  printf 'sounio-loom-kernel-peer-microhost-v12-host-gate: FAIL: %s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --initramfs PATH --initramfs-sha256 HEX --kernel-sha256 HEX\n' "$0" >&2
  exit 64
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "microhost receipt omitted field: $key"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --initramfs) [[ $# -ge 2 ]] || usage; INITRAMFS="$2"; shift 2 ;;
    --initramfs-sha256) [[ $# -ge 2 ]] || usage; INITRAMFS_SHA256="$2"; shift 2 ;;
    --kernel-sha256) [[ $# -ge 2 ]] || usage; KERNEL_SHA256="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$INITRAMFS" == /var/tmp/loom-kernel-peer-microhost-v12-* ]] || fail 'initramfs path is unsafe'
[[ "$INITRAMFS_SHA256" =~ ^[0-9a-f]{64}$ && "$KERNEL_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'microhost input hash is malformed'
[[ "$(id -u)" == 0 ]] || fail 'KVM gate requires the root guardian'
for tool in qemu-system-x86_64 sha256sum timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required host tool is absent: $tool"
done
[[ -c /dev/kvm ]] || fail '/dev/kvm is absent'
[[ -f "$INITRAMFS" && ! -L "$INITRAMFS" ]] || fail 'initramfs is absent or linked'
[[ "$(sha256sum "$INITRAMFS" | cut -d ' ' -f 1)" == "$INITRAMFS_SHA256" ]] ||
  fail 'initramfs hash drifted'
KERNEL="/boot/vmlinuz-$(uname -r)"
[[ -f "$KERNEL" && ! -L "$KERNEL" && -r "$KERNEL" ]] || fail 'host kernel image is unavailable'
[[ "$(sha256sum "$KERNEL" | cut -d ' ' -f 1)" == "$KERNEL_SHA256" ]] || fail 'kernel image hash drifted'
HOST_BOOT_ID="$(tr -d '\n' < /proc/sys/kernel/random/boot_id)"

set +e
qemu_output="$(timeout --signal=TERM --kill-after=5s 45s \
  qemu-system-x86_64 \
    -name loom-kernel-peer-microhost-v12 \
    -machine q35,accel=kvm \
    -cpu host \
    -m 512M \
    -smp 1 \
    -nodefaults \
    -no-user-config \
    -display none \
    -serial stdio \
    -monitor none \
    -no-reboot \
    -nic none \
    -kernel "$KERNEL" \
    -initrd "$INITRAMFS" \
    -append 'rdinit=/init console=ttyS0 quiet loglevel=3 panic=-1 lsm=lockdown,capability,yama,apparmor,bpf,ima,evm' \
    2>&1)"
qemu_status=$?
set -e
[[ $qemu_status -eq 0 ]] || fail "KVM microhost failed status=$qemu_status output=$qemu_output"
[[ "$(printf '%s' "$qemu_output" | wc -c)" -le 1048576 ]] || fail 'KVM output crossed 1 MiB bound'
summary="$(printf '%s\n' "$qemu_output" | tr -d '\r' | grep '^LOOM_KERNEL_PEER_MICROHOST_V12_BOOT PASS ' || true)"
[[ -n "$summary" && "$(printf '%s\n' "$summary" | wc -l)" == 1 ]] ||
  fail "microhost boot receipt is absent or duplicated: $qemu_output"
GUEST_BOOT_ID="$(field "$summary" boot_id)"
ACTIVE_LSM="$(field "$summary" active_lsm)"
[[ "$GUEST_BOOT_ID" =~ ^[0-9a-f-]{36}$ && "$GUEST_BOOT_ID" != "$HOST_BOOT_ID" ]] ||
  fail 'guest boot identity is absent or aliases the host'
[[ ",$ACTIVE_LSM," == *,bpf,* ]] || fail 'guest BPF LSM is not active'
for fact in pid=1 bpf_lsm_active=true securityfs=true bpffs=true btf=true guest_disk=none guest_network=none init_language=C++20 init_role=MATERIAL_BOOTSTRAP semantic_authority=Sounio action=9025 material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
  [[ " $summary " == *" $fact "* ]] || fail "microhost receipt omitted $fact"
done
if pgrep -f 'qemu-system-x86_64.*loom-kernel-peer-microhost-v12' >/dev/null 2>&1; then
  fail 'KVM microhost process did not become extinct'
fi

printf '%s\n' "$qemu_output"
printf 'sounio-loom-kernel-peer-microhost-v12-host-gate: HOST_MEASUREMENT_PASS host=%s kernel=%s kernel_sha256=%s hypervisor=KVM qemu_version=11.0.0 guest_boot_id=%s host_boot_id=%s guest_distinct=true active_lsm=%s bpf_lsm_active=true securityfs=true bpffs=true btf=true guest_disk=none guest_network=none qemu_extinct=true initramfs_sha256=%s semantic_authority=Sounio action=9025 material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 claim_ready=false\n' \
  "$(hostname)" "$(uname -r)" "$KERNEL_SHA256" "$GUEST_BOOT_ID" "$HOST_BOOT_ID" \
  "$ACTIVE_LSM" "$INITRAMFS_SHA256"
