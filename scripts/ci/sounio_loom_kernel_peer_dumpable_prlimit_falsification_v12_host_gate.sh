#!/usr/bin/env bash

set -euo pipefail
umask 077

BASE_INITRAMFS=''
BASE_INITRAMFS_SHA256=''
PACKER=''
PACKER_SHA256=''
KERNEL_SHA256=''

fail() {
  printf 'sounio-loom-kernel-peer-dumpable-prlimit-falsification-v12-host-gate: FAIL: %s\n' "$*" >&2
  exit 70
}
usage() {
  printf 'usage: %s --base-initramfs PATH --base-initramfs-sha256 HEX --packer PATH --packer-sha256 HEX --kernel-sha256 HEX\n' "$0" >&2
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
  fail "receipt omitted field: $key"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-initramfs) [[ $# -ge 2 ]] || usage; BASE_INITRAMFS="$2"; shift 2 ;;
    --base-initramfs-sha256) [[ $# -ge 2 ]] || usage; BASE_INITRAMFS_SHA256="$2"; shift 2 ;;
    --packer) [[ $# -ge 2 ]] || usage; PACKER="$2"; shift 2 ;;
    --packer-sha256) [[ $# -ge 2 ]] || usage; PACKER_SHA256="$2"; shift 2 ;;
    --kernel-sha256) [[ $# -ge 2 ]] || usage; KERNEL_SHA256="$2"; shift 2 ;;
    *) usage ;;
  esac
done

for path in "$BASE_INITRAMFS" "$PACKER"; do
  [[ "$path" == /var/tmp/loom-kernel-peer-dumpable-prlimit-v12-* ]] ||
    fail "unsafe input path: $path"
  [[ -f "$path" && ! -L "$path" ]] || fail "input is absent or linked: $path"
done
for digest in "$BASE_INITRAMFS_SHA256" "$PACKER_SHA256" "$KERNEL_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail 'input hash is malformed'
done
[[ "$(sha256sum "$BASE_INITRAMFS" | cut -d ' ' -f 1)" == "$BASE_INITRAMFS_SHA256" ]] ||
  fail 'base initramfs hash drifted'
[[ "$(sha256sum "$PACKER" | cut -d ' ' -f 1)" == "$PACKER_SHA256" ]] ||
  fail 'packer hash drifted'
[[ "$(id -u)" == 0 ]] || fail 'KVM gate requires root guardian'
for tool in gzip qemu-system-x86_64 sha256sum timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required host tool is absent: $tool"
done
[[ -c /dev/kvm ]] || fail '/dev/kvm is absent'
KERNEL="/boot/vmlinuz-$(uname -r)"
[[ -f "$KERNEL" && ! -L "$KERNEL" && -r "$KERNEL" ]] ||
  fail 'kernel image is unavailable'
[[ "$(sha256sum "$KERNEL" | cut -d ' ' -f 1)" == "$KERNEL_SHA256" ]] ||
  fail 'kernel hash drifted'

WORK="$(mktemp -d /var/tmp/loom-kernel-peer-dumpable-prlimit-v12-host-gate.XXXXXX)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
gzip -cd "$BASE_INITRAMFS" > "$WORK/base.cpio"
"$PACKER" --extract "$WORK/base.cpio" "$WORK/tree"
[[ -x "$WORK/tree/init" && ! -L "$WORK/tree/init" ]] || fail 'init is absent or linked'
init_selftest="$("$WORK/tree/init" --selftest)"
for fact in 'LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_INIT_V12_SELFTEST PASS' operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT semantic_authority=Sounio; do
  [[ " $init_selftest " == *" $fact "* ]] || fail "init selftest omitted $fact"
done
INIT_SHA256="$(sha256sum "$WORK/tree/init" | cut -d ' ' -f 1)"
HOST_BOOT_ID="$(tr -d '\n' < /proc/sys/kernel/random/boot_id)"

set +e
qemu_output="$(timeout --signal=TERM --kill-after=5s 90s \
  qemu-system-x86_64 \
    -name loom-kernel-peer-dumpable-prlimit-v12 \
    -machine q35,accel=kvm \
    -cpu host \
    -m 512M \
    -smp 2 \
    -nodefaults \
    -no-user-config \
    -display none \
    -serial stdio \
    -monitor none \
    -no-reboot \
    -nic none \
    -kernel "$KERNEL" \
    -initrd "$BASE_INITRAMFS" \
    -append 'rdinit=/init console=ttyS0 quiet loglevel=3 panic=-1 lsm=lockdown,capability,bpf,ima,evm' \
    2>&1)"
qemu_status=$?
set -e
[[ $qemu_status -eq 0 ]] || fail "KVM falsification failed status=$qemu_status output=$qemu_output"
[[ "$(printf '%s' "$qemu_output" | wc -c)" -le 262144 ]] ||
  fail 'KVM output crossed 256 KiB bound'
normalized_output="$(printf '%s\n' "$qemu_output" | tr -d '\r')"
counterexample="$(printf '%s\n' "$normalized_output" | grep '^COUNTEREXAMPLE vertex=DUMPABLE_ONLY_CONTROL ' || true)"
[[ -n "$counterexample" && "$(printf '%s\n' "$counterexample" | wc -l)" == 1 ]] ||
  fail "counterexample observation is absent or duplicated: $qemu_output"
for fact in operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT same_four_uids=true same_user_namespace=true distinct_processes=true distinct_pidfds=true distinct_start_ticks=true distinct_cgroups=true target_dumpable=0 attacker_seccomp=0 mediator=absent principal_capability=CAP_SYS_NICE_ONLY target_limit_restored=true all_epoch_objects_extinct=true python_executed=false rust_executed=false; do
  [[ " $counterexample " == *" $fact "* ]] || fail "counterexample omitted $fact"
done
MATERIAL_OBSERVED="$(field "$counterexample" material_observed)"
case "$MATERIAL_OBSERVED" in
  EFFECT_COMPLETED)
    [[ " $counterexample " == *' completion=LIMIT_CHANGED_RESTORED '* &&
       " $counterexample " == *' errno=NONE '* ]] ||
      fail 'completion observation omitted its typed witness'
    V12_HYPOTHESIS_FALSIFIED=true
    ;;
  REFUSED_BEFORE_EFFECT)
    [[ " $counterexample " == *' completion=TARGET_STATE_UNCHANGED '* &&
       ( " $counterexample " == *' errno=EACCES '* ||
         " $counterexample " == *' errno=EPERM '* ) ]] ||
      fail 'refusal observation omitted its typed witness'
    V12_HYPOTHESIS_FALSIFIED=false
    ;;
  *) fail "inadmissible material observation: $MATERIAL_OBSERVED" ;;
esac
for hash_field in invariant_sha256 delta_sha256 attempt_sha256 target_sha256 extinction_sha256; do
  [[ "$(field "$counterexample" "$hash_field")" =~ ^[0-9a-f]{64}$ ]] ||
    fail "counterexample carried malformed $hash_field"
done

summary="$(printf '%s\n' "$normalized_output" | grep '^LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_V12_BOOT PASS ' || true)"
[[ -n "$summary" && "$(printf '%s\n' "$summary" | wc -l)" == 1 ]] ||
  fail "falsification summary is absent or duplicated: $qemu_output"
GUEST_BOOT_ID="$(field "$summary" boot_id)"
ACTIVE_LSM="$(field "$summary" active_lsm)"
[[ "$ACTIVE_LSM" == 'lockdown,capability,bpf,ima,evm' ]] ||
  fail "causal LSM stack drifted: $ACTIVE_LSM"
[[ "$GUEST_BOOT_ID" =~ ^[0-9a-f-]{36}$ && "$GUEST_BOOT_ID" != "$HOST_BOOT_ID" ]] ||
  fail 'guest boot identity aliases host'
for fact in frozen_expected=REFUSED_BEFORE_EFFECT "material_observed=$MATERIAL_OBSERVED" "v12_hypothesis_falsified=$V12_HYPOTHESIS_FALSIFIED" same_four_uids=true target_dumpable=0 attacker_seccomp=0 mediator=absent principal_capability=CAP_SYS_NICE_ONLY all_epoch_objects_extinct=true guest_root_traversable=true guest_disk=none guest_network=none semantic_authority=Sounio action=9025 controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false next_stage=SOUNIO_V13_GARDEN; do
  [[ " $summary " == *" $fact "* ]] || fail "summary omitted $fact"
done
if pgrep -f 'qemu-system-x86_64.*loom-kernel-peer-dumpable-prlimit-v12' >/dev/null 2>&1; then
  fail 'KVM falsification process did not become extinct'
fi

printf '%s\n' "$qemu_output"
printf 'sounio-loom-kernel-peer-dumpable-prlimit-falsification-v12-host-gate: HOST_MEASUREMENT_PASS host=%s kernel=%s kernel_sha256=%s hypervisor=KVM qemu_version=11.0.0 guest_boot_id=%s host_boot_id=%s guest_distinct=true active_lsm=%s operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT material_observed=%s v12_hypothesis_falsified=%s counterexamples=%s same_four_uids=true target_dumpable=0 attacker_seccomp=0 mediator=absent principal_capability=CAP_SYS_NICE_ONLY typed_witness=true all_epoch_objects_extinct=true guest_root_traversable=true guest_disk=none guest_network=none qemu_extinct=true base_initramfs_sha256=%s init_sha256=%s packer_sha256=%s semantic_authority=Sounio action=9025 controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false next_stage=SOUNIO_V13_GARDEN\n' \
  "$(hostname)" "$(uname -r)" "$KERNEL_SHA256" "$GUEST_BOOT_ID" "$HOST_BOOT_ID" \
  "$ACTIVE_LSM" "$MATERIAL_OBSERVED" "$V12_HYPOTHESIS_FALSIFIED" \
  "$([[ "$V12_HYPOTHESIS_FALSIFIED" == true ]] && printf 1 || printf 0)" \
  "$BASE_INITRAMFS_SHA256" "$INIT_SHA256" "$PACKER_SHA256"
