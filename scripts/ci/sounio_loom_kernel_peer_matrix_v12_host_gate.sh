#!/usr/bin/env bash

set -euo pipefail
umask 077

BASE_INITRAMFS=''
BASE_INITRAMFS_SHA256=''
PACKER=''
PACKER_SHA256=''
LOADER_SOURCE=''
LOADER_SOURCE_SHA256=''
BPF_OBJECT_SHA256=''
KERNEL_SHA256=''

fail() {
  printf 'sounio-loom-kernel-peer-matrix-v12-host-gate: FAIL: %s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --base-initramfs PATH --base-initramfs-sha256 HEX --packer PATH --packer-sha256 HEX --loader-source PATH --loader-source-sha256 HEX --bpf-object-sha256 HEX --kernel-sha256 HEX\n' "$0" >&2
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
  fail "matrix receipt omitted field: $key"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-initramfs) [[ $# -ge 2 ]] || usage; BASE_INITRAMFS="$2"; shift 2 ;;
    --base-initramfs-sha256) [[ $# -ge 2 ]] || usage; BASE_INITRAMFS_SHA256="$2"; shift 2 ;;
    --packer) [[ $# -ge 2 ]] || usage; PACKER="$2"; shift 2 ;;
    --packer-sha256) [[ $# -ge 2 ]] || usage; PACKER_SHA256="$2"; shift 2 ;;
    --loader-source) [[ $# -ge 2 ]] || usage; LOADER_SOURCE="$2"; shift 2 ;;
    --loader-source-sha256) [[ $# -ge 2 ]] || usage; LOADER_SOURCE_SHA256="$2"; shift 2 ;;
    --bpf-object-sha256) [[ $# -ge 2 ]] || usage; BPF_OBJECT_SHA256="$2"; shift 2 ;;
    --kernel-sha256) [[ $# -ge 2 ]] || usage; KERNEL_SHA256="$2"; shift 2 ;;
    *) usage ;;
  esac
done

for path in "$BASE_INITRAMFS" "$PACKER" "$LOADER_SOURCE"; do
  [[ "$path" == /var/tmp/loom-kernel-peer-matrix-v12-* ]] || fail "unsafe input path: $path"
  [[ -f "$path" && ! -L "$path" ]] || fail "input is absent or linked: $path"
done
for digest in "$BASE_INITRAMFS_SHA256" "$PACKER_SHA256" "$LOADER_SOURCE_SHA256" \
  "$BPF_OBJECT_SHA256" "$KERNEL_SHA256"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail 'input hash is malformed'
done
[[ "$(sha256sum "$BASE_INITRAMFS" | cut -d ' ' -f 1)" == "$BASE_INITRAMFS_SHA256" ]] ||
  fail 'base initramfs hash drifted'
[[ "$(sha256sum "$PACKER" | cut -d ' ' -f 1)" == "$PACKER_SHA256" ]] || fail 'packer hash drifted'
[[ "$(sha256sum "$LOADER_SOURCE" | cut -d ' ' -f 1)" == "$LOADER_SOURCE_SHA256" ]] ||
  fail 'loader source hash drifted'
[[ "$(id -u)" == 0 ]] || fail 'KVM gate requires root guardian'
for tool in c++ gzip ldd qemu-system-x86_64 sha256sum timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required host tool is absent: $tool"
done
[[ -c /dev/kvm ]] || fail '/dev/kvm is absent'
KERNEL="/boot/vmlinuz-$(uname -r)"
[[ -f "$KERNEL" && ! -L "$KERNEL" && -r "$KERNEL" ]] || fail 'kernel image is unavailable'
[[ "$(sha256sum "$KERNEL" | cut -d ' ' -f 1)" == "$KERNEL_SHA256" ]] || fail 'kernel hash drifted'

LIBDIR=/lib/x86_64-linux-gnu
INTERPRETER=/lib64/ld-linux-x86-64.so.2
for library in libbpf.so.1 libelf.so.1 libz.so.1 libzstd.so.1 libc.so.6 \
  libstdc++.so.6 libgcc_s.so.1 libm.so.6; do
  [[ -f "$LIBDIR/$library" && -r "$LIBDIR/$library" ]] || fail "guest dependency is absent: $library"
done
[[ -f "$INTERPRETER" && -r "$INTERPRETER" ]] || fail 'ELF interpreter is absent'

WORK="$(mktemp -d /var/tmp/loom-kernel-peer-matrix-v12-host-gate.XXXXXX)"
cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT
TREE="$WORK/tree"
RAW_BASE="$WORK/base.cpio"
RAW_FINAL="$WORK/final.cpio"
FINAL_INITRAMFS="$WORK/final.cpio.gz"
LOADER="$WORK/loom-bpf-lsm-loader-v12"
LOADER_TWIN="$WORK/loom-bpf-lsm-loader-v12-twin"
NORMALIZED_LOADER_SOURCE="$WORK/loom_bpf_lsm_loader_v12.cpp"
install -m 0400 "$LOADER_SOURCE" "$NORMALIZED_LOADER_SOURCE"

compile_loader() {
  local output="$1"
  SOURCE_DATE_EPOCH=0 c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
    -fno-record-gcc-switches -frandom-seed=loom-bpf-lsm-loader-v12 \
    -ffile-prefix-map="$WORK"=. -fdebug-prefix-map="$WORK"=. \
    -fmacro-prefix-map="$WORK"=. -Wl,--build-id=none \
    "$NORMALIZED_LOADER_SOURCE" -o "$output" \
    "$LIBDIR/libbpf.so.1" "$LIBDIR/libelf.so.1" "$LIBDIR/libz.so.1" \
    "$LIBDIR/libzstd.so.1"
}
compile_loader "$LOADER"
compile_loader "$LOADER_TWIN"
LOADER_SHA256="$(sha256sum "$LOADER" | cut -d ' ' -f 1)"
[[ "$LOADER_SHA256" == "$(sha256sum "$LOADER_TWIN" | cut -d ' ' -f 1)" ]] ||
  fail 'loader reproducibility twin diverged'
"$LOADER" --selftest >/dev/null || fail 'source-built loader selftest refused'
loader_ldd="$(ldd "$LOADER")"
[[ "$loader_ldd" != *'not found'* ]] || fail "loader dependency unresolved: $loader_ldd"
for dependency in libbpf.so.1 libelf.so.1 libz.so.1 libzstd.so.1 libc.so.6 \
  libstdc++.so.6 libgcc_s.so.1 libm.so.6; do
  [[ "$loader_ldd" == *"$dependency"* ]] || fail "loader dependency set omitted $dependency"
done

gzip -cd "$BASE_INITRAMFS" > "$RAW_BASE"
"$PACKER" --extract "$RAW_BASE" "$TREE"
[[ "$(sha256sum "$TREE/loom/policy.bpf.o" | cut -d ' ' -f 1)" == "$BPF_OBJECT_SHA256" ]] ||
  fail 'BPF object hash drifted after extraction'
install -m 0755 "$LOADER" "$TREE/loom/loom-bpf-lsm-loader-v12"
for library in libbpf.so.1 libelf.so.1 libz.so.1 libzstd.so.1 libc.so.6 \
  libstdc++.so.6 libgcc_s.so.1 libm.so.6; do
  install -m 0555 -D "$LIBDIR/$library" "$TREE$LIBDIR/$library"
done
install -m 0555 -D "$INTERPRETER" "$TREE$INTERPRETER"
"$PACKER" --create "$TREE" "$RAW_FINAL"
gzip -n -9 < "$RAW_FINAL" > "$FINAL_INITRAMFS"
FINAL_INITRAMFS_SHA256="$(sha256sum "$FINAL_INITRAMFS" | cut -d ' ' -f 1)"
HOST_BOOT_ID="$(tr -d '\n' < /proc/sys/kernel/random/boot_id)"

set +e
qemu_output="$(timeout --signal=TERM --kill-after=5s 150s \
  qemu-system-x86_64 \
    -name loom-kernel-peer-matrix-v12 \
    -machine q35,accel=kvm \
    -cpu host \
    -m 1024M \
    -smp 2 \
    -nodefaults \
    -no-user-config \
    -display none \
    -serial stdio \
    -monitor none \
    -no-reboot \
    -nic none \
    -kernel "$KERNEL" \
    -initrd "$FINAL_INITRAMFS" \
    -append 'rdinit=/init console=ttyS0 quiet loglevel=3 panic=-1 lsm=lockdown,capability,bpf,ima,evm' \
    2>&1)"
qemu_status=$?
set -e
[[ $qemu_status -eq 0 ]] || fail "KVM peer matrix failed status=$qemu_status output=$qemu_output"
[[ "$(printf '%s' "$qemu_output" | wc -c)" -le 2097152 ]] || fail 'KVM output crossed 2 MiB bound'
normalized_output="$(printf '%s\n' "$qemu_output" | tr -d '\r')"
[[ "$(printf '%s\n' "$normalized_output" | grep -c '^PAIR operation=')" == 10 ]] ||
  fail "peer matrix did not emit exactly ten pairs: $qemu_output"
expected_syscalls=(kill_SIGTERM tgkill_SIGTERM rt_sigqueueinfo pidfd_send_signal ptrace_ATTACH process_vm_readv open_read_proc_pid_mem pidfd_getfd prlimit64 process_madvise)
expected_completions=(TARGET_TERMINATED TARGET_THREAD_TERMINATED SIGNAL_PAYLOAD_OBSERVED TARGET_TERMINATED PTRACE_ATTACH_DETACH CANARY_BYTES_READ PROC_MEM_CANARY_READ TARGET_FD_DUPLICATED LIMIT_CHANGED_RESTORED MADVISE_COMPLETED_4096_BYTES)
for index in {1..10}; do
  pair="$(printf '%s\n' "$normalized_output" | grep "^PAIR operation=$index " || true)"
  [[ -n "$pair" && "$(printf '%s\n' "$pair" | wc -l)" == 1 ]] || fail "pair $index absent or duplicated"
  offset=$((index - 1))
  [[ "$(field "$pair" syscall)" == "${expected_syscalls[$offset]}" ]] ||
    fail "pair $index syscall drifted"
  [[ "$(field "$pair" completion)" == "${expected_completions[$offset]}" ]] ||
    fail "pair $index completion witness drifted"
  for fact in treatment=REFUSED_BEFORE_EFFECT sabotage=EFFECT_COMPLETED same_four_uids=true attacker_seccomp=0 distinct_cgroups=true same_process_epoch=true only_delta=mediator_presence+policy_hash mediator_links_extinct=true mediator_programs_extinct=true mediator_quiescence_ms=250 ptracer_aperture=NOT_REQUIRED competing_ptrace_lsms=absent guest_root_traversable=true principal_capability=CAP_SYS_NICE_ONLY; do
    [[ " $pair " == *" $fact "* ]] || fail "pair $index omitted $fact"
  done
  [[ " $pair " == *' treatment_errno=EACCES '* || " $pair " == *' treatment_errno=EPERM '* ]] ||
    fail "pair $index carried an inadmissible treatment errno"
  for hash_field in invariant_sha256 treatment_delta_sha256 sabotage_delta_sha256 treatment_attempt_sha256 treatment_target_sha256 sabotage_attempt_sha256 sabotage_target_sha256 extinction_sha256; do
    [[ "$(field "$pair" "$hash_field")" =~ ^[0-9a-f]{64}$ ]] ||
      fail "pair $index carried malformed $hash_field"
  done
done
summary="$(printf '%s\n' "$normalized_output" | grep '^LOOM_KERNEL_PEER_MATRIX_V12_BOOT PASS ' || true)"
[[ -n "$summary" && "$(printf '%s\n' "$summary" | wc -l)" == 1 ]] ||
  fail "peer matrix summary is absent or duplicated: $qemu_output"
GUEST_BOOT_ID="$(field "$summary" boot_id)"
ACTIVE_LSM="$(field "$summary" active_lsm)"
PAIR_SET_SHA256="$(field "$summary" pair_set_sha256)"
[[ "$ACTIVE_LSM" == 'lockdown,capability,bpf,ima,evm' ]] ||
  fail "causal LSM stack drifted: $ACTIVE_LSM"
[[ "$GUEST_BOOT_ID" =~ ^[0-9a-f-]{36}$ && "$GUEST_BOOT_ID" != "$HOST_BOOT_ID" ]] ||
  fail 'guest boot identity aliases host'
[[ "$PAIR_SET_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail 'pair-set hash is malformed'
for fact in decisive_pairs=10 treatment_refused=10 mediator_removed_completed=10 same_kuid_pair_observed=true all_four_kernel_uid_slots_equal=true attacker_syscalls_open=true receiver_mediator_active=true only_delta_mediator=true competing_ptrace_lsms=absent all_epoch_objects_extinct=true guest_root_traversable=true guest_disk=none guest_network=none semantic_authority=Sounio action=9025 controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
  [[ " $summary " == *" $fact "* ]] || fail "summary omitted $fact"
done
if pgrep -f 'qemu-system-x86_64.*loom-kernel-peer-matrix-v12' >/dev/null 2>&1; then
  fail 'KVM peer matrix process did not become extinct'
fi

printf '%s\n' "$qemu_output"
printf 'sounio-loom-kernel-peer-matrix-v12-host-gate: HOST_MEASUREMENT_PASS host=%s kernel=%s kernel_sha256=%s hypervisor=KVM qemu_version=11.0.0 guest_boot_id=%s host_boot_id=%s guest_distinct=true active_lsm=%s operations=10 decisive_pairs=10 treatment_refused=10 mediator_removed_completed=10 same_kuid_pair_observed=true all_four_kernel_uid_slots_equal=true attacker_syscalls_open=true receiver_mediator_active=true only_delta_mediator=true competing_ptrace_lsms=absent all_epoch_objects_extinct=true pair_set_sha256=%s loader_reproducibility_twin=true guest_root_traversable=true guest_disk=none guest_network=none qemu_extinct=true base_initramfs_sha256=%s final_initramfs_sha256=%s bpf_object_sha256=%s loader_source_sha256=%s loader_sha256=%s packer_sha256=%s semantic_authority=Sounio action=9025 controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false\n' \
  "$(hostname)" "$(uname -r)" "$KERNEL_SHA256" "$GUEST_BOOT_ID" "$HOST_BOOT_ID" \
  "$ACTIVE_LSM" "$PAIR_SET_SHA256" "$BASE_INITRAMFS_SHA256" \
  "$FINAL_INITRAMFS_SHA256" "$BPF_OBJECT_SHA256" "$LOADER_SOURCE_SHA256" \
  "$LOADER_SHA256" "$PACKER_SHA256"
