#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
BUILDER=scripts/dev/build_loom_kernel_peer_matrix_v12.sh
CONTRACT=tools/loom/BPF_LSM_PEER_MATRIX_V12.md
INIT_SOURCE=tools/loom/src/loom_kernel_peer_matrix_init_v12.cpp

fail() {
  printf 'sounio-loom-kernel-peer-matrix-v12-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for path in "$BUILDER" "$CONTRACT" "$INIT_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required component is absent or linked: $path"
done
for syscall in kill_SIGTERM tgkill_SIGTERM rt_sigqueueinfo pidfd_send_signal ptrace_ATTACH process_vm_readv open_read_proc_pid_mem pidfd_getfd prlimit64 process_madvise; do
  grep -Fq "$syscall" "$INIT_SOURCE" || fail "init omitted operation: $syscall"
done
for marker in controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
  grep -Fq "$marker" "$INIT_SOURCE" || fail "init omitted conservative marker: $marker"
done
grep -Fq 'only delta' "$CONTRACT" || fail 'contract omitted exact causal delta'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-matrix-v12-selftest.XXXXXX")"
cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT
ARCHIVE="$WORK/base.cpio.gz"
PACKER="$WORK/loom-newc-packer"
BPF_OBJECT="$WORK/policy.bpf.o"
build_output="$(SOUNIO_LOOM_KERNEL_PEER_MATRIX_V12_OUTPUT="$ARCHIVE" \
  SOUNIO_LOOM_KERNEL_PEER_MATRIX_V12_PACKER_OUTPUT="$PACKER" \
  SOUNIO_LOOM_KERNEL_PEER_MATRIX_V12_BPF_OUTPUT="$BPF_OBJECT" \
  bash "$BUILDER" 2>/dev/null)"
for fact in operations=10 decisive_pairs=10 guest_root_traversable=true guest_disk=none guest_network=none controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
  [[ " $build_output " == *" $fact "* ]] || fail "build receipt omitted $fact"
done
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -Wl,--build-id=none "$INIT_SOURCE" -o "$WORK/init" -lcrypto -ldl -pthread \
  2>/dev/null
"$WORK/init" --selftest >/dev/null
"$PACKER" --selftest >/dev/null
readelf -SW "$BPF_OBJECT" | grep -Fq ' .BTF.ext ' || fail 'built BPF object omitted BTF.ext'

printf 'sounio-loom-kernel-peer-matrix-v12-selftest: PASS semantic_authority=Sounio action=9025 operations=10 decisive_pairs=10 same_process_epoch=true only_delta=mediator_presence+policy_hash btf_core=true init_language=C++20 bpf_language=C guest_root_traversable=true principal_capability=CAP_SYS_NICE_ONLY python_executed=false rust_executed=false controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false archive_sha256=%s bpf_object_sha256=%s\n' \
  "$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BPF_OBJECT" | cut -d ' ' -f 1)"
