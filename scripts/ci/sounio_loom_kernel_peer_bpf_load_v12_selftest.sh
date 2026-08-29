#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_kernel_peer_bpf_load_v12.sh"
CONTRACT="$ROOT_DIR/tools/loom/BPF_LSM_PEER_MEDIATOR_LOAD_V12.md"
INIT_SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_bpf_load_init_v12.cpp"
LOADER_SOURCE="$ROOT_DIR/tools/loom/src/loom_bpf_lsm_loader_v12.cpp"
BPF_SOURCE="$ROOT_DIR/tools/loom/bpf/loom_kernel_peer_v12.bpf.c"

fail() {
  printf 'sounio-loom-kernel-peer-bpf-load-v12-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for path in "$BUILDER" "$CONTRACT" "$INIT_SOURCE" "$LOADER_SOURCE" "$BPF_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required component is absent or linked: $path"
done
for marker in \
  'material_peer_matrix=false' \
  'same_uid_peer_isolation=false' \
  'action_9025_decision=DENY451' \
  'claim_ready=false'; do
  grep -Fq "$marker" "$CONTRACT" || fail "contract omitted conservative marker: $marker"
done
grep -Fq 'BPF_LINK_GET_FD_BY_ID' "$INIT_SOURCE" || fail 'init omitted identity-based extinction query'
grep -Fq 'bpf_link__pin' "$LOADER_SOURCE" || fail 'loader omitted persistent link pinning'
[[ "$(grep -Ec '^SEC\("lsm/(task_kill|ptrace_access_check|task_prlimit)"\)$' "$BPF_SOURCE")" == 3 ]] ||
  fail 'BPF source omitted a frozen LSM hook'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-bpf-load-v12-selftest.XXXXXX")"
cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT
ARCHIVE="$WORK/base.cpio.gz"
PACKER="$WORK/loom-newc-packer"
BPF_OBJECT="$WORK/policy.bpf.o"
build_output="$(SOUNIO_LOOM_KERNEL_PEER_BPF_LOAD_V12_OUTPUT="$ARCHIVE" \
  SOUNIO_LOOM_KERNEL_PEER_BPF_LOAD_V12_PACKER_OUTPUT="$PACKER" \
  SOUNIO_LOOM_KERNEL_PEER_BPF_LOAD_V12_BPF_OUTPUT="$BPF_OBJECT" \
  bash "$BUILDER")"
for fact in programs=3 guest_disk=none guest_network=none material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false; do
  [[ " $build_output " == *" $fact "* ]] || fail "build receipt omitted $fact"
done
"$PACKER" --selftest >/dev/null
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -Wl,--build-id=none "$INIT_SOURCE" -o "$WORK/init"
"$WORK/init" --selftest >/dev/null
c++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -c "$LOADER_SOURCE" -o "$WORK/loader.o"
readelf -SW "$BPF_OBJECT" | grep -Fq ' .BTF.ext ' || fail 'built BPF object omitted BTF.ext'

printf 'sounio-loom-kernel-peer-bpf-load-v12-selftest: PASS semantic_authority=Sounio action=9025 programs=3 btf_core=true loader_language=C++20 init_language=C++20 bpf_language=C python_executed=false rust_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false archive_sha256=%s bpf_object_sha256=%s\n' \
  "$(sha256sum "$ARCHIVE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BPF_OBJECT" | cut -d ' ' -f 1)"
