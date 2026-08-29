#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_KERNEL_PEER_MATRIX_V12_CXX:-c++}"
CLANG="${SOUNIO_LOOM_KERNEL_PEER_MATRIX_V12_CLANG:-clang}"
INIT_SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_matrix_init_v12.cpp"
LOADER_SOURCE="$ROOT_DIR/tools/loom/src/loom_bpf_lsm_loader_v12.cpp"
BPF_SOURCE="$ROOT_DIR/tools/loom/bpf/loom_kernel_peer_v12.bpf.c"
PACKER_SOURCE="$ROOT_DIR/tools/loom/src/loom_newc_packer.cpp"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12.freeze.v1"
LOAD_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_bpf_load_v12.freeze.v1"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_MATRIX_V12_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-kernel-peer-matrix-v12.base.cpio.gz}"
PACKER_OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_MATRIX_V12_PACKER_OUTPUT:-}"
BPF_OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_MATRIX_V12_BPF_OUTPUT:-}"

fail() {
  printf 'build-loom-kernel-peer-matrix-v12: FAIL: %s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" "$CLANG" gzip readelf sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is absent: $tool"
done
for source in "$INIT_SOURCE" "$LOADER_SOURCE" "$BPF_SOURCE" "$PACKER_SOURCE"; do
  [[ -f "$source" && ! -L "$source" ]] || fail "source is absent or linked: $source"
done
[[ "$(sha256sum "$SEMANTIC_MANIFEST" | cut -d ' ' -f 1)" == daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30 ]] ||
  fail 'V12 Sounio semantic freeze drifted'
[[ "$(sha256sum "$LOAD_MANIFEST" | cut -d ' ' -f 1)" == 6109ff02d1078ca3f0f21dcb98c189db12f59950899e375cac77c4ec7d4bfe75 ]] ||
  fail 'V12 BPF mediator load freeze drifted'
grep -Fxq 'next_stage=BPF_LSM_PEER_MATRIX' "$LOAD_MANIFEST" ||
  fail 'V12 load freeze does not authorize the peer matrix'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-matrix-v12.XXXXXX")"
TREE="$WORK/tree"
INIT="$TREE/init"
POLICY="$TREE/loom/policy.bpf.o"
PACKER="$WORK/loom-newc-packer"
LOADER_OBJECT="$WORK/loom-bpf-lsm-loader-v12.o"
RAW_ARCHIVE="$WORK/base.cpio"
cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p "$TREE/dev" "$TREE/proc" "$TREE/sys/kernel/security" \
  "$TREE/sys/fs/bpf" "$TREE/sys/fs/cgroup" "$TREE/tmp" "$TREE/loom" \
  "$TREE/lib/x86_64-linux-gnu" "$TREE/lib64"
chmod 0755 "$TREE"
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -static -fno-record-gcc-switches \
  -frandom-seed=loom-kernel-peer-matrix-init-v12 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$INIT_SOURCE" -o "$INIT" -lcrypto -ldl -pthread
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches -frandom-seed=loom-bpf-lsm-loader-v12 \
  -ffile-prefix-map="$ROOT_DIR"=. -c "$LOADER_SOURCE" -o "$LOADER_OBJECT"
SOURCE_DATE_EPOCH=0 "$CLANG" -target bpf -D__TARGET_ARCH_x86 -O2 -g \
  -Wall -Wextra -Werror -ffile-prefix-map="$ROOT_DIR"=. \
  -c "$BPF_SOURCE" -o "$POLICY"
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -static -fno-record-gcc-switches -frandom-seed=loom-newc-packer-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$PACKER_SOURCE" -o "$PACKER"
chmod 0755 "$INIT"
chmod 0444 "$POLICY"

section_table="$(readelf -SW "$POLICY")"
for section in 'lsm/task_kill' 'lsm/ptrace_access_check' 'lsm/task_prlimit' '.BTF' '.BTF.ext' 'license'; do
  [[ "$section_table" == *" $section "* ]] || fail "BPF object omitted section: $section"
done
[[ "$(printf '%s\n' "$section_table" | grep -Ec ' lsm/(task_kill|ptrace_access_check|task_prlimit) ')" == 3 ]] ||
  fail 'BPF object did not contain exactly three frozen LSM sections'

mkdir -p "$(dirname "$OUTPUT")"
"$PACKER" --create "$TREE" "$RAW_ARCHIVE"
gzip -n -9 < "$RAW_ARCHIVE" > "$OUTPUT"
if [[ -n "$PACKER_OUTPUT" ]]; then
  mkdir -p "$(dirname "$PACKER_OUTPUT")"
  install -m 0755 "$PACKER" "$PACKER_OUTPUT"
fi
if [[ -n "$BPF_OUTPUT" ]]; then
  mkdir -p "$(dirname "$BPF_OUTPUT")"
  install -m 0444 "$POLICY" "$BPF_OUTPUT"
fi

printf 'BUILT_LOOM_KERNEL_PEER_MATRIX_V12 base_initramfs=%s language=C+BPF+C++20 role=MATERIAL_BOOTSTRAP transitory=true semantic_authority=Sounio action=9025 init_sha256=%s loader_object_sha256=%s bpf_object_sha256=%s packer_sha256=%s base_initramfs_sha256=%s operations=10 decisive_pairs=10 guest_root_traversable=true guest_disk=none guest_network=none controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false\n' \
  "$OUTPUT" "$(sha256sum "$INIT" | cut -d ' ' -f 1)" \
  "$(sha256sum "$LOADER_OBJECT" | cut -d ' ' -f 1)" \
  "$(sha256sum "$POLICY" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PACKER" | cut -d ' ' -f 1)" \
  "$(sha256sum "$OUTPUT" | cut -d ' ' -f 1)"
