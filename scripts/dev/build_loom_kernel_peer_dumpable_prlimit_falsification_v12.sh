#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_CXX:-c++}"
INIT_SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_dumpable_prlimit_falsification_init_v12.cpp"
BASE_SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_matrix_init_v12.cpp"
PACKER_SOURCE="$ROOT_DIR/tools/loom/src/loom_newc_packer.cpp"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12.freeze.v1"
MATRIX_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_matrix_v12.freeze.v1"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-kernel-peer-dumpable-prlimit-falsification-v12.cpio.gz}"
PACKER_OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_V12_PACKER_OUTPUT:-}"

fail() {
  printf 'build-loom-kernel-peer-dumpable-prlimit-falsification-v12: FAIL: %s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" gzip readelf sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is absent: $tool"
done
for source in "$INIT_SOURCE" "$BASE_SOURCE" "$PACKER_SOURCE"; do
  [[ -f "$source" && ! -L "$source" ]] || fail "source is absent or linked: $source"
done
[[ "$(sha256sum "$SEMANTIC_MANIFEST" | cut -d ' ' -f 1)" == daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30 ]] ||
  fail 'V12 Sounio semantic freeze drifted'
[[ "$(sha256sum "$MATRIX_MANIFEST" | cut -d ' ' -f 1)" == 1692782657cbe6fe7a548b6f11d4d542d24fe05569686d536a4c69af0775cd75 ]] ||
  fail 'V12 peer-matrix freeze drifted'
grep -Fxq 'next_stage=BPF_LSM_PEER_CONTROLS' "$MATRIX_MANIFEST" ||
  fail 'V12 peer-matrix freeze does not authorize controls'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-dumpable-prlimit-v12.XXXXXX")"
TREE="$WORK/tree"
INIT="$TREE/init"
PACKER="$WORK/loom-newc-packer"
RAW_ARCHIVE="$WORK/base.cpio"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

mkdir -p "$TREE/dev" "$TREE/proc" "$TREE/sys/kernel/security" \
  "$TREE/sys/fs/bpf" "$TREE/sys/fs/cgroup" "$TREE/tmp"
chmod 0755 "$TREE"
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -static -fno-record-gcc-switches \
  -frandom-seed=loom-kernel-peer-dumpable-prlimit-falsification-init-v12 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$INIT_SOURCE" -o "$INIT" -lcrypto -ldl -pthread
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -static -fno-record-gcc-switches -frandom-seed=loom-newc-packer-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$PACKER_SOURCE" -o "$PACKER"
chmod 0755 "$INIT"
"$INIT" --selftest >/dev/null

mkdir -p "$(dirname "$OUTPUT")"
"$PACKER" --create "$TREE" "$RAW_ARCHIVE"
gzip -n -9 < "$RAW_ARCHIVE" > "$OUTPUT"
if [[ -n "$PACKER_OUTPUT" ]]; then
  mkdir -p "$(dirname "$PACKER_OUTPUT")"
  install -m 0755 "$PACKER" "$PACKER_OUTPUT"
fi

printf 'BUILT_LOOM_KERNEL_PEER_DUMPABLE_PRLIMIT_FALSIFICATION_V12 base_initramfs=%s language=C++20 role=MATERIAL_BOOTSTRAP transitory=true semantic_authority=Sounio action=9025 init_sha256=%s packer_sha256=%s base_initramfs_sha256=%s operation=9 syscall=prlimit64 frozen_expected=REFUSED_BEFORE_EFFECT observations=1 v12_hypothesis_falsified=unmeasured controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 claim_ready=false\n' \
  "$OUTPUT" "$(sha256sum "$INIT" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PACKER" | cut -d ' ' -f 1)" \
  "$(sha256sum "$OUTPUT" | cut -d ' ' -f 1)"
