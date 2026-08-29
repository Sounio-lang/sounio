#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_KERNEL_PEER_MICROHOST_V12_CXX:-c++}"
SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_microhost_init_v12.cpp"
PACKER_SOURCE="$ROOT_DIR/tools/loom/src/loom_newc_packer.cpp"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12.freeze.v1"
BACKEND_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_backend_discovery_v12.freeze.v1"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_MICROHOST_V12_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-kernel-peer-microhost-v12.cpio.gz}"
PACKER_OUTPUT="${SOUNIO_LOOM_NEWC_PACKER_OUTPUT:-}"

fail() {
  printf 'build-loom-kernel-peer-microhost-v12: FAIL: %s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" gzip sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is absent: $tool"
done
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail 'microhost init source is absent or linked'
[[ -f "$PACKER_SOURCE" && ! -L "$PACKER_SOURCE" ]] || fail 'newc packer source is absent or linked'
[[ "$(sha256sum "$SEMANTIC_MANIFEST" | cut -d ' ' -f 1)" == daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30 ]] ||
  fail 'V12 semantic freeze drifted'
[[ "$(sha256sum "$BACKEND_MANIFEST" | cut -d ' ' -f 1)" == bb695f07b9d752f025be0f101fde27fb42d635cc13ab3a9b34f2241ccab3b8c5 ]] ||
  fail 'V12 negative backend freeze drifted'
grep -Fxq 'next_stage=DEDICATED_BPF_LSM_HOST_REQUIRED' "$BACKEND_MANIFEST" ||
  fail 'V12 backend result does not authorize the dedicated microhost'

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-peer-microhost-v12.XXXXXX")"
TREE="$WORK/tree"
INIT="$TREE/init"
PACKER="$WORK/loom-newc-packer"
RAW_ARCHIVE="$WORK/microhost.cpio"
cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p "$TREE/dev" "$TREE/proc" "$TREE/sys/kernel/security" "$TREE/sys/fs/bpf" \
  "$TREE/tmp" "$TREE/loom"
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches -frandom-seed=loom-kernel-peer-microhost-v12 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none "$SOURCE" -o "$INIT"
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches -frandom-seed=loom-newc-packer-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none "$PACKER_SOURCE" -o "$PACKER"
chmod 0755 "$INIT"
mkdir -p "$(dirname "$OUTPUT")"
"$PACKER" --create "$TREE" "$RAW_ARCHIVE"
gzip -n -9 < "$RAW_ARCHIVE" > "$OUTPUT"
if [[ -n "$PACKER_OUTPUT" ]]; then
  mkdir -p "$(dirname "$PACKER_OUTPUT")"
  install -m 0755 "$PACKER" "$PACKER_OUTPUT"
fi

INIT_SHA256="$(sha256sum "$INIT" | cut -d ' ' -f 1)"
ARCHIVE_SHA256="$(sha256sum "$OUTPUT" | cut -d ' ' -f 1)"
PACKER_SHA256="$(sha256sum "$PACKER" | cut -d ' ' -f 1)"
printf 'BUILT_LOOM_KERNEL_PEER_MICROHOST_V12 path=%s language=C++20 role=MATERIAL_BOOTSTRAP transitory=true semantic_authority=Sounio action=9025 init_sha256=%s packer_sha256=%s archive_sha256=%s guest_disk=none guest_network=none bpf_lsm=required material_peer_matrix=false same_uid_peer_isolation=false claim_ready=false\n' \
  "$OUTPUT" "$INIT_SHA256" "$PACKER_SHA256" "$ARCHIVE_SHA256"
