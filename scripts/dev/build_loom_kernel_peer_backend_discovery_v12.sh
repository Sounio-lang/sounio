#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_KERNEL_PEER_BACKEND_DISCOVERY_V12_CXX:-c++}"
SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_peer_backend_discovery_v12.cpp"
SEMANTIC_MANIFEST="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12.freeze.v1"
OUTPUT="${SOUNIO_LOOM_KERNEL_PEER_BACKEND_DISCOVERY_V12_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-kernel-peer-backend-discovery-v12}"

fail() {
  printf 'build-loom-kernel-peer-backend-discovery-v12: FAIL: %s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail 'V12 backend discovery source is absent or linked'
[[ -f "$SEMANTIC_MANIFEST" && ! -L "$SEMANTIC_MANIFEST" ]] || fail 'V12 semantic freeze is absent or linked'
[[ "$(sha256sum "$SEMANTIC_MANIFEST" | cut -d ' ' -f 1)" == daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30 ]] ||
  fail 'V12 semantic freeze drifted'
grep -Fxq 'stage=SEMANTICS_FROZEN_V12' "$SEMANTIC_MANIFEST" || fail 'V12 semantics are not frozen'
grep -Fxq 'backend_discovery=false' "$SEMANTIC_MANIFEST" || fail 'V12 prematerial boundary drifted'

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp "${TMPDIR:-/tmp}/loom-kernel-peer-backend-discovery-v12.XXXXXX")"
trap 'rm -f "$stage"' EXIT
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches -frandom-seed=loom-kernel-peer-backend-discovery-v12 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none "$SOURCE" -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_KERNEL_PEER_BACKEND_DISCOVERY_V12 path=%s language=C++20 role=MATERIAL_DISCOVERY transitory=true semantic_authority=Sounio action=9025 semantic_manifest_sha256=daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30 material_peer_matrix=false same_uid_peer_isolation=false\n' \
  "$OUTPUT"
