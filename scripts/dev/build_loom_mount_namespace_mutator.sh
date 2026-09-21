#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_MOUNT_NAMESPACE_MUTATOR_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_MOUNT_NAMESPACE_MUTATOR_SOURCE:-$ROOT_DIR/tools/loom/src/loom_mount_namespace_mutator.cpp}"
OUTPUT="${SOUNIO_LOOM_MOUNT_NAMESPACE_MUTATOR_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-mount-namespace-mutator}"

fail() {
  printf 'build-loom-mount-namespace-mutator: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is missing: $tool"
done
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail "source is absent or linked: $SOURCE"
mkdir -p "$(dirname "$OUTPUT")"

stage="$(mktemp "${TMPDIR:-/tmp}/loom-mount-namespace-mutator.XXXXXX")"
trap 'rm -f "$stage"' EXIT
SOURCE_DATE_EPOCH=0 "$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -static -fno-record-gcc-switches \
  -frandom-seed=loom-mount-namespace-mutator-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -o "$stage"
if readelf -l "$stage" | grep -q 'INTERP'; then
  fail 'namespace mutator retained a dynamic interpreter'
fi
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_MOUNT_NAMESPACE_MUTATOR path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 static=true operations=live-procfs+writable-proc-bind semantic_decision=false\n' \
  "$OUTPUT"
