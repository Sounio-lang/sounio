#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOURCE="$ROOT_DIR/tools/loom/src/loom_mount_namespace_mutator.cpp"
BUILDER="$ROOT_DIR/scripts/dev/build_loom_mount_namespace_mutator.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-mount-namespace-mutator.XXXXXX")"
ONE="$WORK/one"
TWO="$WORK/two"

cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-mount-namespace-mutator-selftest: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for path in "$SOURCE" "$BUILDER"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "required input is absent or linked: $path"
done
SOUNIO_LOOM_MOUNT_NAMESPACE_MUTATOR_OUTPUT="$ONE" "$BUILDER" >/dev/null
SOUNIO_LOOM_MOUNT_NAMESPACE_MUTATOR_OUTPUT="$TWO" "$BUILDER" >/dev/null
cmp "$ONE" "$TWO" || fail 'two namespace-mutator builds differ'
[[ "$(stat -c '%a' "$ONE")" == 755 && ! -u "$ONE" && ! -g "$ONE" ]] ||
  fail 'namespace-mutator mode drifted'
readelf -l "$ONE" | grep -q 'INTERP' && fail 'namespace mutator is dynamically linked'

selftest="$($ONE --selftest)"
[[ "$selftest" == \
  'LOOM_MOUNT_NAMESPACE_MUTATOR_SELFTEST PASS language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 operations=live-procfs+writable-proc-bind descriptor_bound_namespace=true descriptor_bound_root=true semantic_decision=false' ]] ||
  fail "native selftest receipt drifted: $selftest"

set +e
"$ONE" --pid 1 --operation live-procfs >/dev/null 2>&1
invalid_status=$?
set -e
[[ $invalid_status -eq 64 ]] || fail "invalid PID did not refuse with 64: $invalid_status"

printf 'sounio-loom-mount-namespace-mutator-selftest: PASS producer=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 source_sha256=%s binary_sha256=%s deterministic=true static=true set_id=false operations=2 semantic_decision=false\n' \
  "$(sha256sum "$SOURCE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ONE" | cut -d ' ' -f 1)"
