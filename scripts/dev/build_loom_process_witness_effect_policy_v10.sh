#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_SOURCE:-$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v10.cpp}"
SHARED_SOURCE="$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v3.cpp"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v10.freeze.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-process-witness-effect-policy-v10}"

fail() {
  printf 'build-loom-process-witness-effect-policy-v10: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" sha256sum readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is missing: $tool"
done
for path in "$SOURCE" "$SHARED_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "source is absent or linked: $path"
done
[[ -f "$POLICY_MANIFEST" && ! -L "$POLICY_MANIFEST" ]] ||
  fail 'frozen Sounio V10 policy manifest is absent or linked'
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  9e7f42fd4bd18fd2b5f996b279a67f46a50546a20ef6949e4dc069c16b3d0dda ]] ||
  fail 'frozen Sounio V10 policy manifest drifted'
grep -Fxq 'stage=SEMANTICS_FROZEN' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 policy is not frozen'
grep -Fxq 'v10_required_for_native=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 native boundary drifted'
grep -Fxq 'systemd_sys_mount_path=/sys' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 sys mountpoint correction drifted'
grep -Fxq 'systemd_var_tmp_path=/var/tmp' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 var-tmp correction drifted'
grep -Fxq 'principal_observer_enumeration=forbidden' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 principal-opacity contract drifted'
grep -Fxq 'empty_observer=ROOT_HOST' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 root-observer contract drifted'
grep -Fxq 'effect_cell_max_bytes=16777216' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 effect-cell bound drifted'
grep -Fxq 'payload_max_bytes=1048576' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 payload bound drifted'
grep -Fxq 'policy_manifest_max_bytes=65536' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 manifest bound drifted'
grep -Fxq 'effective_dynamic_user=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 DynamicUser contract drifted'
grep -Fxq 'effective_private_tmp=disconnected' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 PrivateTmp contract drifted'
grep -Fxq 'effective_protect_system=strict' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 ProtectSystem contract drifted'
grep -Fxq 'effective_protect_home=read-only' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 ProtectHome contract drifted'
grep -Fxq 'filesystem_authority=ROOT_HOST_MOUNTINFO' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 filesystem authority drifted'
grep -Fxq 'proc_treatment=CAPSULE_EMPTY_BIND' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 typed /proc treatment drifted'
grep -Fxq 'proc_mount_identity=device+inode' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 /proc object identity drifted'
grep -Fxq 'bootstrap_live_procfs_code=453' "$POLICY_MANIFEST" ||
  fail 'Sounio V10 live-procfs control drifted'

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp "${TMPDIR:-/tmp}/loom-process-witness-effect-policy-v10.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches \
  -frandom-seed=loom-process-witness-effect-policy-v10 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -lcrypto -o "$stage"
if readelf -l "$stage" | grep -q 'INTERP'; then
  fail 'native V10 cell retained a dynamic interpreter'
fi
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10 path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 policy_v10_bound=true static=true object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB effective_mount_truth=DynamicUser+disconnected+strict+read-only identity_typed_mounts=CAPSULE_EMPTY_BIND filesystem_authority=ROOT_HOST_MOUNTINFO systemd_mount=/run/systemd/incoming principal_readable=false principal_enumeration=forbidden empty_observer=ROOT_HOST systemd_sys_mount=/sys systemd_var_tmp=/var/tmp material_coverage=false complete_effects=false material_execution=false launch_open=false\n' \
  "$OUTPUT"
