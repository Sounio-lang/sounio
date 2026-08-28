#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V9_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V9_SOURCE:-$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v9.cpp}"
SHARED_SOURCE="$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v3.cpp"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v9.freeze.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V9_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-process-witness-effect-policy-v9}"

fail() {
  printf 'build-loom-process-witness-effect-policy-v9: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" sha256sum readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is missing: $tool"
done
for path in "$SOURCE" "$SHARED_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "source is absent or linked: $path"
done
[[ -f "$POLICY_MANIFEST" && ! -L "$POLICY_MANIFEST" ]] ||
  fail 'frozen Sounio V9 policy manifest is absent or linked'
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  9d747d937a6a2316dd8894b37e243180031b8518f2696b9200ee7d1f1d81868c ]] ||
  fail 'frozen Sounio V9 policy manifest drifted'
grep -Fxq 'stage=SEMANTICS_FROZEN' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 policy is not frozen'
grep -Fxq 'v9_required_for_native=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 native boundary drifted'
grep -Fxq 'systemd_sys_mount_path=/sys' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 sys mountpoint correction drifted'
grep -Fxq 'systemd_var_tmp_path=/var/tmp' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 var-tmp correction drifted'
grep -Fxq 'principal_observer_enumeration=forbidden' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 principal-opacity contract drifted'
grep -Fxq 'empty_observer=ROOT_HOST' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 root-observer contract drifted'
grep -Fxq 'effect_cell_max_bytes=16777216' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 effect-cell bound drifted'
grep -Fxq 'payload_max_bytes=1048576' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 payload bound drifted'
grep -Fxq 'policy_manifest_max_bytes=65536' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 manifest bound drifted'
grep -Fxq 'effective_dynamic_user=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 DynamicUser contract drifted'
grep -Fxq 'effective_private_tmp=disconnected' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 PrivateTmp contract drifted'
grep -Fxq 'effective_protect_system=strict' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 ProtectSystem contract drifted'
grep -Fxq 'effective_protect_home=read-only' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 ProtectHome contract drifted'
grep -Fxq 'filesystem_authority=ROOT_HOST_MOUNTINFO' "$POLICY_MANIFEST" ||
  fail 'Sounio V9 filesystem authority drifted'

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp "${TMPDIR:-/tmp}/loom-process-witness-effect-policy-v9.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches \
  -frandom-seed=loom-process-witness-effect-policy-v9 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -lcrypto -o "$stage"
if readelf -l "$stage" | grep -q 'INTERP'; then
  fail 'native V9 cell retained a dynamic interpreter'
fi
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V9 path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 policy_v9_bound=true static=true object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB effective_mount_truth=DynamicUser+disconnected+strict+read-only filesystem_authority=ROOT_HOST_MOUNTINFO systemd_mount=/run/systemd/incoming principal_readable=false principal_enumeration=forbidden empty_observer=ROOT_HOST systemd_sys_mount=/sys systemd_var_tmp=/var/tmp material_coverage=false complete_effects=false material_execution=false launch_open=false\n' \
  "$OUTPUT"
