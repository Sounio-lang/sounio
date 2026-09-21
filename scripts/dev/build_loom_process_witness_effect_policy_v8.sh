#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V8_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V8_SOURCE:-$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v8.cpp}"
SHARED_SOURCE="$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v3.cpp"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v8.freeze.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V8_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-process-witness-effect-policy-v8}"

fail() {
  printf 'build-loom-process-witness-effect-policy-v8: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" sha256sum readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is missing: $tool"
done
for path in "$SOURCE" "$SHARED_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "source is absent or linked: $path"
done
[[ -f "$POLICY_MANIFEST" && ! -L "$POLICY_MANIFEST" ]] ||
  fail 'frozen Sounio V8 policy manifest is absent or linked'
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  f97bd4c3c8cd93978da27b361bc7fec3d8316775fb58a9a4bf94ddf53513293a ]] ||
  fail 'frozen Sounio V8 policy manifest drifted'
grep -Fxq 'stage=SEMANTICS_FROZEN' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 policy is not frozen'
grep -Fxq 'v8_required_for_native=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 native boundary drifted'
grep -Fxq 'systemd_sys_mount_path=/sys' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 sys mountpoint correction drifted'
grep -Fxq 'systemd_var_tmp_path=/var/tmp' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 var-tmp correction drifted'
grep -Fxq 'principal_observer_enumeration=forbidden' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 principal-opacity contract drifted'
grep -Fxq 'empty_observer=ROOT_HOST' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 root-observer contract drifted'
grep -Fxq 'effect_cell_max_bytes=16777216' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 effect-cell bound drifted'
grep -Fxq 'payload_max_bytes=1048576' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 payload bound drifted'
grep -Fxq 'policy_manifest_max_bytes=65536' "$POLICY_MANIFEST" ||
  fail 'Sounio V8 manifest bound drifted'

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp "${TMPDIR:-/tmp}/loom-process-witness-effect-policy-v8.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches \
  -frandom-seed=loom-process-witness-effect-policy-v8 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -lcrypto -o "$stage"
if readelf -l "$stage" | grep -q 'INTERP'; then
  fail 'native V8 cell retained a dynamic interpreter'
fi
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V8 path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 policy_v8_bound=true static=true object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB systemd_mount=/run/systemd/incoming principal_readable=false principal_enumeration=forbidden empty_observer=ROOT_HOST systemd_sys_mount=/sys systemd_var_tmp=/var/tmp material_coverage=false complete_effects=false material_execution=false launch_open=false\n' \
  "$OUTPUT"
