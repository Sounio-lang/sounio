#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4_SOURCE:-$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v4.cpp}"
SHARED_SOURCE="$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v3.cpp"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v4.freeze.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-process-witness-effect-policy-v4}"

fail() {
  printf 'build-loom-process-witness-effect-policy-v4: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" sha256sum readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is missing: $tool"
done
for path in "$SOURCE" "$SHARED_SOURCE"; do
  [[ -f "$path" && ! -L "$path" ]] || fail "source is absent or linked: $path"
done
[[ -f "$POLICY_MANIFEST" && ! -L "$POLICY_MANIFEST" ]] ||
  fail 'frozen Sounio V4 policy manifest is absent or linked'
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  60cff91db90e9214e62a6fa5b45521249e31649c63dce297683ca477fcd3d627 ]] ||
  fail 'frozen Sounio V4 policy manifest drifted'
grep -Fxq 'stage=SEMANTICS_FROZEN' "$POLICY_MANIFEST" ||
  fail 'Sounio V4 policy is not frozen'
grep -Fxq 'v4_required_for_native=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V4 native boundary drifted'
grep -Fxq 'systemd_mount_path=/run/systemd/incoming' "$POLICY_MANIFEST" ||
  fail 'Sounio V4 mountpoint correction drifted'

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp "${TMPDIR:-/tmp}/loom-process-witness-effect-policy-v4.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches \
  -frandom-seed=loom-process-witness-effect-policy-v4 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -lcrypto -o "$stage"
if readelf -l "$stage" | grep -q 'INTERP'; then
  fail 'native V4 cell retained a dynamic interpreter'
fi
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4 path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 policy_v4_bound=true static=true object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE systemd_mount=/run/systemd/incoming material_coverage=false complete_effects=false material_execution=false launch_open=false\n' \
  "$OUTPUT"
