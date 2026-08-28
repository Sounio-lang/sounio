#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3_SOURCE:-$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v3.cpp}"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v3.freeze.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-process-witness-effect-policy-v3}"

fail() {
  printf 'build-loom-process-witness-effect-policy-v3: FAIL reason=%s\n' "$*" >&2
  exit 1
}

for tool in "$CXX" sha256sum readelf; do
  command -v "$tool" >/dev/null 2>&1 || fail "required build tool is missing: $tool"
done
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail "source is absent or linked: $SOURCE"
[[ -f "$POLICY_MANIFEST" && ! -L "$POLICY_MANIFEST" ]] ||
  fail 'frozen Sounio V3 policy manifest is absent or linked'
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  40407323594e37d44b9002d1cdd390677416048221ace446693919f8415ca480 ]] ||
  fail 'frozen Sounio V3 policy manifest drifted'
grep -Fxq 'stage=SEMANTICS_FROZEN' "$POLICY_MANIFEST" ||
  fail 'Sounio V3 policy is not frozen'
grep -Fxq 'v3_required_for_native=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V3 native boundary drifted'
grep -Fxq 'static_native_required=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V3 static-cell requirement drifted'

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp "${TMPDIR:-/tmp}/loom-process-witness-effect-policy-v3.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -static \
  -fno-record-gcc-switches \
  -frandom-seed=loom-process-witness-effect-policy-v3 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -lcrypto -o "$stage"
if readelf -l "$stage" | grep -q 'INTERP'; then
  fail 'native V3 cell retained a dynamic interpreter'
fi
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3 path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 policy_v3_bound=true static=true object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE material_coverage=false complete_effects=false material_execution=false launch_open=false\n' \
  "$OUTPUT"
