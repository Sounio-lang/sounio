#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
CXX="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_CXX:-c++}"
SOURCE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_SOURCE:-$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy.cpp}"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v2.freeze.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/loom-process-witness-effect-policy}"

fail() {
  printf 'build-loom-process-witness-effect-policy: FAIL reason=%s\n' "$*" >&2
  exit 1
}

command -v "$CXX" >/dev/null 2>&1 || fail "C++ compiler is missing: $CXX"
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail "source is absent or linked: $SOURCE"
[[ -f "$POLICY_MANIFEST" && ! -L "$POLICY_MANIFEST" ]] ||
  fail 'frozen Sounio V2 policy manifest is absent or linked'
[[ "$(sha256sum "$POLICY_MANIFEST" | cut -d ' ' -f 1)" == \
  d66b13252479252d5922ee0091e51a5bdb6a5eca9a592bb21f5db9dde344fee9 ]] ||
  fail 'frozen Sounio V2 policy manifest drifted'
grep -Fxq 'stage=SEMANTICS_FROZEN' "$POLICY_MANIFEST" ||
  fail 'Sounio V2 policy is not frozen'
grep -Fxq 'v2_required_for_native=true' "$POLICY_MANIFEST" ||
  fail 'Sounio V2 native boundary drifted'

mkdir -p "$(dirname "$OUTPUT")"
stage="$(mktemp "${TMPDIR:-/tmp}/loom-process-witness-effect-policy.XXXXXX")"
trap 'rm -f "$stage"' EXIT
"$CXX" -std=c++20 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-record-gcc-switches \
  -frandom-seed=loom-process-witness-effect-policy-v1 \
  -ffile-prefix-map="$ROOT_DIR"=. -Wl,--build-id=none \
  "$SOURCE" -lcrypto -o "$stage"
install -m 0755 "$stage" "$OUTPUT"

printf 'BUILT_LOOM_PROCESS_WITNESS_EFFECT_POLICY path=%s language=C++20 role=MATERIAL_PARITY transitory=true semantic_authority=Sounio action=9025 policy_v2_bound=true material_coverage=false complete_effects=false material_execution=false launch_open=false\n' \
  "$OUTPUT"
