#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V2_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V2_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v2_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V2.md"
V1_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan.freeze.v1"
EFFECT_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
PROCESS_MANIFEST="$ROOT_DIR/tools/loom/process_witness_host.runtime.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V2_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v2}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v2: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_regular_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen input is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen input hash drifted: $path"
}

[[ -x "$SOUC" ]] || fail "Sounio compiler is missing: $SOUC"
expect_regular_hash "$GARDEN" \
  08edcae4f07091b999f6c56e6a95cf7bca6d0bfa54ed6b9e13f2e49ea14a90ad
expect_regular_hash "$V1_MANIFEST" \
  14ee27eee71f04d1aa5462426379b37bb9c775215e94e17a864dbea308e43f21
expect_regular_hash "$EFFECT_MANIFEST" \
  c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
expect_regular_hash "$PROCESS_MANIFEST" \
  eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail 'Sounio V2 policy-plan source is absent or linked'

grep -Fxq 'stage=SEMANTICS_FROZEN' "$V1_MANIFEST" || fail 'V1 policy is not frozen'
grep -Fxq 'expected_results_source=Sounio' "$V1_MANIFEST" ||
  fail 'V1 expected-result authority drifted'
grep -Fxq 'material_coverage=false' "$V1_MANIFEST" ||
  fail 'V1 material boundary drifted'
grep -Fxq 'complete_effects=false' "$PROCESS_MANIFEST" ||
  fail 'ProcessWitness parent no longer records closed effects'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v2.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v2"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V2 policy executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V2 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=2 families=12 treatments=12 sabotages=12 allowed_syscalls=4' ]] ||
  fail "V2 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V2 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s families=12 treatments=12 sabotages=12 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
