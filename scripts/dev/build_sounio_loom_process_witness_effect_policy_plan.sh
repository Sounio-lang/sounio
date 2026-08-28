#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_CELL_V1.md"
EFFECT_MANIFEST="$ROOT_DIR/tools/loom/effect_closure_authority.freeze.v1"
PROCESS_MANIFEST="$ROOT_DIR/tools/loom/process_witness_host.runtime.v1"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan: FAIL: %s\n' "$*" >&2
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
  21630b55ce12d7823e7d66408a2ef7af53d833a4b83182b130a76e83c6395cb3
expect_regular_hash "$EFFECT_MANIFEST" \
  c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
expect_regular_hash "$PROCESS_MANIFEST" \
  eda00fee106a9f4090d381194b9f1bcd3838f3dcc0bafb0c7769a0877e05aa00
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail 'Sounio policy-plan source is absent or linked'

grep -Fxq 'stage=SEMANTICS_FROZEN' "$EFFECT_MANIFEST" ||
  fail 'action 9025 is not frozen'
grep -Fxq 'producing_language=Sounio' "$EFFECT_MANIFEST" ||
  fail 'action 9025 semantic producer drifted'
grep -Fxq 'language_role=SEMANTIC_AUTHORITY' "$EFFECT_MANIFEST" ||
  fail 'action 9025 language role drifted'
grep -Fxq 'action=9025' "$EFFECT_MANIFEST" || fail 'action 9025 identity drifted'
grep -Fxq 'stage=MATERIAL_EXECUTION_CORE_FROZEN' "$PROCESS_MANIFEST" ||
  fail 'ProcessWitness material parent is not frozen'
grep -Fxq 'complete_effects=false' "$PROCESS_MANIFEST" ||
  fail 'ProcessWitness parent no longer records closed effects'
grep -Fxq 'material_execution=false' "$PROCESS_MANIFEST" ||
  fail 'ProcessWitness parent no longer records closed material execution'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio policy-plan executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V1 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=1 families=12 treatments=12 sabotages=12' ]] ||
  fail "policy-plan metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s families=12 treatments=12 sabotages=12\n' \
  "$OUTPUT" "$ENGINE"
