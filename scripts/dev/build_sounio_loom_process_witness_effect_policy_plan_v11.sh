#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V11_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V11_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v11_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V11.md"
V10_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v10.freeze.v1"
V10_ROOT_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v10-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V11_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v11}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v11: FAIL: %s\n' "$*" >&2
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
  c065f6f30c721711ffa9cff74b3961043a9ca5ce1ab4938b440d984afab524cb
expect_regular_hash "$V10_MANIFEST" \
  9e7f42fd4bd18fd2b5f996b279a67f46a50546a20ef6949e4dc069c16b3d0dda
expect_regular_hash "$V10_ROOT_EVIDENCE" \
  96bea5a8306d61ed4528b5b29f92493c98fe6e95c1c6c8ee28930b0f5c2b0ca5
expect_regular_hash "$SOURCE" \
  42fe8c08510f00159f0ad63cb0aac620c35776d661a41691d290ffd44ae402e2
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V10_MANIFEST" || fail 'V10 policy is not frozen'
grep -Fxq 'proc_treatment=CAPSULE_EMPTY_BIND' "$V10_MANIFEST" ||
  fail 'V10 typed proc treatment is absent'
grep -Fxq 'root_treatment=true' "$V10_ROOT_EVIDENCE" ||
  fail 'V10 root treatment did not pass'
grep -Fxq 'bootstrap_sabotage=true' "$V10_ROOT_EVIDENCE" ||
  fail 'V10 bootstrap sabotage did not pass'
grep -Fxq 'bootstrap_negative_controls=7' "$V10_ROOT_EVIDENCE" ||
  fail 'V10 bootstrap negative-control count drifted'
grep -Fxq 'typed_proc_sabotages=4' "$V10_ROOT_EVIDENCE" ||
  fail 'V10 typed proc sabotage count drifted'
grep -Fxq 'material_coverage=false' "$V10_ROOT_EVIDENCE" ||
  fail 'V10 unexpectedly claims material effect coverage'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v11.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v11"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V11 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V11 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=11 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 probes=13 mechanism_dimensions=18 vertices=40 bootstrap_cases=8 action_cases=14 allowed_syscalls=4' ]] ||
  fail "V11 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V11 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 probes=13 mechanism_dimensions=18 vertices=40 bootstrap_cases=8 action_cases=14 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
