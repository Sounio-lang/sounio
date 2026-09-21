#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V9_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V9_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v9_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V9.md"
V8_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v8.freeze.v1"
FAILURE_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v8-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V9_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v9}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v9: FAIL: %s\n' "$*" >&2
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
  1d15d6713d1659b7d539706c5f963eead0c275481e0ab8ff9a7f113558bde69c
expect_regular_hash "$V8_MANIFEST" \
  f97bd4c3c8cd93978da27b361bc7fec3d8316775fb58a9a4bf94ddf53513293a
expect_regular_hash "$FAILURE_EVIDENCE" \
  f58fbc4513831cb5d503a1c65b1f5e32865829f24360264a11b2192e0338cae7
expect_regular_hash "$SOURCE" \
  020cd5770d907725eb867fa28ce1bbb7da318d80b303cf114aa350e9e9d7c0aa
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V8_MANIFEST" || fail 'V8 policy is not frozen'
grep -Fxq 'falsifying_status=HOST_GATE_FAIL' "$FAILURE_EVIDENCE" ||
  fail 'V8 READY-stage refusal is absent'
grep -Fxq 'falsifying_property=PrivateTmp' "$FAILURE_EVIDENCE" ||
  fail 'V8 effective-property boundary drifted'
grep -Fxq 'diagnostic_effective_value=yes' "$FAILURE_EVIDENCE" ||
  fail 'V8 DynamicUser PrivateTmp coercion is absent'
grep -Fxq 'v8_requested_property_model_sufficient=false' "$FAILURE_EVIDENCE" ||
  fail 'V8 requested-property insufficiency is absent'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v9.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v9"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V9 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V9 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=9 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=4 allowed_syscalls=4' ]] ||
  fail "V9 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V9 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=4 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
