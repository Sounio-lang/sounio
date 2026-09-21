#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v10_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V10.md"
V9_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v9.freeze.v1"
FAILURE_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v9-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V10_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v10}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v10: FAIL: %s\n' "$*" >&2
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
  12c7f802ab79de2a9bdc894b93ffe01f3943d9aa133c808de65abcd067923081
expect_regular_hash "$V9_MANIFEST" \
  9d747d937a6a2316dd8894b37e243180031b8518f2696b9200ee7d1f1d81868c
expect_regular_hash "$FAILURE_EVIDENCE" \
  260a993e35974bb4d1899fb376b3682fbb6813b063c271c8f7c551d6ebfc6725
expect_regular_hash "$SOURCE" \
  6ba54ff1a9301c2621d778a2b2d02aba9affc5a8e43e57304a9b3050cc7a725a
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V9_MANIFEST" || fail 'V9 policy is not frozen'
grep -Fxq 'falsifying_status=HOST_GATE_FAIL' "$FAILURE_EVIDENCE" ||
  fail 'V9 READY-stage refusal is absent'
grep -Fxq 'falsifying_rule=forbidden_mounts' "$FAILURE_EVIDENCE" ||
  fail 'V9 path-only mount boundary drifted'
grep -Fxq 'falsifying_mountpoint=/proc' "$FAILURE_EVIDENCE" ||
  fail 'V9 /proc falsifier is absent'
grep -Fxq 'falsifying_mount_filesystem=ext4' "$FAILURE_EVIDENCE" ||
  fail 'V9 inert backing-filesystem observation is absent'
grep -Fxq 'v9_requested_mount_model_sufficient=false' "$FAILURE_EVIDENCE" ||
  fail 'V9 path-only mount model insufficiency is absent'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v10.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v10"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V10 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V10 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=10 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=8 authority_cases=18 allowed_syscalls=4' ]] ||
  fail "V10 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V10 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE identity_typed_mounts=CAPSULE_EMPTY_BIND families=12 treatments=12 sabotages=12 bootstrap_cases=8 authority_cases=18 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
