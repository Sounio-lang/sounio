#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V5_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V5_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v5_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V5.md"
V4_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v4.freeze.v1"
FAILURE_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v4-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V5_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v5}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v5: FAIL: %s\n' "$*" >&2
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
  fa79e0c56bd5e5083a9281c266fd3660a453ef51d1a4640c63e2fd90056b9300
expect_regular_hash "$V4_MANIFEST" \
  60cff91db90e9214e62a6fa5b45521249e31649c63dce297683ca477fcd3d627
expect_regular_hash "$FAILURE_EVIDENCE" \
  2659e6881403784034ab0078a5de64a1eb35c2c96d8c563b98a951c45ac09b9e
expect_regular_hash "$SOURCE" \
  e6f00b3a244f8f56fa44e9571b4e2ebef39a708caafc3173884ee3120943f155
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V4_MANIFEST" || fail 'V4 policy is not frozen'
grep -Fxq 'falsifying_status=226/NAMESPACE' "$FAILURE_EVIDENCE" ||
  fail 'V4 namespace refusal is absent'
grep -Fxq 'falsifying_required_path=/sys' "$FAILURE_EVIDENCE" ||
  fail 'V4 required systemd mountpoint drifted'
grep -Fxq 'v4_exact_root_schema_sufficient=false' "$FAILURE_EVIDENCE" ||
  fail 'V4 root insufficiency is absent'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v5.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v5"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V5 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V5 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=5 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=3 allowed_syscalls=4' ]] ||
  fail "V5 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V5 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=3 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
