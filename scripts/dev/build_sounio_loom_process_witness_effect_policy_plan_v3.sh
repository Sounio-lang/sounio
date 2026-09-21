#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v3_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V3.md"
V2_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v2.freeze.v1"
FAILURE_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-policy-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V3_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v3}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v3: FAIL: %s\n' "$*" >&2
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
  9c49364e27d36b2057e9ba33606c6dcbeb83c89b86b501645de9ea7fbf8e8185
expect_regular_hash "$V2_MANIFEST" \
  d66b13252479252d5922ee0091e51a5bdb6a5eca9a592bb21f5db9dde344fee9
expect_regular_hash "$FAILURE_EVIDENCE" \
  e702ceb3e2149d2d83cd054b147f9130e97fdb2d082b4ee839e24d4fcfdd24bb
[[ -f "$SOURCE" && ! -L "$SOURCE" ]] || fail 'Sounio V3 source is absent or linked'
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V2_MANIFEST" || fail 'V2 policy is not frozen'
grep -Fxq 'landlock_abi=-1' "$FAILURE_EVIDENCE" || fail 'Landlock refusal is absent'
grep -Fxq 'landlock_errno=95' "$FAILURE_EVIDENCE" || fail 'Landlock errno drifted'
grep -Fxq 'fallback_used=false' "$FAILURE_EVIDENCE" || fail 'V2 failure used a fallback'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v3.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v3"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V3 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V3 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=3 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 allowed_syscalls=4' ]] ||
  fail "V3 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V3 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
