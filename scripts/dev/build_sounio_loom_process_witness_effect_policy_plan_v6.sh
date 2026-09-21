#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V6_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V6_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v6_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V6.md"
V5_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v5.freeze.v1"
FAILURE_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v5-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V6_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v6}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v6: FAIL: %s\n' "$*" >&2
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
  dc41cec54c13a44f3d49ba59cea4fb6f79c93b569d351ace829d946ad06b39d9
expect_regular_hash "$V5_MANIFEST" \
  f17fc7d776db557d2655e00036f4014b4a7a38d8ed16e74786471415c49908f7
expect_regular_hash "$FAILURE_EVIDENCE" \
  1cfd0bba84732d156f220b20ede0cfd9cbf22b3902f474ebada922f29506272f
expect_regular_hash "$SOURCE" \
  f589cf66d8b188e4d8024eb31d761999c77f9fc2a6db3e7a7d58688b85a300a1
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V5_MANIFEST" || fail 'V5 policy is not frozen'
grep -Fxq 'falsifying_status=226/NAMESPACE' "$FAILURE_EVIDENCE" ||
  fail 'V5 namespace refusal is absent'
grep -Fxq 'falsifying_required_path=/var/tmp' "$FAILURE_EVIDENCE" ||
  fail 'V5 required systemd mountpoint drifted'
grep -Fxq 'v5_exact_root_schema_sufficient=false' "$FAILURE_EVIDENCE" ||
  fail 'V5 root insufficiency is absent'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v6.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v6"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V6 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V6 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=6 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=4 allowed_syscalls=4' ]] ||
  fail "V6 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V6 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=4 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
