#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v4_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V4.md"
V3_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v3.freeze.v1"
FAILURE_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v3-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V4_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v4}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v4: FAIL: %s\n' "$*" >&2
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
  a0339f8cfc3e070db19e86fe29cd301a3e09f0bc5883e85e8dab7eaf21a87744
expect_regular_hash "$V3_MANIFEST" \
  40407323594e37d44b9002d1cdd390677416048221ace446693919f8415ca480
expect_regular_hash "$FAILURE_EVIDENCE" \
  baeb296039daf112f66d22e7ad7f57e2a605702a964b444dc8a8a1c6325c37e5
expect_regular_hash "$SOURCE" \
  d0e8db991e56952ed950bebf03b20823867004fe75ae3e7cd9710f1d35df0222
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V3_MANIFEST" || fail 'V3 policy is not frozen'
grep -Fxq 'falsifying_status=226/NAMESPACE' "$FAILURE_EVIDENCE" ||
  fail 'V3 namespace refusal is absent'
grep -Fxq 'falsifying_required_path=/run/systemd/incoming' "$FAILURE_EVIDENCE" ||
  fail 'V3 required systemd mountpoint drifted'
grep -Fxq 'v3_exact_root_schema_sufficient=false' "$FAILURE_EVIDENCE" ||
  fail 'V3 root insufficiency is absent'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v4.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v4"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V4 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V4 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=4 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=2 allowed_syscalls=4' ]] ||
  fail "V4 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V4 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=2 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
