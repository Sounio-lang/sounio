#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
SOUC="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V7_SOUC:-$ROOT_DIR/bin/souc}"
ENGINE="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V7_ENGINE:-lean_single}"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v7_main.sio"
GARDEN="$ROOT_DIR/tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V7.md"
V6_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v6.freeze.v1"
FAILURE_EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-effect-root-v6-host-attempt-v1-20260828.txt"
OUTPUT="${SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V7_OUTPUT:-$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-process-witness-effect-policy-plan-v7}"

fail() {
  printf 'build-sounio-loom-process-witness-effect-policy-plan-v7: FAIL: %s\n' "$*" >&2
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
  8460371e45502db50501b4385bfe438677002a595601f161cc36cec5a95d80c4
expect_regular_hash "$V6_MANIFEST" \
  6ec33f3554236e7ccf73f5b5c16a15ba8006705b83d9d62265a2cd8f94437d66
expect_regular_hash "$FAILURE_EVIDENCE" \
  04e296b2f27c54b598f6902ee936a1d6eee1354f046bafb38e0d928a6f1941b5
expect_regular_hash "$SOURCE" \
  f5f0a881e2b2d55b06f1ed6a9120aad24df2b2460ea8913283d2d46d1ef8c81f
grep -Fxq 'stage=SEMANTICS_FROZEN' "$V6_MANIFEST" || fail 'V6 policy is not frozen'
grep -Fxq 'falsifying_status=70/SOFTWARE' "$FAILURE_EVIDENCE" ||
  fail 'V6 post-exec refusal is absent'
grep -Fxq 'falsifying_path=/run/systemd/incoming' "$FAILURE_EVIDENCE" ||
  fail 'V6 opaque rendezvous path drifted'
grep -Fxq 'v6_principal_readability_assumption_sufficient=false' "$FAILURE_EVIDENCE" ||
  fail 'V6 observer-assignment insufficiency is absent'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-v7.XXXXXX")"
trap 'rm -rf "$work"' EXIT
compiled="$work/sounio-loom-process-witness-effect-policy-plan-v7"
SOUNIO_SOUC_ENGINE="$ENGINE" "$SOUC" compile "$SOURCE" -o "$compiled"
[[ -f "$compiled" ]] || fail 'compiler omitted the Sounio V7 executable'
mkdir -p "$(dirname "$OUTPUT")"
install -m 0755 "$compiled" "$OUTPUT"

metadata="$($OUTPUT | sed -n '1p')"
[[ "$metadata" == \
  'LOOM_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V7 producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 schema=7 object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=4 allowed_syscalls=4' ]] ||
  fail "V7 policy metadata diverged: $metadata"

printf 'BUILT_PROCESS_WITNESS_EFFECT_POLICY_PLAN_V7 path=%s language=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 engine=%s object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE families=12 treatments=12 sabotages=12 bootstrap_cases=4 allowed_syscalls=4\n' \
  "$OUTPUT" "$ENGINE"
