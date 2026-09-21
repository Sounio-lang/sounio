#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-native.XXXXXX")"
BINARY_ONE="$TEST_ROOT/policy-one"
BINARY_TWO="$TEST_ROOT/policy-two"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v2.freeze.v1"
V1_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan.freeze.v1"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-policy-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$BINARY_ONE" "$BINARY_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_policy.sh" \
      >/dev/null
done
cmp "$BINARY_ONE" "$BINARY_TWO" || fail 'two source-fresh native builds differ'
[[ "$(stat -c '%a' "$BINARY_ONE")" == 755 && ! -u "$BINARY_ONE" &&
   ! -g "$BINARY_ONE" ]] || fail 'native policy executable mode is unsafe'

result="$($BINARY_ONE --selftest --policy-manifest "$POLICY_MANIFEST")"
[[ "$result" == LOOM_PROCESS_WITNESS_EFFECT_POLICY_SELFTEST\ PASS* ]] ||
  fail "native policy selftest failed: $result"
[[ "$result" == *'semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true action=9025 policy_v2_bound=true'* ]] ||
  fail 'native policy language authority drifted'
[[ "$result" == *'seccomp_default=EPERM architecture=AUDIT_ARCH_X86_64 allowed_syscalls=0+1+60+322 allowed_io=true seccomp_treatments=12 '* ]] ||
  fail 'native policy treatment or sabotage boundary drifted'
[[ "$result" == *'structural_sabotages=12 material_sabotages=0 '* &&
   "$result" == *'host_gate_required=true material_coverage=false'* ]] ||
  fail 'native policy did not preserve the host material boundary'
[[ "$result" == *'material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'native policy selftest promoted beyond evidence'

if "$BINARY_ONE" --selftest --policy-manifest "$V1_MANIFEST" >/dev/null 2>&1; then
  fail 'native policy accepted the insufficient V1 manifest'
fi
tampered="$TEST_ROOT/tampered-v2.freeze.v1"
sed 's/allowed_syscalls=0,1,60,322/allowed_syscalls=0,1,39,60,322/' \
  "$POLICY_MANIFEST" > "$tampered"
if "$BINARY_ONE" --selftest --policy-manifest "$tampered" >/dev/null 2>&1; then
  fail 'native policy accepted a tampered syscall surface'
fi

dependencies="$(ldd "$BINARY_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'native policy executable has a prohibited runtime dependency'
fi

printf 'sounio-loom-process-witness-effect-policy-selftest: PASS semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true action=9025 policy_v2_bound=true source_sha256=%s executable_sha256=%s seccomp_treatments=12 structural_sabotages=12 material_sabotages=0 host_gate_required=true v1_native=refused tamper=refused deterministic=true runtime_dependencies=clean material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy.cpp" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BINARY_ONE" | cut -d ' ' -f 1)"
