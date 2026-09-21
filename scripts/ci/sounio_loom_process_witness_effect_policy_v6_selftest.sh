#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-policy-native-v6.XXXXXX")"
BINARY_ONE="$TEST_ROOT/policy-one"
BINARY_TWO="$TEST_ROOT/policy-two"
POLICY_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v6.freeze.v1"
V5_MANIFEST="$ROOT_DIR/tools/loom/process_witness_effect_policy_plan_v5.freeze.v1"

cleanup() { rm -rf "$TEST_ROOT"; }
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-policy-v6-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

for output in "$BINARY_ONE" "$BINARY_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_EFFECT_POLICY_V6_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_process_witness_effect_policy_v6.sh" \
      >/dev/null
done
cmp "$BINARY_ONE" "$BINARY_TWO" || fail 'two source-fresh native V6 builds differ'
[[ "$(stat -c '%a' "$BINARY_ONE")" == 755 && ! -u "$BINARY_ONE" &&
   ! -g "$BINARY_ONE" ]] || fail 'native V6 executable mode is unsafe'
if readelf -l "$BINARY_ONE" | grep -q 'INTERP'; then
  fail 'native V6 executable is dynamically linked'
fi

result="$($BINARY_ONE --selftest --policy-manifest "$POLICY_MANIFEST")"
[[ "$result" == LOOM_PROCESS_WITNESS_EFFECT_POLICY_V6_SELFTEST\ PASS* ]] ||
  fail "native V6 policy selftest failed: $result"
[[ "$result" == *'semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true action=9025 policy_v6_bound=true static=true'* ]] ||
  fail 'native V6 policy language authority drifted'
[[ "$result" == *'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE landlock_required=false family10=personality_change seccomp_default=EPERM architecture=AUDIT_ARCH_X86_64 allowed_syscalls=0+1+60+322 allowed_io=true'* ]] ||
  fail 'native V6 object or syscall boundary drifted'
[[ "$result" == *'seccomp_treatments=11 structural_root_treatments=1 structural_sabotages=12 material_sabotages=0 '* &&
   "$result" == *'root_gate_required=true host_gate_required=true material_coverage=false'* ]] ||
  fail 'native V6 policy did not preserve its material boundary'

if "$BINARY_ONE" --selftest --policy-manifest "$V5_MANIFEST" >/dev/null 2>&1; then
  fail 'native V6 policy accepted the insufficient V5 manifest'
fi
tampered="$TEST_ROOT/tampered-v6.freeze.v1"
sed 's#systemd_var_tmp_path=/var/tmp#systemd_var_tmp_path=/host-var-tmp#' \
  "$POLICY_MANIFEST" > "$tampered"
if "$BINARY_ONE" --selftest --policy-manifest "$tampered" >/dev/null 2>&1; then
  fail 'native V6 policy accepted a widened sys mountpoint'
fi

dependencies="$(ldd "$BINARY_ONE" 2>&1 || true)"
if ! printf '%s\n' "$dependencies" | grep -Eq 'not a dynamic executable|statically linked'; then
  fail 'native V6 executable did not prove static linkage'
fi
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'native V6 policy executable has a prohibited runtime dependency'
fi

printf 'sounio-loom-process-witness-effect-policy-v6-selftest: PASS semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true action=9025 policy_v6_bound=true static=true object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE systemd_mount=/run/systemd/incoming systemd_sys_mount=/sys systemd_var_tmp=/var/tmp wrapper_source_sha256=%s shared_source_sha256=%s executable_sha256=%s seccomp_treatments=11 structural_root_treatments=1 structural_sabotages=12 material_sabotages=0 family10=personality_change root_gate_required=true host_gate_required=true v5_native=refused tamper=refused deterministic=true runtime_dependencies=static material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v6.cpp" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_process_witness_effect_policy_v3.cpp" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BINARY_ONE" | cut -d ' ' -f 1)"
