#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v5.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v5-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v5-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "frozen path is absent or linked: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen path hash drifted: $path"
}

require_line() {
  local path="$1" value="$2"
  grep -Fxq "$value" "$path" || fail "required line is absent: $value"
}

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V5.md \
  fa79e0c56bd5e5083a9281c266fd3660a453ef51d1a4640c63e2fd90056b9300
expect_hash tools/loom/process_witness_effect_policy_plan_v5_main.sio \
  e6f00b3a244f8f56fa44e9571b4e2ebef39a708caafc3173884ee3120943f155
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v5.sh \
  6a5aee17a2271803b7010a83702cffd2a2817cd3612f76563d815be53d7511f5
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v5_selftest.sh \
  900583c027bcb99ce02c78da33247b3519dd1352bf4e7c23e52d8b281fb77916
expect_hash tools/loom/process_witness_effect_policy_plan_v4.freeze.v1 \
  60cff91db90e9214e62a6fa5b45521249e31649c63dce297683ca477fcd3d627
expect_hash tools/loom/evidence/loom-process-witness-effect-root-v4-host-attempt-v1-20260828.txt \
  2659e6881403784034ab0078a5de64a1eb35c2c96d8c563b98a951c45ac09b9e
expect_hash "$EVIDENCE" \
  f90b3c267aec751e6778b8b1a6b2925de8db3796db4562d23451f2ab7dd2a3bd

for line in \
  'schema=loom-process-witness-effect-policy-plan-v5-freeze-v1' \
  'stage=SEMANTICS_FROZEN' \
  'producing_language=Sounio' \
  'language_role=SEMANTIC_POLICY_PLAN' \
  'semantic_authority=Sounio' \
  'action=9025' \
  'garden_commit=c4ab02c7f2205fa7b106007da4692d2a560604de' \
  'sounio_executable_commit=a1b39d503fddcec01ca2623f062c86ad9a21ae01' \
  'executable_sha256=e4318b853a7485352f5cce829456fd90e1306c6e526fb48ebb5876f761459b17' \
  'bundle_sha256=f8f7f064cdf4382656f4620052ce7daf5b311fac789cfcde607627e9462f3af3' \
  'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE' \
  'systemd_mount_path=/run/systemd/incoming' \
  'systemd_mount_source=/run/systemd/propagate/EXACT_UNIT' \
  'systemd_sys_mount_path=/sys' \
  'systemd_sys_ready_filesystem=sysfs' \
  'systemd_sys_ready_source=sysfs' \
  'systemd_sys_ready_read_only=true' \
  'systemd_version=257' \
  'bootstrap_case_count=3' \
  'bootstrap_treatment_code=0' \
  'bootstrap_missing_incoming_code=226' \
  'bootstrap_missing_sys_code=226' \
  'allowed_syscalls=0,1,60,322' \
  'family_10_probe=personality_change' \
  'v4_materializable=false' \
  'v5_required_for_native=true' \
  'expected_results_encoded_in_shell=false' \
  'python_executable_invoked=false' \
  'rust_executable_invoked=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'material_coverage=false' \
  'complete_effects=false' \
  'material_execution=false' \
  'launch_open=false' \
  'recycle_open=false' \
  'exec_attached=false' \
  'commit_attached=false' \
  'ci_attached=false' \
  'parity_open=false' \
  'claim_ready=false' \
  'evidence_sha256=f90b3c267aec751e6778b8b1a6b2925de8db3796db4562d23451f2ab7dd2a3bd'; do
  require_line "$MANIFEST" "$line"
done

for line in \
  'hardware_arch=x86_64' \
  'hardware_kernel=Linux_7.0.2-5-pve' \
  'systemd_mount_path=/run/systemd/incoming' \
  'systemd_sys_mount_path=/sys' \
  'bootstrap_treatment_decision=SOUNIO_ROOT_BOOTSTRAP_ALLOW code=0 reason=systemd-mountpoints-present stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_incoming_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=incoming-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_sys_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=sys-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'v4_native_consumption=refused' \
  'native_executable_invoked=false' \
  'expected_results_source=Sounio' \
  'expected_results_encoded_in_shell=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'complete_effects=false' \
  'material_execution=false'; do
  require_line "$EVIDENCE" "$line"
done

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v5_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v5-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V5 gate failed'
[[ "$result" == *'families=12 treatments=12 sabotages=12 bootstrap_cases=3 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change'* ]] ||
  fail 'Sounio V5 bootstrap, surface, or decision classes drifted'
[[ "$result" == *'root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V5 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v5-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s systemd_mount=/run/systemd/incoming systemd_sys_mount=/sys bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
