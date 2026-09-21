#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v6.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v6-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v6-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V6.md \
  dc41cec54c13a44f3d49ba59cea4fb6f79c93b569d351ace829d946ad06b39d9
expect_hash tools/loom/process_witness_effect_policy_plan_v6_main.sio \
  f589cf66d8b188e4d8024eb31d761999c77f9fc2a6db3e7a7d58688b85a300a1
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v6.sh \
  c2a32ae832ec7514633aeaff76cd4c09e883b57589e7681bf9eaaba4b0b6ce35
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v6_selftest.sh \
  59330e54ab38c9c4f374aee5603c48183c973fb9c14fabb240a434d6b980adec
expect_hash tools/loom/process_witness_effect_policy_plan_v5.freeze.v1 \
  f17fc7d776db557d2655e00036f4014b4a7a38d8ed16e74786471415c49908f7
expect_hash tools/loom/evidence/loom-process-witness-effect-root-v5-host-attempt-v1-20260828.txt \
  1cfd0bba84732d156f220b20ede0cfd9cbf22b3902f474ebada922f29506272f
expect_hash "$EVIDENCE" \
  e1caa82c570e89062a3f73c7493e3c468adbdd6fefe1343fddc61bf54db8612e

for line in \
  'schema=loom-process-witness-effect-policy-plan-v6-freeze-v1' \
  'stage=SEMANTICS_FROZEN' \
  'producing_language=Sounio' \
  'language_role=SEMANTIC_POLICY_PLAN' \
  'semantic_authority=Sounio' \
  'action=9025' \
  'garden_commit=0af569ec4ca0c5b142dce37315602d3ebf09c3c2' \
  'sounio_executable_commit=7403f73c48fe529e1d25820a0f036e877ed11abd' \
  'executable_sha256=b709dd9291d35d63217f3dedb4a609ffe7254099bde66c750341bf4304398386' \
  'bundle_sha256=974c90d17f91321291e6ad95337a4c22138253c65114fdde41076b5af96613b9' \
  'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE' \
  'systemd_mount_path=/run/systemd/incoming' \
  'systemd_mount_source=/run/systemd/propagate/EXACT_UNIT' \
  'systemd_sys_mount_path=/sys' \
  'systemd_sys_ready_filesystem=sysfs' \
  'systemd_sys_ready_source=sysfs' \
  'systemd_sys_ready_read_only=true' \
  'systemd_var_tmp_path=/var/tmp' \
  'systemd_var_tmp_ready_source=IMMUTABLE_ROOT_TMP' \
  'systemd_var_tmp_ready_read_only=true' \
  'systemd_version=257' \
  'bootstrap_case_count=4' \
  'bootstrap_treatment_code=0' \
  'bootstrap_missing_incoming_code=226' \
  'bootstrap_missing_sys_code=226' \
  'bootstrap_missing_var_tmp_code=226' \
  'allowed_syscalls=0,1,60,322' \
  'family_10_probe=personality_change' \
  'v5_materializable=false' \
  'v6_required_for_native=true' \
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
  'evidence_sha256=e1caa82c570e89062a3f73c7493e3c468adbdd6fefe1343fddc61bf54db8612e'; do
  require_line "$MANIFEST" "$line"
done

for line in \
  'hardware_arch=x86_64' \
  'hardware_kernel=Linux_7.0.2-5-pve' \
  'systemd_mount_path=/run/systemd/incoming' \
  'systemd_sys_mount_path=/sys' \
  'systemd_var_tmp_path=/var/tmp' \
  'bootstrap_treatment_decision=SOUNIO_ROOT_BOOTSTRAP_ALLOW code=0 reason=systemd-mountpoints-present stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_incoming_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=incoming-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_sys_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=sys-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_var_tmp_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=var-tmp-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'v5_native_consumption=refused' \
  'native_executable_invoked=false' \
  'expected_results_source=Sounio' \
  'expected_results_encoded_in_shell=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'complete_effects=false' \
  'material_execution=false'; do
  require_line "$EVIDENCE" "$line"
done

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v6_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v6-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V6 gate failed'
[[ "$result" == *'families=12 treatments=12 sabotages=12 bootstrap_cases=4 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change'* ]] ||
  fail 'Sounio V6 bootstrap, surface, or decision classes drifted'
[[ "$result" == *'root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V6 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v6-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s systemd_mount=/run/systemd/incoming systemd_sys_mount=/sys systemd_var_tmp=/var/tmp bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
