#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v9.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v9-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v9-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V9.md \
  1d15d6713d1659b7d539706c5f963eead0c275481e0ab8ff9a7f113558bde69c
expect_hash tools/loom/process_witness_effect_policy_plan_v9_main.sio \
  020cd5770d907725eb867fa28ce1bbb7da318d80b303cf114aa350e9e9d7c0aa
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v9.sh \
  00144c6f778a2f3cd7437fa653156dd38fdd3f4494ba480e832fec9245b9cd5e
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v9_selftest.sh \
  2425901165e0809a4495271199999e7773c8ac9f4aa215d1cf9eb0efb5fa8b21
expect_hash tools/loom/process_witness_effect_policy_plan_v8.freeze.v1 \
  f97bd4c3c8cd93978da27b361bc7fec3d8316775fb58a9a4bf94ddf53513293a
expect_hash tools/loom/evidence/loom-process-witness-effect-root-v8-host-attempt-v1-20260828.txt \
  f58fbc4513831cb5d503a1c65b1f5e32865829f24360264a11b2192e0338cae7
expect_hash "$EVIDENCE" \
  227cebb4949fc44abf456df7593b9a03e19c10c09016105ab3f21292278b3d18

for line in \
  'schema=loom-process-witness-effect-policy-plan-v9-freeze-v1' \
  'stage=SEMANTICS_FROZEN' \
  'producing_language=Sounio' \
  'language_role=SEMANTIC_POLICY_PLAN' \
  'semantic_authority=Sounio' \
  'action=9025' \
  'garden_commit=37c802eb3371f34ac32e34eced280e975daa78ea' \
  'sounio_executable_commit=2fb3941f020527c3c0bc6fc019e58a3321bd637a' \
  'executable_sha256=3e8d27444b1e58c0fb3e0c0de81c263375550f7e1952eb99db84acf8edc3caed' \
  'bundle_sha256=a2e840297b491f130d0c3b80e6355db8ba64fc3957767e2bebe16af5e2f83c0d' \
  'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE' \
  'effect_cell_max_bytes=16777216' \
  'payload_max_bytes=1048576' \
  'policy_manifest_max_bytes=65536' \
  'payload_manifest_max_bytes=65536' \
  'file_bound_diagnostic=object+observed_size+configured_max' \
  'systemd_mount_path=/run/systemd/incoming' \
  'systemd_mount_source=/run/systemd/propagate/EXACT_UNIT' \
  'principal_observer_readable=false' \
  'principal_observer_enumeration=forbidden' \
  'empty_observer=ROOT_HOST' \
  'mount_observer=ROOT_HOST' \
  'extinction_observer=ROOT_HOST' \
  'systemd_sys_mount_path=/sys' \
  'systemd_sys_ready_filesystem=sysfs' \
  'systemd_sys_ready_source=sysfs' \
  'systemd_sys_ready_read_only=true' \
  'systemd_var_tmp_path=/var/tmp' \
  'systemd_var_tmp_ready_source=IMMUTABLE_ROOT_TMP' \
  'systemd_var_tmp_ready_read_only=true' \
  'systemd_version=257' \
  'effective_dynamic_user=true' \
  'effective_private_tmp=disconnected' \
  'property_private_tmp_observed=yes' \
  'effective_protect_system=strict' \
  'effective_protect_home=read-only' \
  'property_authority=CONFIGURATION_ONLY' \
  'filesystem_authority=ROOT_HOST_MOUNTINFO' \
  'temporary_sources=SAME_IMMUTABLE_ROOT_TMP' \
  'temporary_read_only=true' \
  'temporary_empty=true' \
  'forbidden_mounts=/proc+/home+/root+/run+/var+/etc' \
  'bootstrap_case_count=4' \
  'bootstrap_treatment_code=0' \
  'bootstrap_missing_incoming_code=226' \
  'bootstrap_missing_sys_code=226' \
  'bootstrap_missing_var_tmp_code=226' \
  'allowed_syscalls=0,1,60,322' \
  'family_10_probe=personality_change' \
  'v8_materializable=false' \
  'v9_required_for_native=true' \
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
  'evidence_sha256=227cebb4949fc44abf456df7593b9a03e19c10c09016105ab3f21292278b3d18'; do
  require_line "$MANIFEST" "$line"
done

for line in \
  'hardware_arch=x86_64' \
  'hardware_kernel=Linux_7.0.2-5-pve' \
  'effect_cell_max_bytes=16777216' \
  'payload_max_bytes=1048576' \
  'policy_manifest_max_bytes=65536' \
  'payload_manifest_max_bytes=65536' \
  'systemd_mount_path=/run/systemd/incoming' \
  'principal_observer_readable=false' \
  'principal_observer_enumeration=forbidden' \
  'empty_observer=ROOT_HOST' \
  'mount_observer=ROOT_HOST' \
  'extinction_observer=ROOT_HOST' \
  'systemd_sys_mount_path=/sys' \
  'systemd_var_tmp_path=/var/tmp' \
  'effective_dynamic_user=true' \
  'effective_private_tmp=disconnected' \
  'property_private_tmp_observed=yes' \
  'effective_protect_system=strict' \
  'effective_protect_home=read-only' \
  'property_authority=CONFIGURATION_ONLY' \
  'filesystem_authority=ROOT_HOST_MOUNTINFO' \
  'temporary_sources=SAME_IMMUTABLE_ROOT_TMP' \
  'temporary_read_only=true' \
  'temporary_empty=true' \
  'forbidden_mounts=/proc+/home+/root+/run+/var+/etc' \
  'bootstrap_treatment_decision=SOUNIO_ROOT_BOOTSTRAP_ALLOW code=0 reason=systemd-mountpoints-present stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_incoming_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=incoming-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_sys_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=sys-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_var_tmp_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=var-tmp-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'v8_native_consumption=refused' \
  'native_executable_invoked=false' \
  'expected_results_source=Sounio' \
  'expected_results_encoded_in_shell=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'complete_effects=false' \
  'material_execution=false'; do
  require_line "$EVIDENCE" "$line"
done

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v9_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v9-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V9 gate failed'
[[ "$result" == *'families=12 treatments=12 sabotages=12 bootstrap_cases=4 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change'* ]] ||
  fail 'Sounio V9 bootstrap, surface, or decision classes drifted'
[[ "$result" == *'effective_mount_truth=DynamicUser+disconnected+strict+read-only'* ]] ||
  fail 'Sounio V9 effective mount truth drifted'
[[ "$result" == *'root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V9 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v9-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB effective_mount_truth=DynamicUser+disconnected+strict+read-only filesystem_authority=ROOT_HOST_MOUNTINFO systemd_mount=/run/systemd/incoming systemd_sys_mount=/sys systemd_var_tmp=/var/tmp bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
