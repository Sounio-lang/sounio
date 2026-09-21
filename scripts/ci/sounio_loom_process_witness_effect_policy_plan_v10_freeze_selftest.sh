#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v10.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v10-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v10-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V10.md \
  12c7f802ab79de2a9bdc894b93ffe01f3943d9aa133c808de65abcd067923081
expect_hash tools/loom/process_witness_effect_policy_plan_v10_main.sio \
  6ba54ff1a9301c2621d778a2b2d02aba9affc5a8e43e57304a9b3050cc7a725a
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v10.sh \
  7b5fbe07f71660028517897b647a5c0d35a1dc88ba1532954fd0b958cce2611b
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v10_selftest.sh \
  81d9efe4c5da8b2ded468b0318fe9d73858f589950eea3918bc79a1b73f061b5
expect_hash tools/loom/process_witness_effect_policy_plan_v9.freeze.v1 \
  9d747d937a6a2316dd8894b37e243180031b8518f2696b9200ee7d1f1d81868c
expect_hash tools/loom/evidence/loom-process-witness-effect-root-v9-host-attempt-v1-20260828.txt \
  260a993e35974bb4d1899fb376b3682fbb6813b063c271c8f7c551d6ebfc6725
expect_hash "$EVIDENCE" \
  92e53a0d995b5511628bffe3d2b91602743cac35ef6c5712471697003d9f2885

for line in \
  'schema=loom-process-witness-effect-policy-plan-v10-freeze-v1' \
  'stage=SEMANTICS_FROZEN' \
  'producing_language=Sounio' \
  'language_role=SEMANTIC_POLICY_PLAN' \
  'semantic_authority=Sounio' \
  'action=9025' \
  'garden_commit=7a9929d04ed3d830242ecacbeda425c558588b68' \
  'sounio_executable_commit=a7a06fc1bac60710b2cd31fdbc96b4f09e16c474' \
  'executable_sha256=d370afbc430f50b6d5da7fc08d8aa1c54fab9737fd51ad89d07f64e09cd8f53b' \
  'bundle_sha256=9589a205b26c3973e3c5af3c6377ac399589f484e4df7d16e0fae92d5d16e36d' \
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
  'proc_treatment=CAPSULE_EMPTY_BIND' \
  'proc_mount_source_object=CAPSULE_ROOT/proc' \
  'proc_mount_filesystem=CAPSULE_ROOT_FILESYSTEM' \
  'proc_mount_contents=empty' \
  'proc_mount_vfs_read_only=true' \
  'procfs_visible=false' \
  'proc_mount_identity=device+inode' \
  'typed_structural_mounts=/proc:CAPSULE_EMPTY_BIND' \
  'forbidden_mounts=/home+/root+/run+/var+/etc' \
  'bootstrap_case_count=8' \
  'authority_case_count=18' \
  'bootstrap_treatment_code=0' \
  'bootstrap_missing_incoming_code=226' \
  'bootstrap_missing_sys_code=226' \
  'bootstrap_missing_var_tmp_code=226' \
  'bootstrap_live_procfs_code=453' \
  'bootstrap_wrong_proc_source_code=454' \
  'bootstrap_writable_proc_bind_code=455' \
  'bootstrap_nonempty_proc_bind_code=456' \
  'allowed_syscalls=0,1,60,322' \
  'family_10_probe=personality_change' \
  'v9_materializable=false' \
  'v10_required_for_native=true' \
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
  'evidence_sha256=92e53a0d995b5511628bffe3d2b91602743cac35ef6c5712471697003d9f2885'; do
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
  'proc_treatment=CAPSULE_EMPTY_BIND' \
  'proc_mount_source_object=CAPSULE_ROOT/proc' \
  'proc_mount_filesystem=CAPSULE_ROOT_FILESYSTEM' \
  'proc_mount_contents=empty' \
  'proc_mount_vfs_read_only=true' \
  'procfs_visible=false' \
  'proc_mount_identity=device+inode' \
  'typed_structural_mounts=/proc:CAPSULE_EMPTY_BIND' \
  'forbidden_mounts=/home+/root+/run+/var+/etc' \
  'bootstrap_treatment_decision=SOUNIO_ROOT_BOOTSTRAP_ALLOW code=0 reason=identity-typed-inert-proc-bind stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_incoming_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=incoming-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_sys_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=sys-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_var_tmp_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=var-tmp-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_live_procfs_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=453 reason=live-procfs-visible stage=SEMANTICS_FROZEN' \
  'bootstrap_wrong_proc_source_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=454 reason=proc-source-identity-mismatch stage=SEMANTICS_FROZEN' \
  'bootstrap_writable_proc_bind_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=455 reason=proc-bind-writable stage=SEMANTICS_FROZEN' \
  'bootstrap_nonempty_proc_bind_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=456 reason=proc-bind-nonempty stage=SEMANTICS_FROZEN' \
  'v9_native_consumption=refused' \
  'native_executable_invoked=false' \
  'expected_results_source=Sounio' \
  'expected_results_encoded_in_shell=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'complete_effects=false' \
  'material_execution=false'; do
  require_line "$EVIDENCE" "$line"
done

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v10_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v10-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V10 gate failed'
[[ "$result" == *'families=12 treatments=12 sabotages=12 bootstrap_cases=8 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 bootstrap_live_procfs=DENY453 bootstrap_wrong_proc_source=DENY454 bootstrap_writable_proc_bind=DENY455 bootstrap_nonempty_proc_bind=DENY456 allowed_syscalls=4 authority_cases=18 complete=ALLOW current=DENY447 family10=personality_change'* ]] ||
  fail 'Sounio V10 bootstrap, surface, or decision classes drifted'
[[ "$result" == *'effective_mount_truth=DynamicUser+disconnected+strict+read-only'* ]] ||
  fail 'Sounio V10 effective mount truth drifted'
[[ "$result" == *'identity_typed_mounts=CAPSULE_EMPTY_BIND'* ]] ||
  fail 'Sounio V10 identity-typed mount contract drifted'
[[ "$result" == *'root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V10 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v10-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB effective_mount_truth=DynamicUser+disconnected+strict+read-only identity_typed_mounts=CAPSULE_EMPTY_BIND filesystem_authority=ROOT_HOST_MOUNTINFO systemd_mount=/run/systemd/incoming systemd_sys_mount=/sys systemd_var_tmp=/var/tmp bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 bootstrap_live_procfs=DENY453 bootstrap_wrong_proc_source=DENY454 bootstrap_writable_proc_bind=DENY455 bootstrap_nonempty_proc_bind=DENY456 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
