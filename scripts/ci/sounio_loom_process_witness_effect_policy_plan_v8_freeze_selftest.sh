#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v8.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v8-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v8-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V8.md \
  1a5cce39dd307af5f5a21ce8d5ace90994ad0d73fe911e121c3b7650ed37d533
expect_hash tools/loom/process_witness_effect_policy_plan_v8_main.sio \
  e512d0b465b170b9f9d50022f8eac5d021228fae6ba944d64b948c902831dc30
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v8.sh \
  0052ae9ccc7f1ba0cccd8b53e234829655830c9a16b761fbd4fd93c3abefb927
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v8_selftest.sh \
  d205a18f73e02dc3b7a3811007647b6c977cb3a3b57b217f83691558caa39063
expect_hash tools/loom/process_witness_effect_policy_plan_v7.freeze.v1 \
  cc7ca5a17babb43e145678879607b2804bdbfc66665f994b73f8649c86e420d9
expect_hash tools/loom/evidence/loom-process-witness-effect-root-v7-host-attempt-v1-20260828.txt \
  a4e9bab136988e6034a775e347ecc6642d7624dedfbf35b3e52d5b14236929bb
expect_hash "$EVIDENCE" \
  a14824e65d9bce3fffc9e9b94d0542b907842efa370113b82e15cd0df489a165

for line in \
  'schema=loom-process-witness-effect-policy-plan-v8-freeze-v1' \
  'stage=SEMANTICS_FROZEN' \
  'producing_language=Sounio' \
  'language_role=SEMANTIC_POLICY_PLAN' \
  'semantic_authority=Sounio' \
  'action=9025' \
  'garden_commit=bdfd472d0baffb8c85584fd0a953d5bc0f4eebc5' \
  'sounio_executable_commit=a8b6ca5ac7794f43d5b3b021a5affced783010d0' \
  'executable_sha256=9c9aae6b14a7e39674ae32e6c378325b0cd1243da587b3fdb4edd970406cc6e7' \
  'bundle_sha256=470e670b13602ca3131170d3e09d6fe7045b602867209d356707018471a18f0d' \
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
  'bootstrap_case_count=4' \
  'bootstrap_treatment_code=0' \
  'bootstrap_missing_incoming_code=226' \
  'bootstrap_missing_sys_code=226' \
  'bootstrap_missing_var_tmp_code=226' \
  'allowed_syscalls=0,1,60,322' \
  'family_10_probe=personality_change' \
  'v7_materializable=false' \
  'v8_required_for_native=true' \
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
  'evidence_sha256=a14824e65d9bce3fffc9e9b94d0542b907842efa370113b82e15cd0df489a165'; do
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
  'bootstrap_treatment_decision=SOUNIO_ROOT_BOOTSTRAP_ALLOW code=0 reason=systemd-mountpoints-present stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_incoming_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=incoming-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_sys_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=sys-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'bootstrap_missing_var_tmp_decision=SOUNIO_ROOT_BOOTSTRAP_DENY code=226 reason=var-tmp-mountpoint-absent stage=SEMANTICS_FROZEN' \
  'v7_native_consumption=refused' \
  'native_executable_invoked=false' \
  'expected_results_source=Sounio' \
  'expected_results_encoded_in_shell=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'complete_effects=false' \
  'material_execution=false'; do
  require_line "$EVIDENCE" "$line"
done

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v8_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v8-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V8 gate failed'
[[ "$result" == *'families=12 treatments=12 sabotages=12 bootstrap_cases=4 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change'* ]] ||
  fail 'Sounio V8 bootstrap, surface, or decision classes drifted'
[[ "$result" == *'root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V8 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v8-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s typed_file_bounds=effect-cell-16MiB+payload-1MiB+manifests-64KiB systemd_mount=/run/systemd/incoming systemd_sys_mount=/sys systemd_var_tmp=/var/tmp bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
