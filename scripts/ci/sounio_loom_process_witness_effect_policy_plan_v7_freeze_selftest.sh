#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_policy_plan_v7.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-policy-plan-v7-20260828.txt

fail() {
  printf 'sounio-loom-process-witness-effect-policy-plan-v7-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_hash tools/loom/GARDEN_PROCESS_WITNESS_EFFECT_POLICY_V7.md \
  8460371e45502db50501b4385bfe438677002a595601f161cc36cec5a95d80c4
expect_hash tools/loom/process_witness_effect_policy_plan_v7_main.sio \
  f5f0a881e2b2d55b06f1ed6a9120aad24df2b2460ea8913283d2d46d1ef8c81f
expect_hash scripts/dev/build_sounio_loom_process_witness_effect_policy_plan_v7.sh \
  b477e54b67ded3e36bbd20d49e43d34eddbd5e227a25698ffbddf06b756734af
expect_hash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v7_selftest.sh \
  b2658b1ff1cd212fc97330351bc065533b8cc6ca542db74d382fc3f0419bf9ea
expect_hash tools/loom/process_witness_effect_policy_plan_v6.freeze.v1 \
  6ec33f3554236e7ccf73f5b5c16a15ba8006705b83d9d62265a2cd8f94437d66
expect_hash tools/loom/evidence/loom-process-witness-effect-root-v6-host-attempt-v1-20260828.txt \
  04e296b2f27c54b598f6902ee936a1d6eee1354f046bafb38e0d928a6f1941b5
expect_hash "$EVIDENCE" \
  b40bcb79d233d22eec3dba371a42e54fe8af9007acbaa5707d7acd7beb3f1d99

for line in \
  'schema=loom-process-witness-effect-policy-plan-v7-freeze-v1' \
  'stage=SEMANTICS_FROZEN' \
  'producing_language=Sounio' \
  'language_role=SEMANTIC_POLICY_PLAN' \
  'semantic_authority=Sounio' \
  'action=9025' \
  'garden_commit=7a4802b6498bac29b6c8be209b9dcbb3c7929db2' \
  'sounio_executable_commit=a23704e565563c97ac584b098aa878dd63d8280e' \
  'executable_sha256=edb568290af8e812d0b02e27ac301a4dd1afcbeb322302f41d6da82552df1c92' \
  'bundle_sha256=9a774718bf67f1ef6b6af9b4758a89239d9fa38e4abb981b8452f5aac90828d2' \
  'object_boundary=IMMUTABLE_ROOT_MOUNT_NAMESPACE' \
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
  'v6_materializable=false' \
  'v7_required_for_native=true' \
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
  'evidence_sha256=b40bcb79d233d22eec3dba371a42e54fe8af9007acbaa5707d7acd7beb3f1d99'; do
  require_line "$MANIFEST" "$line"
done

for line in \
  'hardware_arch=x86_64' \
  'hardware_kernel=Linux_7.0.2-5-pve' \
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
  'v6_native_consumption=refused' \
  'native_executable_invoked=false' \
  'expected_results_source=Sounio' \
  'expected_results_encoded_in_shell=false' \
  'root_treatment=false' \
  'bootstrap_sabotage=false' \
  'complete_effects=false' \
  'material_execution=false'; do
  require_line "$EVIDENCE" "$line"
done

result="$(bash scripts/ci/sounio_loom_process_witness_effect_policy_plan_v7_selftest.sh)"
[[ "$result" == sounio-loom-process-witness-effect-policy-plan-v7-selftest:\ PASS* ]] ||
  fail 'source-fresh Sounio V7 gate failed'
[[ "$result" == *'families=12 treatments=12 sabotages=12 bootstrap_cases=4 bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=4 authority_cases=14 complete=ALLOW current=DENY447 family10=personality_change'* ]] ||
  fail 'Sounio V7 bootstrap, surface, or decision classes drifted'
[[ "$result" == *'root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'Sounio V7 gate promoted beyond evidence'

printf 'sounio-loom-process-witness-effect-policy-plan-v7-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 policy_manifest_sha256=%s systemd_mount=/run/systemd/incoming systemd_sys_mount=/sys systemd_var_tmp=/var/tmp bootstrap_treatment=ALLOW bootstrap_missing_incoming=DENY226 bootstrap_missing_sys=DENY226 bootstrap_missing_var_tmp=DENY226 allowed_syscalls=0+1+60+322 families=12 treatments=12 sabotages=12 root_treatment=false bootstrap_sabotage=false material_coverage=false complete_effects=false material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
