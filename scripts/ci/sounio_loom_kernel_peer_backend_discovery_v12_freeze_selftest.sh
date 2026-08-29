#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/kernel_peer_backend_discovery_v12.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-backend-discovery-v12-host-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-backend-discovery-v12-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

file_hash() {
  sha256sum "$1" | cut -d ' ' -f 1
}

stream_hash() {
  sha256sum | cut -d ' ' -f 1
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in $path"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

field() {
  record_field "$MANIFEST" "$1"
}

evidence_field() {
  record_field "$EVIDENCE" "$1"
}

expect_field() {
  local key="$1" expected="$2" actual
  actual="$(field "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "$key drifted: expected=$expected actual=$actual"
}

expect_evidence() {
  local key="$1" expected="$2" actual
  actual="$(evidence_field "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "evidence $key drifted: expected=$expected actual=$actual"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is absent or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'backend evidence is absent or linked'

expect_field schema loom-kernel-peer-backend-discovery-v12-freeze-v1
expect_field stage BACKEND_DISCOVERY_FROZEN_V12
expect_field semantic_authority Sounio
expect_field action 9025
expect_field semantic_manifest_sha256 daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30
expect_field material_source_commit 1a7006b93e8a72baa035d0272e35516cf0d4589e
expect_field producing_language C++20
expect_field language_role MATERIAL_DISCOVERY
expect_field transitory true
expect_field source_sha256 f953107e08e5e2a0f010f78bcfe9c99913569bfed738870e42a65f00fa0f9c9b
expect_field profile_sha256 323eac775e75bfd453d694354f92ca481652b6e0f2d31116f48a2c33d451e2f3
expect_field build_script_sha256 1a867cc4b0e2d6307d0fdb01f86c250e7e47a3f5539514d6a2d045da42c2f004
expect_field selftest_sha256 6a18a544d2f0da2d815a944ab75bbeb79d4ac6dfa2391305632669074d6d3b31
expect_field host_gate_sha256 6aa8a71bd80bf3b37825a816ca7dd9da8c687998f586e64c5a3ff06ab2fa4da3
expect_field host_probe_sha256 887d4c66563fcd6cd00b89297ea9184fe659040aed5a36bbfad35026558f1b4e
expect_field binary_sha256 10c8a662db184c00f18d6cc59240a36a1a24137502dc78a76c6d12d21a107838
expect_field evidence_sha256 3998545970aa167b8239e9006057dd8d4d26b5df2be145bea9ead5caab26fa3f
expect_field toolchain c++_Ubuntu_13.3.0-6ubuntu2~24.04.1
expect_field hardware_host t560-proxmox
expect_field hardware_arch x86_64
expect_field kernel 7.0.2-5-pve
expect_field transport kubectl+hostPID+nsenter
expect_field backend AppArmor
expect_field active_lsm lockdown,capability,yama,apparmor,ima,evm
expect_field bpf_lsm_config true
expect_field bpf_lsm_active false
expect_field btf_sha256 5575d7ba5d53ae6b72c45586dca83cac6884d3a3f1000ac9d58c509c432b5aa0
expect_field security_task_prlimit_hook true
expect_field apparmor_task_prlimit_hook false
expect_field same_kuid true
expect_field all_four_uid_slots_equal true
expect_field attacker_syscalls_open true
expect_field target_label_enforced true
expect_field target_cgroup_distinct true
expect_field attacker_cgroup_distinct true
expect_field signal_operation kill_SIGTERM
expect_field signal_observation REFUSED_BEFORE_EFFECT
expect_field signal_errno 13
expect_field prlimit_operation prlimit64
expect_field prlimit_observation EFFECT_COMPLETED
expect_field prlimit_errno 0
expect_field prlimit_prior_soft 1024
expect_field prlimit_target_soft 768
expect_field probed_operations 2
expect_field frozen_operations 10
expect_field policy_extinct true
expect_field cgroups_extinct true
expect_field backend_discovery true
expect_field backend_discovery_decision BACKEND_INCOMPLETE
expect_field backend_candidate_complete false
expect_field stop_rule no-admissible-receiver-mediator
expect_field semantic_order_preserved true
expect_field semantic_results_encoded_in_cpp false
expect_field python_executed false
expect_field rust_executed false
expect_field native_discovery_bytes_created true
expect_field native_material_matrix_bytes_created false
for boundary in material_peer_matrix same_uid_peer_isolation material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done
expect_field action_9025_decision DENY451
expect_field next_stage DEDICATED_BPF_LSM_HOST_REQUIRED

SOURCE_COMMIT="$(field material_source_commit)"
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'material discovery commit is absent'
for pair in source_path:source_sha256 profile_path:profile_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256 host_gate_path:host_gate_sha256 host_probe_path:host_probe_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the material discovery commit"
done
if git cat-file -e "$SOURCE_COMMIT:tools/loom/src/loom_kernel_peer_authority_v12.cpp" 2>/dev/null; then
  fail 'full V12 material matrix bytes existed during backend discovery'
fi

[[ "$(file_hash "$(field semantic_manifest_path)")" == "$(field semantic_manifest_sha256)" ]] ||
  fail 'V12 semantic manifest drifted'
[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'backend evidence hash drifted'

expect_evidence schema loom-kernel-peer-backend-discovery-v12-evidence-v1
expect_evidence stage BACKEND_DISCOVERY_V12
expect_evidence semantic_authority Sounio
expect_evidence action 9025
expect_evidence semantic_manifest_sha256 "$(field semantic_manifest_sha256)"
expect_evidence material_source_commit "$SOURCE_COMMIT"
expect_evidence producing_language C++20
expect_evidence language_role MATERIAL_DISCOVERY
expect_evidence source_sha256 "$(field source_sha256)"
expect_evidence profile_sha256 "$(field profile_sha256)"
expect_evidence binary_sha256 "$(field binary_sha256)"
expect_evidence hardware_host t560-proxmox
expect_evidence kernel 7.0.2-5-pve
expect_evidence backend AppArmor
expect_evidence bpf_lsm_config true
expect_evidence bpf_lsm_active false
expect_evidence apparmor_task_prlimit_hook false
expect_evidence same_kuid true
expect_evidence all_four_uid_slots_equal true
expect_evidence signal_observation REFUSED_BEFORE_EFFECT
expect_evidence signal_errno 13
expect_evidence signal_target_seen 0
expect_evidence prlimit_observation EFFECT_COMPLETED
expect_evidence prlimit_errno 0
expect_evidence prlimit_prior_soft 1024
expect_evidence prlimit_observed_soft 768
expect_evidence prlimit_target_soft 768
expect_evidence policy_extinct true
expect_evidence cgroups_extinct true
expect_evidence backend_discovery_decision BACKEND_INCOMPLETE
expect_evidence backend_candidate_complete false
expect_evidence stop_rule no-admissible-receiver-mediator
expect_evidence native_material_matrix_bytes_created false
expect_evidence material_peer_matrix false
expect_evidence same_uid_peer_isolation false
expect_evidence action_9025_decision DENY451
expect_evidence next_stage DEDICATED_BPF_LSM_HOST_REQUIRED

[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'host command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'host result hash drifted'
[[ "$(printf '%s' "$(evidence_field local_selftest_command)" | stream_hash)" == "$(evidence_field local_selftest_command_sha256)" ]] ||
  fail 'local selftest command hash drifted'
[[ "$(printf '%s' "$(evidence_field local_selftest_result)" | stream_hash)" == "$(evidence_field local_selftest_result_sha256)" ]] ||
  fail 'local selftest result hash drifted'
[[ "$(evidence_field result)" == *"host_output_sha256=$(evidence_field host_output_sha256)"* ]] ||
  fail 'transport receipt does not bind the host output'

semantic_result="$(bash scripts/ci/sounio_loom_kernel_peer_authority_plan_v12_freeze_selftest.sh 2>/dev/null)"
[[ "$semantic_result" == sounio-loom-kernel-peer-authority-plan-v12-freeze-selftest:\ PASS* ]] ||
  fail 'frozen V12 Sounio semantics failed'
[[ "$semantic_result" == *'same_kuid_required=true attacker_syscalls_open_required=true receiver_side_required=true'* ]] ||
  fail 'V12 receiver-side semantic boundary drifted'

local_result="$(bash "$(field selftest_path)" 2>/dev/null)"
[[ "$local_result" == "$(evidence_field local_selftest_result)" ]] ||
  fail 'source-fresh C++20 discovery gate drifted'
[[ "$local_result" == *'semantic_results_encoded=false python_executed=false rust_executed=false'* ]] ||
  fail 'material discovery crossed the language-authority boundary'

printf 'sounio-loom-kernel-peer-backend-discovery-v12-freeze-selftest: PASS semantic_authority=Sounio action=9025 material_producer=C++20 material_role=MATERIAL_DISCOVERY manifest_sha256=%s evidence_sha256=%s host=t560-proxmox kernel=7.0.2-5-pve backend=AppArmor bpf_lsm_config=true bpf_lsm_active=false apparmor_task_prlimit_hook=false same_kuid=true all_four_uid_slots_equal=true attacker_syscalls_open=true signal=REFUSED_BEFORE_EFFECT/EACCES prlimit64=EFFECT_COMPLETED/0 prior_soft=1024 target_soft=768 policy_extinct=true cgroups_extinct=true backend_discovery=true backend_discovery_decision=BACKEND_INCOMPLETE backend_candidate_complete=false stop_rule=no-admissible-receiver-mediator native_discovery_bytes_created=true native_material_matrix_bytes_created=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=DEDICATED_BPF_LSM_HOST_REQUIRED\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
