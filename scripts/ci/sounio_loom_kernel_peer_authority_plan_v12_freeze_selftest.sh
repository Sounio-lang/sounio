#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/kernel_peer_authority_plan_v12.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-authority-plan-v12-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-authority-plan-v12-freeze-selftest: FAIL: %s\n' "$*" >&2
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
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'V12 evidence is absent or linked'

expect_field schema loom-kernel-peer-authority-plan-v12-freeze-v1
expect_field stage SEMANTICS_FROZEN_V12
expect_field producing_language Sounio
expect_field language_role SEMANTIC_POLICY_PLAN
expect_field semantic_authority Sounio
expect_field action 9025
expect_field garden_commit 3828ecca8655d97c17c40ae53525614f83293ba5
expect_field garden_sha256 f426b8b1c0b6dd0233345225ff56bce2c65fb95d8e227076fdb5611c9808bec2
expect_field sounio_executable_commit 62060eeb7e126810383d8b48e24b6350cf4adb3b
expect_field source_sha256 42f71e0c77b5997bc35ab5df73e50108ee41b444d59b2ed736e015b14864b2d9
expect_field build_script_sha256 2d4fec08c0edae4cef4d60b9398f3eac5955d223d7e876450e87aa51f2a80744
expect_field selftest_sha256 ec94b9ca6578aa986ca9984e2ef8409cabc278bb47e91734377fdce23b0b740f
expect_field executable_sha256 020bbaa3f24b7f6e82383ce0ce235a59ceb825ce9458b19708a625675f7604c5
expect_field bundle_sha256 94d22ea974168f41200684c60da5e673a4afebb7e34f0b4cd8f228d0303e7b97
expect_field bundle_line_count 84
expect_field bundle_byte_count 19852
expect_field output_bound_bytes 65536
expect_field output_bound_enforced true
expect_field v11_judgment_manifest_sha256 f227cca70aa30351517403e13f60143c683bb86d445320661d68c08317c81b89
expect_field v11_judgment_evidence_sha256 4aa5704fe529ee93c88992a630976395b49a28ed13189af9d7a07aeb7ecc4c64
expect_field action_9025_manifest_sha256 c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
expect_field evidence_sha256 b2f0ba19a7a8568006b18a0ab264b9ce9d2d0e84ae7c4e86e51dd87c283867b3
expect_field principal_vertices 5
expect_field operations 10
expect_field observations 50
expect_field decisive_pairs 10
expect_field receiver_properties 7
expect_field expected_refused 26
expect_field expected_completed 14
expect_field expected_unavailable 10
expect_field expected_crossed 0
expect_field treatment_refused 10
expect_field mediator_removed_completed 10
expect_field distinct_kuid_control 10
expect_field caller_seccomp_invalid_proof 10
expect_field dumpable_partial_completed 4
expect_field dumpable_partial_refused 6
expect_field same_kuid_required true
expect_field all_four_kernel_uid_slots_equal true
expect_field attacker_syscalls_open_required true
expect_field receiver_side_required true
expect_field distinct_kuid_is_not_same_uid_proof true
expect_field caller_seccomp_is_not_receiver_proof true
expect_field pid_secrecy_is_not_isolation true
expect_field hash_binding invariant_sha256+delta_sha256+attempt_sha256+target_sha256+extinction_sha256
expect_field causal_pair TREATMENT+MEDIATOR_REMOVED
expect_field only_permitted_delta mediator_presence+policy_hash
expect_field complete_hypothesis ACTION_9025_ALLOW
expect_field current_v11 ACTION_9025_DENY451
expect_field coverage_missing ACTION_9025_DENY447
expect_field pre_freeze_attempt_1 REFUSED_OUTPUT_BOUND_CROSSED
expect_field pre_freeze_attempt_1_source_sha256 496761eb0d081a63a0a2d22ad6ab05957e94dbd7912cacdd06960f58eb8f49fd
expect_field pre_freeze_attempt_1_observation long_direct_action_frame_repeated_by_lean_single_output
expect_field pre_freeze_attempt_1_observed_bytes_gt 1200000000
expect_field pre_freeze_correction short_typed_action_facts_plus_bounded_mechanical_frame_adapter
expect_field deterministic_rebuild true
expect_field shell_expected_results false
expect_field python_executed false
expect_field rust_executed false
expect_field garden_v12 true
expect_field sounio_executable_v12 true
expect_field semantics_frozen_v12 true
for boundary in backend_discovery native_v12_bytes_created material_peer_matrix same_uid_peer_isolation material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done
expect_field action_9025_decision DENY451
expect_field next_stage BACKEND_DISCOVERY_V12

GARDEN_COMMIT="$(field garden_commit)"
SOUNIO_COMMIT="$(field sounio_executable_commit)"
git cat-file -e "${GARDEN_COMMIT}^{commit}" || fail 'Garden commit is absent'
git cat-file -e "${SOUNIO_COMMIT}^{commit}" || fail 'Sounio executable commit is absent'

for pair in garden_path:garden_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$GARDEN_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the Garden commit"
done

for pair in source_path:source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOUNIO_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the Sounio executable commit"
done

if git cat-file -e "$SOUNIO_COMMIT:tools/loom/src/loom_kernel_peer_authority_v12.cpp" 2>/dev/null; then
  fail 'native V12 bytes existed before semantic freeze'
fi

for pair in v11_judgment_manifest_path:v11_judgment_manifest_sha256 v11_judgment_evidence_path:v11_judgment_evidence_sha256 action_9025_manifest_path:action_9025_manifest_sha256 evidence_path:evidence_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  [[ "$(file_hash "$(field "$path_key")")" == "$(field "$hash_key")" ]] ||
    fail "$(field "$path_key") drifted"
done

expect_evidence schema loom-kernel-peer-authority-plan-v12-evidence-v1
expect_evidence stage SOUNIO_EXECUTABLE_V12
expect_evidence producing_language Sounio
expect_evidence language_role SEMANTIC_POLICY_PLAN
expect_evidence semantic_authority Sounio
expect_evidence action 9025
expect_evidence garden_commit "$GARDEN_COMMIT"
expect_evidence sounio_executable_commit "$SOUNIO_COMMIT"
expect_evidence source_sha256 "$(field source_sha256)"
expect_evidence selftest_sha256 "$(field selftest_sha256)"
expect_evidence bundle_sha256 "$(field bundle_sha256)"
expect_evidence output_bound_bytes 65536
expect_evidence pre_freeze_attempt_1 REFUSED_OUTPUT_BOUND_CROSSED
expect_evidence pre_freeze_attempt_1_observed_bytes_gt 1200000000
expect_evidence output_bounded true
expect_evidence deterministic true
expect_evidence shell_expected_results false
expect_evidence python_executed false
expect_evidence rust_executed false
expect_evidence garden_v12 true
expect_evidence sounio_executable_v12 true
expect_evidence semantics_frozen_v12 false
expect_evidence backend_discovery false
expect_evidence native_v12_bytes_created false
expect_evidence material_peer_matrix false
expect_evidence same_uid_peer_isolation false
expect_evidence action_9025_decision DENY451
expect_evidence claim_ready false

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'V12 evidence hash drifted'
[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'V12 command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'V12 result hash drifted'

v11_result="$(bash scripts/ci/sounio_loom_process_witness_effect_material_judgment_v11_freeze_selftest.sh 2>/dev/null)"
[[ "$v11_result" == sounio-loom-process-witness-effect-material-judgment-v11-freeze-selftest:\ PASS* ]] ||
  fail 'frozen V11 material judgment gate failed'
[[ "$v11_result" == *'action_9025=DENY451 reason=same-uid-peer-isolation-absent'* ]] ||
  fail 'V11 same-UID denial boundary drifted'

v12_result="$(
  SOUNIO_LOOM_KERNEL_PEER_PLAN_V12_PREMATERIAL_COMMIT="$SOUNIO_COMMIT" \
    bash "$(field selftest_path)" 2>/dev/null
)"
[[ "$v12_result" == "$(evidence_field result)" ]] ||
  fail 'source-fresh V12 Sounio plan drifted'
[[ "$v12_result" == *'observations=50 decisive_pairs=10 receiver_properties=7 refused=26 completed=14 unavailable=10'* ]] ||
  fail 'V12 causal matrix drifted'
[[ "$v12_result" == *'output_bounded=true shell_expected_results=false python_executed=false rust_executed=false'* ]] ||
  fail 'V12 oracle or output boundary drifted'

printf 'sounio-loom-kernel-peer-authority-plan-v12-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 manifest_sha256=%s evidence_sha256=%s principal_vertices=5 operations=10 observations=50 decisive_pairs=10 receiver_properties=7 refused=26 completed=14 unavailable=10 causal_pair=TREATMENT+MEDIATOR_REMOVED same_kuid_required=true attacker_syscalls_open_required=true receiver_side_required=true output_bound_bytes=65536 output_bound_enforced=true deterministic_rebuild=true shell_expected_results=false python_executed=false rust_executed=false garden_v12=true sounio_executable_v12=true semantics_frozen_v12=true backend_discovery=false native_v12_bytes_created=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=BACKEND_DISCOVERY_V12\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
