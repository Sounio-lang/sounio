#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
MANIFEST=tools/loom/kernel_peer_authority_plan_v13.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-authority-plan-v13-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-authority-plan-v13-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}
file_hash() { sha256sum "$1" | cut -d ' ' -f 1; }
stream_hash() { sha256sum | cut -d ' ' -f 1; }
record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in $path"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}
field() { record_field "$MANIFEST" "$1"; }
evidence_field() { record_field "$EVIDENCE" "$1"; }
expect_field() {
  local actual
  actual="$(field "$1")"
  [[ "$actual" == "$2" ]] || fail "$1 drifted: expected=$2 actual=$actual"
}
expect_evidence() {
  local actual
  actual="$(evidence_field "$1")"
  [[ "$actual" == "$2" ]] || fail "evidence $1 drifted: expected=$2 actual=$actual"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" && -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] ||
  fail 'V13 freeze inputs are absent or linked'
expect_field schema loom-kernel-peer-authority-plan-v13-freeze-v1
expect_field stage SEMANTICS_FROZEN_V13
expect_field producing_language Sounio
expect_field language_role SEMANTIC_POLICY_PLAN
expect_field semantic_authority Sounio
expect_field action 9025
expect_field garden_commit e876c5de26477f4772b75ebf76dbabf9cf5be2f9
expect_field garden_sha256 8c5f456b3979517ab42a62050bf07c6c0e66db9c79b5b55b9d244a0d715289e9
expect_field sounio_executable_commit e876c5de26477f4772b75ebf76dbabf9cf5be2f9
expect_field source_sha256 3545f75dca264b4378ab4cf633a686ffcde5152cb02ac18b74ab00192baed7f0
expect_field build_script_sha256 9a6d411b6f189264476868fcb7d1cadaa792f2f95f5ae6d5819bd4dd9f9b715e
expect_field selftest_sha256 fbe4c135664ae5772eddbdba49d27b14fa47621de06f6df2ff9b4a9a2dbc367c
expect_field executable_sha256 252f605a16a5419a3b980a1b5480f1dddfe8bffe6c6dae837a5b1f9400a05732
expect_field bundle_sha256 44a3052926f0958ee970fe21c772276102d6ff9069907f5f80f8b5aa5063ae87
expect_field bundle_line_count 90
expect_field bundle_byte_count 20733
expect_field output_bound_bytes 65536
expect_field output_bound_enforced true
expect_field v12_semantic_manifest_sha256 daaf9b056bde02078163849fdcc0f0b3acb96a357741c70e3d23c688c2bd0b30
expect_field v12_matrix_manifest_sha256 1692782657cbe6fe7a548b6f11d4d542d24fe05569686d536a4c69af0775cd75
expect_field v12_falsification_manifest_sha256 d4b3cdc1dfc6c139538cffecddca60fe34498908b38a2476a7beba8e7e60db7e
expect_field action_9025_manifest_sha256 c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
expect_field evidence_sha256 24694d1826263dc0026992896ff29f5b360582d3987cf771d80d244bcd136bcf
expect_field correction_predecessor V12
expect_field falsified_vertex DUMPABLE_ONLY_CONTROL
expect_field falsified_operation 9
expect_field frozen_expected REFUSED_BEFORE_EFFECT
expect_field material_observed EFFECT_COMPLETED
expect_field corrected_expected EFFECT_COMPLETED
expect_field correction_witness LIMIT_CHANGED_RESTORED
expect_field retrospective_rewrite false
expect_field v12_hypothesis_falsified true
expect_field principal_vertices 5
expect_field operations 10
expect_field observations 50
expect_field decisive_pairs 10
expect_field receiver_properties 7
expect_field sabotage_twins 5
expect_field expected_refused 25
expect_field expected_completed 15
expect_field expected_unavailable 10
expect_field expected_crossed 0
expect_field treatment_refused 10
expect_field mediator_removed_completed 10
expect_field distinct_kuid_control 10
expect_field caller_seccomp_invalid_proof 10
expect_field dumpable_partial_completed 5
expect_field dumpable_partial_refused 5
for enabled in same_kuid_required all_four_kernel_uid_slots_equal attacker_syscalls_open_required receiver_side_required distinct_kuid_is_not_same_uid_proof caller_seccomp_is_not_receiver_proof pid_secrecy_is_not_isolation deterministic_rebuild garden_v13 sounio_executable_v13 semantics_frozen_v13; do
  expect_field "$enabled" true
done
expect_field hash_binding invariant_sha256+delta_sha256+attempt_sha256+target_sha256+extinction_sha256
expect_field causal_pair TREATMENT+MEDIATOR_REMOVED
expect_field only_permitted_delta mediator_presence+policy_hash
expect_field sabotage_1 TREATMENT+REMOVE_MEDIATOR+ALL_COMPLETED
expect_field sabotage_2 MEDIATOR_REMOVED+INSTALL_MEDIATOR+ALL_REFUSED
expect_field sabotage_3 DISTINCT_KUID_CONTROL+COLLAPSE_TO_SAME_KUID+CREDENTIAL_REFUSAL_DISAPPEARS
expect_field sabotage_4 CALLER_SECCOMP_CONTROL+OPEN_CALLER_FILTER+UNAVAILABILITY_DISAPPEARS
expect_field sabotage_5 DUMPABLE_ONLY_CONTROL+SET_DUMPABLE_ONE+FIVE_PARTIAL_REFUSALS_COMPLETE
expect_field complete_hypothesis ACTION_9025_ALLOW
expect_field current_v13_prematerial ACTION_9025_DENY451
expect_field coverage_missing ACTION_9025_DENY447
expect_field shell_expected_results false
expect_field python_executed false
expect_field rust_executed false
for boundary in native_v13_bytes_created controls_executed material_peer_matrix same_uid_peer_isolation material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done
expect_field action_9025_decision DENY451
expect_field next_stage MATERIAL_CONTROL_MATRIX_V13

GARDEN_COMMIT="$(field garden_commit)"
SOUNIO_COMMIT="$(field sounio_executable_commit)"
git cat-file -e "${GARDEN_COMMIT}^{commit}" || fail 'Garden commit is absent'
git cat-file -e "${SOUNIO_COMMIT}^{commit}" || fail 'Sounio executable commit is absent'
for pair in garden_path:garden_sha256 source_path:source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  path="$(field "$path_key")"; expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOUNIO_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the V13 Sounio commit"
done
if git cat-file -e "$SOUNIO_COMMIT:tools/loom/src/loom_kernel_peer_controls_init_v13.cpp" 2>/dev/null; then
  fail 'native V13 control bytes existed before semantic freeze'
fi
for pair in v12_semantic_manifest_path:v12_semantic_manifest_sha256 v12_matrix_manifest_path:v12_matrix_manifest_sha256 v12_falsification_manifest_path:v12_falsification_manifest_sha256 action_9025_manifest_path:action_9025_manifest_sha256 evidence_path:evidence_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  [[ "$(file_hash "$(field "$path_key")")" == "$(field "$hash_key")" ]] ||
    fail "$(field "$path_key") drifted"
done
[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'V13 evidence hash drifted'

expect_evidence schema loom-kernel-peer-authority-plan-v13-evidence-v1
expect_evidence stage SOUNIO_EXECUTABLE_V13
expect_evidence producing_language Sounio
expect_evidence language_role SEMANTIC_POLICY_PLAN
expect_evidence semantic_authority Sounio
expect_evidence action 9025
expect_evidence garden_commit "$GARDEN_COMMIT"
expect_evidence sounio_executable_commit "$SOUNIO_COMMIT"
for key in source_sha256 build_script_sha256 selftest_sha256 executable_sha256 bundle_sha256 bundle_line_count bundle_byte_count output_bound_bytes v12_semantic_manifest_sha256 v12_matrix_manifest_sha256 v12_falsification_manifest_sha256 action_9025_manifest_sha256 correction_predecessor falsified_vertex falsified_operation frozen_expected material_observed corrected_expected correction_witness retrospective_rewrite v12_hypothesis_falsified principal_vertices operations observations decisive_pairs receiver_properties sabotage_twins expected_refused expected_completed expected_unavailable expected_crossed treatment_refused mediator_removed_completed distinct_kuid_control caller_seccomp_invalid_proof dumpable_partial_completed dumpable_partial_refused same_kuid_required all_four_kernel_uid_slots_equal attacker_syscalls_open_required receiver_side_required distinct_kuid_is_not_same_uid_proof caller_seccomp_is_not_receiver_proof pid_secrecy_is_not_isolation hash_binding causal_pair only_permitted_delta sabotage_1 sabotage_2 sabotage_3 sabotage_4 sabotage_5 complete_hypothesis current_v13_prematerial coverage_missing shell_expected_results python_executed rust_executed; do
  expect_evidence "$key" "$(field "$key")"
done
expect_evidence output_bounded true
expect_evidence deterministic true
expect_evidence garden_v13 true
expect_evidence sounio_executable_v13 true
expect_evidence semantics_frozen_v13 false
expect_evidence native_v13_bytes_created false
expect_evidence controls_executed false
expect_evidence material_peer_matrix false
expect_evidence same_uid_peer_isolation false
expect_evidence action_9025_decision DENY451
expect_evidence claim_ready false
expect_evidence next_stage SEMANTICS_FREEZE_V13
[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'V13 command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'V13 result hash drifted'

falsifier_result="$(bash scripts/ci/sounio_loom_kernel_peer_dumpable_prlimit_falsification_v12_freeze_selftest.sh 2>/dev/null)"
[[ "$falsifier_result" == *'material_observed=EFFECT_COMPLETED completion=LIMIT_CHANGED_RESTORED v12_hypothesis_falsified=true'* ]] ||
  fail 'frozen V12 counterexample drifted'
v13_result="$(
  SOUNIO_LOOM_KERNEL_PEER_PLAN_V13_PREMATERIAL_COMMIT="$SOUNIO_COMMIT" \
    bash "$(field selftest_path)" 2>/dev/null
)"
[[ "$v13_result" == "$(evidence_field result)" ]] ||
  fail 'source-fresh V13 Sounio plan drifted'
[[ "$v13_result" == *'refused=25 completed=15 unavailable=10'* &&
   "$v13_result" == *'dumpable_partial=5+5 v12_hypothesis_falsified=true'* ]] ||
  fail 'V13 corrected control matrix drifted'

printf 'sounio-loom-kernel-peer-authority-plan-v13-freeze-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 manifest_sha256=%s evidence_sha256=%s principal_vertices=5 operations=10 observations=50 decisive_pairs=10 receiver_properties=7 sabotage_twins=5 refused=25 completed=15 unavailable=10 dumpable_partial=5+5 v12_hypothesis_falsified=true retrospective_rewrite=false causal_pair=TREATMENT+MEDIATOR_REMOVED same_kuid_required=true attacker_syscalls_open_required=true receiver_side_required=true output_bound_bytes=65536 output_bound_enforced=true deterministic_rebuild=true shell_expected_results=false python_executed=false rust_executed=false garden_v13=true sounio_executable_v13=true semantics_frozen_v13=true native_v13_bytes_created=false controls_executed=false material_peer_matrix=false same_uid_peer_isolation=false action_9025=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=MATERIAL_CONTROL_MATRIX_V13\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
