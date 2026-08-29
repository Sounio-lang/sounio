#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"

MANIFEST=tools/loom/process_witness_effect_material_judgment_v11.freeze.v1
EVIDENCE=tools/loom/evidence/loom-process-witness-effect-material-judgment-v11-20260829.txt

fail() {
  printf 'sounio-loom-process-witness-effect-material-judgment-v11-freeze-selftest: FAIL: %s\n' "$*" >&2
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
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'judgment evidence is absent or linked'

expect_field schema loom-process-witness-effect-material-judgment-v11-freeze-v1
expect_field stage ACTION_9025_JUDGMENT_FROZEN
expect_field producing_language Sounio
expect_field language_role SEMANTIC_AUTHORITY
expect_field semantic_authority Sounio
expect_field action 9025
expect_field source_commit ceacb88cc1a9490aedc0f631b31cd4651f1fcd10
expect_field source_sha256 48f49e4da369bc2704523692db99966d8647c64f5449b993236288d411aa2017
expect_field build_script_sha256 e5052548d1ac2b1712696b38cccc888824b09273f3f21e9eb5bdde57ef615527
expect_field selftest_sha256 a46ca1ad751bdc3e9eafd55d3bab337fd896467b04f3fda3443029a2cab4f764
expect_field executable_sha256 c1b7a5117eda986ac2c86c6ad7a60eff1215b31f3c8dabdbe024887b453bfdf6
expect_field fixture_bundle_sha256 0ec00121b7347260574747abb54743b032f0aef3724bd074d6854025007fc308
expect_field policy_manifest_sha256 adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c
expect_field material_evidence_sha256 57bc9730b0b5662a548af8271bdca6ed1651c5684c7999182e6c3d6e6ad53738
expect_field certificate_bundle_sha256 1c92fcd7c97a5df4e8316b722f769f6777ea5979edcd09c207e88f9930f8d3dd
expect_field material_command_result_sha256 f0a31473d9e530de862a1b25911ecb6df0c3b69de343961899208dcb42c5155d
expect_field host_principal_evidence_sha256 01c63677ab36668c17fe4454f9792c4595350d0d091ba21407a0e5061c36c7f7
expect_field grant_stack_sha256 1d7b8a3b1dfba1d1f9e60b5392cdf7e57a8d085cd872659feea5e333e43759b1
expect_field action_9025_manifest_sha256 c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
expect_field evidence_sha256 4aa5704fe529ee93c88992a630976395b49a28ed13189af9d7a07aeb7ecc4c64
expect_field material_receipt_decision ALLOW
expect_field action_9025_decision DENY451
expect_field action_9025_reason same-uid-peer-isolation-absent
expect_field action_9025_decision_sha256 345af1bca9022a9ecc5b1c06bbd379bc9135b1efc06fce5e874addcc49b82466
expect_field same_uid_peer_isolation false
expect_field causal_rule peer-isolation-truth
expect_field causal_rule_sabotage ACTION_9025_ALLOW
expect_field causal_sabotage PASS
expect_field negative_material DENY447x4
expect_field authority_laundering DENY459
expect_field python_oracle DENY459
expect_field python_executed false
expect_field evidence_substitution DENY450
expect_field malformed DENY424
expect_field deterministic_rebuild true
expect_field semantics_frozen true
expect_field material_hypercube true
for boundary in material_coverage complete_effects material_execution production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done
expect_field action_9025_judged true
expect_field next_stage V12_KERNEL_PEER_AUTHORITY

SOURCE_COMMIT="$(field source_commit)"
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'source commit is absent'
for pair in source_path:source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the source commit"
done

for pair in policy_manifest_path:policy_manifest_sha256 material_evidence_path:material_evidence_sha256 host_principal_evidence_path:host_principal_evidence_sha256 grant_stack_path:grant_stack_sha256 action_9025_manifest_path:action_9025_manifest_sha256 evidence_path:evidence_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  [[ "$(file_hash "$(field "$path_key")")" == "$(field "$hash_key")" ]] ||
    fail "$(field "$path_key") drifted"
done

expect_evidence schema loom-process-witness-effect-material-judgment-v11-evidence-v1
expect_evidence stage ACTION_9025_JUDGMENT
expect_evidence producing_language Sounio
expect_evidence language_role SEMANTIC_AUTHORITY
expect_evidence semantic_authority Sounio
expect_evidence action 9025
expect_evidence source_commit "$SOURCE_COMMIT"
expect_evidence source_sha256 "$(field source_sha256)"
expect_evidence material_evidence_sha256 "$(field material_evidence_sha256)"
expect_evidence certificate_bundle_sha256 "$(field certificate_bundle_sha256)"
expect_evidence material_receipt_decision ALLOW
expect_evidence action_9025_decision DENY451
expect_evidence action_9025_reason same-uid-peer-isolation-absent
expect_evidence same_uid_peer_isolation false
expect_evidence same_uid_control_signal ALLOWED
expect_evidence same_uid_control_pidfd_signal ALLOWED
expect_evidence causal_rule peer-isolation-truth
expect_evidence causal_rule_sabotage ACTION_9025_ALLOW
expect_evidence causal_sabotage PASS
expect_evidence python_oracle DENY459
expect_evidence python_executed false
expect_evidence material_hypercube true
expect_evidence material_coverage false
expect_evidence action_9025_judged true
expect_evidence next_stage V12_KERNEL_PEER_AUTHORITY

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] ||
  fail 'judgment evidence hash drifted'
[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'judgment command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'judgment result hash drifted'
[[ "$(printf '%s' 'SOUNIO_EFFECT_CLOSURE_DENY code=451 reason=same-uid-peer-isolation-absent stage=SEMANTICS_FROZEN' | stream_hash)" == "$(field action_9025_decision_sha256)" ]] ||
  fail 'action-9025 denial hash drifted'

material_result="$(bash scripts/ci/sounio_loom_process_witness_effect_hypercube_v11_freeze_selftest.sh 2>/dev/null)"
[[ "$material_result" == sounio-loom-process-witness-effect-hypercube-v11-freeze-selftest:\ PASS* ]] ||
  fail 'frozen V11 material hypercube gate failed'
[[ "$material_result" == *'material_hypercube=true material_coverage=false'* ]] ||
  fail 'material hypercube boundary drifted'

judgment_result="$(bash "$(field selftest_path)" 2>/dev/null)"
[[ "$judgment_result" == "$(evidence_field result)" ]] ||
  fail 'source-fresh Sounio material judgment drifted'
[[ "$judgment_result" == *'receipt=ALLOW action_9025=DENY451'* ]] ||
  fail 'source-fresh action-9025 decision drifted'
[[ "$judgment_result" == *'peer_rule_sabotage_promotes_9025=ALLOW'* ]] ||
  fail 'peer-isolation causal sabotage no longer proves the refusing rule'
[[ "$judgment_result" == *'python_oracle=DENY459'* && "$judgment_result" == *'python_executed=false'* ]] ||
  fail 'Python oracle negative control drifted'

printf 'sounio-loom-process-witness-effect-material-judgment-v11-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9025 manifest_sha256=%s evidence_sha256=%s material_receipt=ALLOW action_9025=DENY451 reason=same-uid-peer-isolation-absent causal_rule=peer-isolation-truth causal_rule_sabotage=ACTION_9025_ALLOW causal_sabotage=PASS python_oracle=DENY459 python_executed=false material_hypercube=true material_coverage=false complete_effects=false material_execution=false action_9025_judged=true production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false next_stage=V12_KERNEL_PEER_AUTHORITY\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)"
