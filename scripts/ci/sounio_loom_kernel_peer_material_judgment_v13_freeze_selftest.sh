#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$ROOT_DIR"
MANIFEST=tools/loom/kernel_peer_material_judgment_v13.freeze.v1
EVIDENCE=tools/loom/evidence/loom-kernel-peer-material-judgment-v13-20260829.txt

fail() {
  printf 'sounio-loom-kernel-peer-material-judgment-v13-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  fail 'V13 judgment freeze inputs are absent or linked'
! grep -Fq '__' "$MANIFEST" || fail 'manifest contains an unresolved marker'
! grep -Fq '__' "$EVIDENCE" || fail 'evidence contains an unresolved marker'

expect_field schema loom-kernel-peer-material-judgment-v13-freeze-v1
expect_field stage SOUNIO_MATERIAL_JUDGMENT_FROZEN_V13
expect_field semantic_authority Sounio
expect_field action 9025
expect_field judgment_source_commit 7b35129137e1ca716684897c9c59ac4a1ef76306
expect_field producing_language Sounio
expect_field language_role SEMANTIC_AUTHORITY
expect_field source_sha256 3383ad078cbdc3d029a1c96a6a3bd20928b206f6a634d2a2d4afced0c693accb
expect_field build_script_sha256 9183db8ada6dfb4e777e606a95d56716d00fd9e3b134d853cff6b7b6b5b6dc2a
expect_field selftest_sha256 0180f3f0869f708a565296536954d91b79527038e4b65c3685d8a5b7cdcf15ea
expect_field semantic_manifest_sha256 b3557d850ce0dc13c900f8dbb10c33f824ac25e908cb4a48dd2ef913267194c2
expect_field material_manifest_sha256 7ffdff3f9dd48753502e9151a117fdcac8ea5149ef4772aaea5594269c54b301
expect_field action_9025_manifest_sha256 c1f0cf93f8427acdf794246a11c3551e265a09be12a3cd000bad25b707e8ca91
expect_field toolchain_engine lean_single
expect_field toolchain_wrapper_sha256 2bd47bda0fa68acd118a00d321498e3dbe1d048a0ff747ccf9ea190c84f7cb71
expect_field toolchain_compiler_sha256 81c4929823460a807762abd4a878ca9adbd053313e7ac6662fa489456269a4a3
expect_field hardware_kernel 7.0.2-5-pve
expect_field hardware_architecture x86_64
expect_field hardware_cpu_model INTEL\(R\)_XEON\(R\)_GOLD_6526Y
expect_field cases 14
expect_field executable_sha256 e084ef14390e825a888efb2dad4e66fe5d463f8f9bf3c5bad21c10eea14b9ff9
expect_field fixture_bundle_sha256 f05c8c4825eb8aa280808b910edd749ba4cf1d47954575f52d54c91ad1caafb6
for enabled in controls_executed material_peer_matrix same_uid_peer_isolation material_coverage complete_effects material_execution action_9025_judged action_9025_allow peer_rule_sabotage_promotes_false_receipt peer_rule_sabotage_promotes_9025; do
  expect_field "$enabled" true
done
for boundary in production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed; do
  expect_field "$boundary" false
done
expect_field next_stage PARITY_OPEN

SOURCE_COMMIT="$(field judgment_source_commit)"
git cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'V13 judgment source commit is absent'
for pair in source_path:source_sha256 build_script_path:build_script_sha256 selftest_path:selftest_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  path="$(field "$path_key")"; expected="$(field "$hash_key")"
  [[ "$(file_hash "$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the V13 judgment source commit"
done
for pair in semantic_manifest_path:semantic_manifest_sha256 material_manifest_path:material_manifest_sha256 action_9025_manifest_path:action_9025_manifest_sha256 toolchain_wrapper_path:toolchain_wrapper_sha256 toolchain_compiler_path:toolchain_compiler_sha256; do
  path_key="${pair%%:*}"; hash_key="${pair#*:}"
  [[ "$(file_hash "$(field "$path_key")")" == "$(field "$hash_key")" ]] ||
    fail "$(field "$path_key") drifted"
done
[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'V13 judgment evidence drifted'

expect_evidence schema loom-kernel-peer-material-judgment-v13-evidence-v1
expect_evidence stage SOUNIO_MATERIAL_JUDGMENT_V13
expect_evidence judgment_source_commit "$SOURCE_COMMIT"
for key in producing_language language_role source_sha256 build_script_sha256 selftest_sha256 semantic_manifest_sha256 material_manifest_sha256 material_producing_language material_language_role action_9025_manifest_sha256 toolchain_engine toolchain_wrapper_sha256 toolchain_compiler_sha256 hardware_kernel hardware_architecture hardware_cpu_model cases executable_sha256 fixture_bundle_sha256 controls_executed material_peer_matrix same_uid_peer_isolation material_coverage complete_effects material_execution action_9025_judged action_9025_allow production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready python_executed rust_executed next_stage; do
  expect_evidence "$key" "$(field "$key")"
done
expect_evidence negative_coverage DENY447x3
expect_evidence authority_laundering DENY459
expect_evidence python_oracle DENY459
expect_evidence provenance_substitution DENY450x6
expect_evidence peer_truth DENY451x2
expect_evidence malformed DENY424
expect_evidence sabotage_operation DELETE_EXACT_SOUNIO_PEER_TRUTH_RULE
expect_evidence sabotage_target_fixture receiver_missing
expect_evidence sabotage_target_delta receiver_mediator_active:1-\>0

accepted="$(evidence_field accepted_receipt)"
[[ "$accepted" == 'SOUNIO_PEER_MATERIAL_RECEIPT_ALLOW code=0 reason=peer-material-certificate-accepted stage=MATERIAL_CONTROL_MATRIX same_uid_peer_isolation=true controls_executed=true material_peer_matrix=true material_coverage=true complete_effects=true material_execution=true action_9025_judged=false' ]] ||
  fail 'accepted receipt drifted'
[[ "$(evidence_field action_9025_decision)" == 'SOUNIO_EFFECT_CLOSURE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail 'action 9025 decision drifted'
[[ "$(evidence_field sabotage_false_receipt)" == 'SOUNIO_PEER_MATERIAL_RECEIPT_ALLOW code=0' ]] ||
  fail 'sabotage false receipt drifted'
[[ "$(evidence_field sabotage_false_action)" == 'SOUNIO_EFFECT_CLOSURE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail 'sabotage false action drifted'

peer_rule='    if decision == 0 && (same_kuid_pair != 1 || attacker_syscalls_open != 1 || receiver_mediator_active != 1 || all_epoch_objects_extinct != 1 || peer_rule_witness != 1) { decision = 451 }'
[[ "$(grep -Fxc "$peer_rule" "$(field source_path)")" == 1 ]] ||
  fail 'the causally tested peer-truth rule is not unique'

[[ "$(printf '%s' "$(evidence_field command)" | stream_hash)" == "$(evidence_field command_sha256)" ]] ||
  fail 'judgment command hash drifted'
[[ "$(printf '%s' "$(evidence_field result)" | stream_hash)" == "$(evidence_field result_sha256)" ]] ||
  fail 'judgment result hash drifted'
for fact in receipt=ALLOW action_9025=ALLOW same_uid_peer_isolation=true material_coverage=true complete_effects=true material_execution=true negative_coverage=DENY447x3 authority_laundering=DENY459 python_oracle=DENY459 provenance_substitution=DENY450x6 peer_truth=DENY451x2 malformed=DENY424 peer_rule_sabotage_promotes_false_receipt=ALLOW peer_rule_sabotage_promotes_9025=ALLOW python_executed=false rust_executed=false production_activation=false claim_ready=false; do
  [[ " $(evidence_field result) " == *" $fact "* ]] || fail "judgment result omitted $fact"
done

tampered="$(mktemp)"
trap 'rm -f "$tampered"' EXIT
sed 's/^same_uid_peer_isolation=true$/same_uid_peer_isolation=false/' "$EVIDENCE" >"$tampered"
[[ "$(file_hash "$tampered")" != "$(field evidence_sha256)" ]] ||
  fail 'tamper control did not break the judgment evidence hash'

actual="$(bash "$(field selftest_path)")"
[[ "$actual" == "$(evidence_field result)" ]] || fail 'source-fresh Sounio V13 judgment drifted'

printf 'sounio-loom-kernel-peer-material-judgment-v13-freeze-selftest: PASS semantic_authority=Sounio action=9025 manifest_sha256=%s evidence_sha256=%s source_commit=%s action_9025=ALLOW same_uid_peer_isolation=true material_coverage=true complete_effects=true material_execution=true peer_truth=DENY451x2 peer_rule_sabotage_promotes_false_receipt=ALLOW peer_rule_sabotage_promotes_9025=ALLOW production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false python_executed=false rust_executed=false next_stage=PARITY_OPEN\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)" "$SOURCE_COMMIT"
