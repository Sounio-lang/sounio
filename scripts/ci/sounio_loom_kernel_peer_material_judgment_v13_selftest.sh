#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOURCE="$ROOT_DIR/tools/loom/kernel_peer_material_judgment_v13_main.sio"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-material-judgment-v13-test.XXXXXX")"
JUDGE_A="$TEST_ROOT/peer-judge-a"
JUDGE_B="$TEST_ROOT/peer-judge-b"
ACTION_9025="$TEST_ROOT/action-9025-authority"

cleanup() { rm -rf "$TEST_ROOT"; }
trap cleanup EXIT
fail() {
  printf 'sounio-loom-kernel-peer-material-judgment-v13-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}
file_hash() { sha256sum "$1" | cut -d ' ' -f 1; }

SOUNIO_LOOM_KERNEL_PEER_MATERIAL_JUDGMENT_V13_OUTPUT="$JUDGE_A" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_material_judgment_v13.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_MATERIAL_JUDGMENT_V13_OUTPUT="$JUDGE_B" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_material_judgment_v13.sh" >/dev/null
[[ "$(file_hash "$JUDGE_A")" == "$(file_hash "$JUDGE_B")" ]] ||
  fail 'Sounio V13 material judgment rebuild is nondeterministic'

SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$ACTION_9025" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null

fixtures="$(printf '0\n' | "$JUDGE_A")"
metadata="$(printf '%s\n' "$fixtures" | sed -n '1p')"
[[ "$metadata" == 'LOOM_KERNEL_PEER_MATERIAL_JUDGMENT_V13 producer=Sounio role=SEMANTIC_AUTHORITY semantic_authority=Sounio action=9025 semantic_manifest_sha256=b3557d850ce0dc13c900f8dbb10c33f824ac25e908cb4a48dd2ef913267194c2 material_manifest_sha256=7ffdff3f9dd48753502e9151a117fdcac8ea5149ef4772aaea5594269c54b301 cases=14' ]] ||
  fail "unexpected Sounio V13 judgment metadata: $metadata"

fixture_line() {
  local label="$1" count
  count="$(printf '%s\n' "$fixtures" | grep -c "^CASE label=${label} " || true)"
  [[ "$count" == 1 ]] || fail "fixture $label occurs $count times"
  printf '%s\n' "$fixtures" | grep -m1 "^CASE label=${label} "
}
fixture_frame() {
  local line
  line="$(fixture_line "$1")"
  printf '%s' "${line#* FRAME }"
}
fixture_code() {
  local line rest
  line="$(fixture_line "$1")"
  rest="${line#* EXPECT code=}"
  printf '%s' "${rest%% *}"
}
run_receipt() {
  local runtime="$1" frame="$2"
  printf '%s\n' "$frame" | "$runtime" || true
}

for label in accepted missing_observation unavailable_drift crossed_rule authority_laundering python_oracle semantic_substitution material_substitution pair_substitution control_substitution sabotage_substitution host_result_substitution receiver_missing peer_rule_sabotaged; do
  output="$(run_receipt "$JUDGE_A" "$(fixture_frame "$label")")"
  first="$(printf '%s\n' "$output" | sed -n '1p')"
  actual_code="$(printf '%s\n' "$first" | sed -n 's/.* code=\([0-9][0-9]*\) .*/\1/p')"
  [[ "$actual_code" == "$(fixture_code "$label")" ]] ||
    fail "$label disagreed with its Sounio-owned expected code: $first"
done

accepted_output="$(run_receipt "$JUDGE_A" "$(fixture_frame accepted)")"
accepted_receipt="$(printf '%s\n' "$accepted_output" | sed -n '1p')"
accepted_action_line="$(printf '%s\n' "$accepted_output" | sed -n '2p')"
[[ "$accepted_receipt" == 'SOUNIO_PEER_MATERIAL_RECEIPT_ALLOW code=0 reason=peer-material-certificate-accepted stage=MATERIAL_CONTROL_MATRIX same_uid_peer_isolation=true controls_executed=true material_peer_matrix=true material_coverage=true complete_effects=true material_execution=true action_9025_judged=false' ]] ||
  fail "accepted peer-material receipt diverged: $accepted_receipt"
[[ "$accepted_action_line" == 'ACTION_9025_FRAME same_uid_peer_isolation=1 frame='* ]] ||
  fail 'accepted peer-material receipt omitted the honest action-9025 frame'
action_frame="${accepted_action_line#* frame=}"
action_decision="$(printf '%s\n' "$action_frame" | "$ACTION_9025" || true)"
[[ "$action_decision" == 'SOUNIO_EFFECT_CLOSURE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail "action 9025 disagreed with the V13 material judgment: $action_decision"

expected_action="$(printf '%s\n' "$fixtures" | grep -m1 '^EXPECTED_ACTION_9025 ')"
[[ "$expected_action" == *'decision=ALLOW reason=allow same_uid_peer_isolation=true material_coverage=true complete_effects=true material_execution=true'* ]] ||
  fail 'Sounio omitted the expected action-9025 promotion'

for label in receiver_missing peer_rule_sabotaged; do
  denied="$(run_receipt "$JUDGE_A" "$(fixture_frame "$label")")"
  [[ "$denied" == 'SOUNIO_PEER_MATERIAL_RECEIPT_DENY code=451 reason=same-uid-peer-isolation-unproven stage=MATERIAL_CONTROL_MATRIX' ]] ||
    fail "$label did not exercise the peer-truth denial: $denied"
done

malformed="$(run_receipt "$JUDGE_A" '1325 6')"
[[ "$malformed" == 'SOUNIO_PEER_MATERIAL_RECEIPT_DENY code=424 reason=malformed-frame stage=INVALID' ]] ||
  fail "malformed receipt was not refused: $malformed"

peer_rule='    if decision == 0 && (same_kuid_pair != 1 || attacker_syscalls_open != 1 || receiver_mediator_active != 1 || all_epoch_objects_extinct != 1 || peer_rule_witness != 1) { decision = 451 }'
grep -Fqx "$peer_rule" "$SOURCE" || fail 'peer-truth rule is absent or changed'
sabotaged_source="$TEST_ROOT/peer-truth-sabotaged.sio"
sabotaged_runtime="$TEST_ROOT/peer-truth-sabotaged"
grep -Fvx "$peer_rule" "$SOURCE" >"$sabotaged_source"
SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$sabotaged_source" \
  -o "$sabotaged_runtime" >/dev/null
chmod 0755 "$sabotaged_runtime"
sabotaged_output="$(run_receipt "$sabotaged_runtime" "$(fixture_frame receiver_missing)")"
sabotaged_receipt="$(printf '%s\n' "$sabotaged_output" | sed -n '1p')"
[[ "$sabotaged_receipt" == SOUNIO_PEER_MATERIAL_RECEIPT_ALLOW\ code=0* ]] ||
  fail "peer-rule sabotage did not admit the targeted false receipt: $sabotaged_receipt"
sabotaged_action_line="$(printf '%s\n' "$sabotaged_output" | sed -n '2p')"
[[ "$sabotaged_action_line" == 'ACTION_9025_FRAME same_uid_peer_isolation=1 frame='* ]] ||
  fail 'peer-rule sabotage did not forge the peer-isolation fact'
sabotaged_action_frame="${sabotaged_action_line#* frame=}"
sabotaged_action_decision="$(printf '%s\n' "$sabotaged_action_frame" | "$ACTION_9025" || true)"
[[ "$sabotaged_action_decision" == 'SOUNIO_EFFECT_CLOSURE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
  fail "peer-rule sabotage did not falsely promote action 9025: $sabotaged_action_decision"

python_output="$(run_receipt "$JUDGE_A" "$(fixture_frame python_oracle)")"
[[ "$python_output" == 'SOUNIO_PEER_MATERIAL_RECEIPT_DENY code=459 reason=material-authority-laundering stage=MATERIAL_CONTROL_MATRIX' ]] ||
  fail 'Python oracle receipt was not refused before promotion'

fixture_bundle_sha256="$(printf '%s\n' "$fixtures" | sha256sum | cut -d ' ' -f 1)"
printf 'sounio-loom-kernel-peer-material-judgment-v13-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9025 cases=14 receipt=ALLOW action_9025=ALLOW same_uid_peer_isolation=true material_coverage=true complete_effects=true material_execution=true negative_coverage=DENY447x3 authority_laundering=DENY459 python_oracle=DENY459 provenance_substitution=DENY450x6 peer_truth=DENY451x2 malformed=DENY424 peer_rule_sabotage_promotes_false_receipt=ALLOW peer_rule_sabotage_promotes_9025=ALLOW python_executed=false rust_executed=false deterministic=true source_sha256=%s executable_sha256=%s fixture_bundle_sha256=%s production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false\n' \
  "$(file_hash "$SOURCE")" "$(file_hash "$JUDGE_A")" "$fixture_bundle_sha256"
