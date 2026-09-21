#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
SOURCE="$ROOT_DIR/tools/loom/process_witness_effect_material_judgment_v11_main.sio"
ENTRYPOINT_9025="$ROOT_DIR/tools/loom/effect_closure_authority_main.sio"
MODULE_9025="$ROOT_DIR/stdlib/coordination/loom_effect_closure_authority.sio"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-material-judgment-v11-test.XXXXXX")"
JUDGE_A="$TEST_ROOT/material-judge-a"
JUDGE_B="$TEST_ROOT/material-judge-b"
ACTION_9025="$TEST_ROOT/action-9025-authority"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-effect-material-judgment-v11-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

file_hash() {
  sha256sum "$1" | cut -d ' ' -f 1
}

SOUNIO_LOOM_EFFECT_MATERIAL_JUDGMENT_V11_OUTPUT="$JUDGE_A" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_material_judgment_v11.sh" >/dev/null
SOUNIO_LOOM_EFFECT_MATERIAL_JUDGMENT_V11_OUTPUT="$JUDGE_B" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_effect_material_judgment_v11.sh" >/dev/null
[[ "$(file_hash "$JUDGE_A")" == "$(file_hash "$JUDGE_B")" ]] ||
  fail 'Sounio material judgment rebuild is nondeterministic'

SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$ACTION_9025" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null

fixtures="$(printf '0\n' | "$JUDGE_A")"
metadata="$(printf '%s\n' "$fixtures" | sed -n '1p')"
[[ "$metadata" == 'LOOM_PROCESS_WITNESS_EFFECT_MATERIAL_JUDGMENT_V11 producer=Sounio role=SEMANTIC_AUTHORITY action=9025 material_evidence_sha256=57bc9730b0b5662a548af8271bdca6ed1651c5684c7999182e6c3d6e6ad53738 certificate_bundle_sha256=1c92fcd7c97a5df4e8316b722f769f6777ea5979edcd09c207e88f9930f8d3dd cases=9' ]] ||
  fail "unexpected Sounio metadata: $metadata"

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

for label in accepted missing_vertex crossed_rule unavailable_vertex invariant_drift authority_laundering python_oracle evidence_substitution same_uid_claim; do
  output="$(run_receipt "$JUDGE_A" "$(fixture_frame "$label")")"
  first="$(printf '%s\n' "$output" | sed -n '1p')"
  actual_code="$(printf '%s\n' "$first" | sed -n 's/.* code=\([0-9][0-9]*\) .*/\1/p')"
  [[ "$actual_code" == "$(fixture_code "$label")" ]] ||
    fail "$label disagreed with its Sounio-owned expected code: $first"
done

accepted_output="$(run_receipt "$JUDGE_A" "$(fixture_frame accepted)")"
accepted_receipt="$(printf '%s\n' "$accepted_output" | sed -n '1p')"
accepted_action_line="$(printf '%s\n' "$accepted_output" | sed -n '2p')"
[[ "$accepted_receipt" == 'SOUNIO_MATERIAL_RECEIPT_ALLOW code=0 reason=material-hypercube-accepted stage=MATERIAL_HYPERCUBE material_hypercube=true material_coverage=false complete_effects=false material_execution=false action_9025_judged=false' ]] ||
  fail "accepted material receipt diverged: $accepted_receipt"
[[ "$accepted_action_line" == 'ACTION_9025_FRAME same_uid_peer_isolation=0 frame='* ]] ||
  fail 'accepted material receipt omitted the honest action-9025 frame'
action_frame="${accepted_action_line#* frame=}"
action_decision="$(printf '%s\n' "$action_frame" | "$ACTION_9025" || true)"
expected_action="$(printf '%s\n' "$fixtures" | grep -m1 '^EXPECTED_ACTION_9025 ')"
[[ "$expected_action" == *'decision=DENY451 reason=same-uid-peer-isolation-absent'* ]] ||
  fail 'Sounio omitted the expected action-9025 denial'
[[ "$action_decision" == 'SOUNIO_EFFECT_CLOSURE_DENY code=451 reason=same-uid-peer-isolation-absent stage=SEMANTICS_FROZEN' ]] ||
  fail "action 9025 disagreed with the material judgment: $action_decision"

malformed="$(run_receipt "$JUDGE_A" '1125 5')"
[[ "$malformed" == 'SOUNIO_MATERIAL_RECEIPT_DENY code=424 reason=malformed-frame stage=INVALID' ]] ||
  fail "malformed receipt was not refused: $malformed"

sabotage() {
  local label="$1" rule="$2" fixture="$3" expect_action_allow="$4"
  local sabotaged_source="$TEST_ROOT/$label.sio"
  local sabotaged_runtime="$TEST_ROOT/$label-runtime"
  grep -Fqx "$rule" "$SOURCE" || fail "$label rule is absent or changed"
  grep -Fvx "$rule" "$SOURCE" >"$sabotaged_source"
  SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$sabotaged_source" \
    -o "$sabotaged_runtime" >/dev/null
  chmod 0755 "$sabotaged_runtime"
  local output receipt
  output="$(run_receipt "$sabotaged_runtime" "$(fixture_frame "$fixture")")"
  receipt="$(printf '%s\n' "$output" | sed -n '1p')"
  [[ "$receipt" == SOUNIO_MATERIAL_RECEIPT_ALLOW\ code=0* ]] ||
    fail "$label sabotage did not admit its targeted false receipt: $receipt"
  if [[ "$expect_action_allow" == true ]]; then
    local line frame decision
    line="$(printf '%s\n' "$output" | sed -n '2p')"
    [[ "$line" == 'ACTION_9025_FRAME same_uid_peer_isolation=1 frame='* ]] ||
      fail "$label sabotage did not forge the peer-isolation bit"
    frame="${line#* frame=}"
    decision="$(printf '%s\n' "$frame" | "$ACTION_9025" || true)"
    [[ "$decision" == 'SOUNIO_EFFECT_CLOSURE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
      fail "$label sabotage did not falsely promote action 9025: $decision"
  fi
}

sabotage material-completeness \
  '    if decision == 0 && (families != 12 || probes != 13 || dimensions != 18 || vertices != 40 || refusals != 25 || completions != 15 || extinctions != 15 || mincuts != 13 || crossed != 0 || unavailable != 0 || invariant_stable != 1 || delta_distinct != 1 || triple_hash_binding != 1 || full_treatments_refused != 1 || open_vertices_completed != 1 || intermediate_matches != 1 || all_witnesses_extinct != 1) { decision = 447 }' \
  missing_vertex false
sabotage authority-laundering \
  '    if decision == 0 && (producer_role != 2 || producer_semantic_decision != 0) { decision = 459 }' \
  authority_laundering false
sabotage provenance-binding \
  '    if decision == 0 && (!effect_material_v11_digest_matches(a0, a1, a2, a3, a4, a5, a6, a7, 2914808145, 3666984210, 2458710105, 2768035869, 3852154889, 1807204030, 1438516226, 4206466860) || !effect_material_v11_digest_matches(b0, b1, b2, b3, b4, b5, b6, b7, 1471977264, 2964678186, 1418393639, 467445485, 374457704, 1283037464, 778845550, 1792358200) || !effect_material_v11_digest_matches(c0, c1, c2, c3, c4, c5, c6, c7, 479395031, 3380239860, 3895552882, 796303207, 2011847033, 3989637570, 132681625, 821613533) || !effect_material_v11_digest_matches(d0, d1, d2, d3, d4, d5, d6, d7, 4037219443, 3655676126, 2250906405, 2434714477, 4039358109, 3812857368, 2569047499, 1120212317) || !effect_material_v11_digest_matches(e0, e1, e2, e3, e4, e5, e6, e7, 29767287, 2872469132, 402539604, 4185467973, 2503281933, 152805908, 127984902, 473352183) || !effect_material_v11_digest_matches(f0, f1, f2, f3, f4, f5, f6, f7, 494635579, 503030225, 4192602963, 2462971877, 2056063068, 3631375775, 4003849011, 3828832689)) { decision = 450 }' \
  evidence_substitution false
sabotage peer-isolation-truth \
  '    if decision == 0 && same_uid_peer_isolation != 0 { decision = 451 }' \
  same_uid_claim true

python_frame="$(fixture_frame python_oracle)"
python_output="$(run_receipt "$JUDGE_A" "$python_frame")"
[[ "$python_output" == 'SOUNIO_MATERIAL_RECEIPT_DENY code=459 reason=material-authority-laundering stage=MATERIAL_HYPERCUBE' ]] ||
  fail 'Python oracle receipt was not refused before promotion'

fixture_bundle_sha256="$(printf '%s\n' "$fixtures" | sha256sum | cut -d ' ' -f 1)"
printf 'sounio-loom-process-witness-effect-material-judgment-v11-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9025 cases=9 receipt=ALLOW action_9025=DENY451 reason=same-uid-peer-isolation-absent negative_material=DENY447x4 authority_laundering=DENY459 python_oracle=DENY459 evidence_substitution=DENY450 malformed=DENY424 causal_sabotage=ALLOWx4 peer_rule_sabotage_promotes_9025=ALLOW python_executed=false deterministic=true source_sha256=%s executable_sha256=%s fixture_bundle_sha256=%s material_hypercube=true material_coverage=false complete_effects=false material_execution=false action_9025_judged=true production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false\n' \
  "$(file_hash "$SOURCE")" "$(file_hash "$JUDGE_A")" "$fixture_bundle_sha256"
