#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-exec-grant-cell.XXXXXX")"
RUNTIME="$TEST_ROOT/kernel-exec-grant-cell-authority"
MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_exec_grant_cell_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-exec-grant-cell-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_exec_grant_cell_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_KERNEL_EXEC_GRANT_CELL_SELFTEST PASS cases=19' ]] ||
  fail "unexpected Sounio selftest: $selftest"

parent_9029='1636926980 3205986131 3323207532 3505413428 706242987 2411760920 1929815169 3727939342'
parent_9021='3497534264 556131944 3943529214 1565657389 3821375173 3204015455 2733765994 2625951936'
parent_9022='4125506095 3601417934 2711931735 20635855 2708941890 3284947684 758124027 2068177262'
one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
bindings="$parent_9029 $parent_9021 $parent_9022 $one $one $one $one $one $one"

issue_transition='1 0 1 2 3 4 5 6 7 100 50'
consume_transition='2 1 3 2 3 4 5 6 7 100 50'
close_transition='3 3 4 2 3 4 5 6 7 100 50'
revoke_transition='4 1 5 2 3 4 5 6 7 100 50'
issue_parents='1 1 0 0 0 1 0 0'
consume_parents='1 1 1 0 0 1 1 0'
close_parents='1 1 1 1 0 1 1 1'
revoke_parents='1 1 0 0 1 1 0 0'
identity='1 1 1 1 1 1 1 1'
peer='1 1 1 1 1 1 1 1 1'
shape='1 1 1 1 1 1 1 1'
consumption='1 1 1 1 1 1 1'
revocation='1 1 1 1 1 1 1'
live_extinction='0 0 0 0 1'
terminal_extinction='1 1 1 1 1'
live_outcome='0 0 0 0 0 0 0 0'
close_outcome='1 1 1 1 1 1 0 0'
revoke_outcome='0 0 0 0 0 1 1 1'
authority='1 1 1 1 1 1 1'
evidence='1 1 11 11'

valid_issue="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
valid_consume="9030 3 1 $consume_transition $consume_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
valid_close="9030 3 1 $close_transition $close_parents $identity $peer $shape $consumption $revocation $terminal_extinction $close_outcome $authority $evidence $bindings"
valid_revoke="9030 3 1 $revoke_transition $revoke_parents $identity $peer $shape $consumption $revocation $terminal_extinction $revoke_outcome $authority $evidence $bindings"

wrong_stage="9030 2 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
wrong_parent_hash="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $one $parent_9021 $parent_9022 $one $one $one $one $one $one"
current_material="9030 3 1 $issue_transition 1 0 0 0 0 0 0 0 $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
missing_parent="9030 3 1 $issue_transition 1 0 0 0 0 1 0 0 $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
substituted_identity="9030 3 1 $issue_transition $issue_parents 1 0 1 1 1 1 1 1 $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
wrong_peer="9030 3 1 $issue_transition $issue_parents $identity 1 1 1 1 1 1 0 0 1 $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
postwrite_validation="9030 3 1 $issue_transition $issue_parents $identity $peer 0 1 1 1 1 1 1 0 $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
bearer_consume="9030 3 1 $issue_transition $issue_parents $identity $peer $shape 0 0 1 1 1 1 0 $revocation $live_extinction $live_outcome $authority $evidence $bindings"
unfenced_crash="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption 1 1 1 1 0 1 1 $live_extinction $live_outcome $authority $evidence $bindings"
silent_extinction="9030 3 1 $close_transition $close_parents $identity $peer $shape $consumption $revocation 0 1 1 1 1 $close_outcome $authority $evidence $bindings"
open_outcome="9030 3 1 $close_transition $close_parents $identity $peer $shape $consumption $revocation $terminal_extinction 1 1 0 1 1 1 0 0 $authority $evidence $bindings"
python_oracle="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome 0 0 0 1 1 1 1 $evidence $bindings"
unbound_result="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $parent_9029 $parent_9021 $parent_9022 $one $one $one $one $one $zero"
incomplete_sabotage="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority 1 1 10 11 $bindings"
malformed_flag="9030 3 1 $issue_transition $issue_parents $identity 2 1 1 1 1 1 1 1 1 $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"

assert_output() {
  local label="$1" frame="$2" expected="$3"
  local actual
  actual="$(printf '%s\n' "$frame" | "$RUNTIME" || true)"
  [[ "$actual" == "$expected" ]] || fail "$label: $actual"
}

allow='SOUNIO_KERNEL_EXEC_GRANT_CELL_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output valid-issue "$valid_issue" "$allow"
assert_output valid-consume "$valid_consume" "$allow"
assert_output valid-close "$valid_close" "$allow"
assert_output valid-revoke "$valid_revoke" "$allow"
assert_output wrong-stage "$wrong_stage" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=405 reason=wrong-stage-or-parent-freeze stage=SOUNIO_EXECUTABLE'
assert_output wrong-parent-hash "$wrong_parent_hash" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=405 reason=wrong-stage-or-parent-freeze stage=SEMANTICS_FROZEN'
assert_output current-material "$current_material" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=491 reason=parent-authority-chain-incomplete stage=SEMANTICS_FROZEN'
assert_output missing-parent "$missing_parent" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=491 reason=parent-authority-chain-incomplete stage=SEMANTICS_FROZEN'
assert_output substituted-identity "$substituted_identity" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=492 reason=grant-identity-incomplete stage=SEMANTICS_FROZEN'
assert_output wrong-peer "$wrong_peer" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=493 reason=kernel-peer-binding-incomplete stage=SEMANTICS_FROZEN'
assert_output postwrite-validation "$postwrite_validation" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=494 reason=prewrite-shape-incomplete stage=SEMANTICS_FROZEN'
assert_output bearer-consume "$bearer_consume" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=495 reason=nonbearer-consumption-incomplete stage=SEMANTICS_FROZEN'
assert_output unfenced-crash "$unfenced_crash" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=496 reason=crash-revocation-incomplete stage=SEMANTICS_FROZEN'
assert_output silent-extinction "$silent_extinction" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=497 reason=affirmative-extinction-incomplete stage=SEMANTICS_FROZEN'
assert_output open-outcome "$open_outcome" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=498 reason=outcome-closure-incomplete stage=SEMANTICS_FROZEN'
assert_output python-oracle "$python_oracle" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=499 reason=authority-laundering stage=SEMANTICS_FROZEN'
assert_output unbound-result "$unbound_result" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=500 reason=provenance-incomplete stage=SEMANTICS_FROZEN'
assert_output incomplete-sabotage "$incomplete_sabotage" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=501 reason=sabotage-incomplete stage=SEMANTICS_FROZEN'
assert_output malformed-flag "$malformed_flag" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=424 reason=malformed-frame stage=SEMANTICS_FROZEN'
assert_output wrong-action "${valid_issue/9030 /9029 }" \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=405 reason=wrong-stage-or-parent-freeze stage=SEMANTICS_FROZEN'
assert_output malformed '9030 3' \
  'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=424 reason=malformed-frame stage=INVALID'

python_sentinel="$TEST_ROOT/python3"
python_executed="$TEST_ROOT/python-executed"
printf '#!/bin/sh\nprintf prohibited > %s\n' "$python_executed" > "$python_sentinel"
chmod 0755 "$python_sentinel"
oracle_decision="$(printf '%s\n' "$python_oracle" | "$RUNTIME" || true)"
if [[ "$oracle_decision" == SOUNIO_KERNEL_EXEC_GRANT_CELL_ALLOW* ]]; then
  "$python_sentinel"
fi
[[ ! -e "$python_executed" ]] || fail 'Python oracle executable crossed the Sounio refusal'

sabotage() {
  local label="$1" rule="$2" frame="$3"
  local sabotaged_module="$TEST_ROOT/$label.sio"
  local combined="$TEST_ROOT/$label-combined.sio"
  local sabotaged_runtime="$TEST_ROOT/$label-runtime"
  grep -Fqx "$rule" "$MODULE" || fail "$label rule is absent or changed"
  grep -Fvx "$rule" "$MODULE" > "$sabotaged_module"
  sed -n '1,$p' "$sabotaged_module" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$combined" \
    -o "$sabotaged_runtime" >/dev/null
  chmod 0755 "$sabotaged_runtime"
  local actual
  actual="$(printf '%s\n' "$frame" | "$sabotaged_runtime")"
  [[ "$actual" == "$allow" ]] ||
    fail "$label sabotage did not admit its unchanged witness: $actual"
}

sabotage parent-authority-chain \
  '    if (transition.operation == 1 && (parents.action_9029_prepare_allow != 1 || parents.action_9029_admit_allow != 0 || parents.action_9029_close_allow != 0 || parents.action_9029_abort_allow != 0 || parents.action_9021_issue_allow != 1 || parents.action_9021_recheck_allow != 0 || parents.action_9022_outcome_allow != 0)) || (transition.operation == 2 && (parents.action_9029_prepare_allow != 1 || parents.action_9029_admit_allow != 1 || parents.action_9029_close_allow != 0 || parents.action_9029_abort_allow != 0 || parents.action_9021_issue_allow != 1 || parents.action_9021_recheck_allow != 1 || parents.action_9022_outcome_allow != 0)) || (transition.operation == 3 && (parents.action_9029_prepare_allow != 1 || parents.action_9029_admit_allow != 1 || parents.action_9029_close_allow != 1 || parents.action_9029_abort_allow != 0 || parents.action_9021_issue_allow != 1 || parents.action_9021_recheck_allow != 1 || parents.action_9022_outcome_allow != 1)) || (transition.operation == 4 && (parents.action_9029_prepare_allow != 1 || parents.action_9029_close_allow != 0 || parents.action_9029_abort_allow != 1 || parents.action_9021_issue_allow != 1 || parents.action_9022_outcome_allow != 0 || (transition.current_state == 1 && (parents.action_9029_admit_allow != 0 || parents.action_9021_recheck_allow != 0)) || ((transition.current_state == 2 || transition.current_state == 3) && (parents.action_9029_admit_allow != 1 || parents.action_9021_recheck_allow != 1)))) { return 491 }' \
  "$missing_parent"
sabotage grant-identity \
  '    if identity.capsule_bound != 1 || identity.invocation_cell_bound != 1 || identity.command_equal != 1 || identity.environment_equal != 1 || identity.principal_equal != 1 || identity.generation_vector_bound != 1 || identity.generation_vector_equal != 1 || identity.material_observation_joined != 1 || transition.broker_epoch <= 0 || transition.lease_generation <= 0 || transition.custody_generation <= 0 || transition.invocation_generation <= 0 || transition.grant_generation <= 0 || transition.event_sequence <= 0 { return 492 }' \
  "$substituted_identity"
sabotage kernel-peer-binding \
  '    if peer.so_peercred_bound != 1 || peer.pidfd_bound != 1 || peer.start_tick_bound != 1 || peer.boot_namespace_bound != 1 || peer.harness_ancestry_bound != 1 || peer.worktree_cgroup_bound != 1 || peer.kernel_principal_distinct != 1 || peer.anti_injection_attested != 1 || peer.peer_operation_equal != 1 { return 493 }' \
  "$wrong_peer"
sabotage prewrite-shape \
  '    if shape.prewrite_validated != 1 || shape.closed_domain != 1 || shape.legal_transition != 1 || shape.generations_monotonic != 1 || shape.all_hashes_bound != 1 || shape.old_state_receipt_bound != 1 || shape.proposed_receipt_bound != 1 || shape.no_mutation_on_deny != 1 || transition.deadline_tick <= 0 || transition.budget_remaining <= 0 || (transition.operation == 1 && (transition.current_state != 0 || transition.next_state != 1)) || (transition.operation == 2 && (transition.current_state != 1 || transition.next_state != 3)) || (transition.operation == 3 && (transition.current_state != 3 || transition.next_state != 4)) || (transition.operation == 4 && ((transition.current_state != 1 && transition.current_state != 2 && transition.current_state != 3) || (transition.next_state != 5 && transition.next_state != 6))) { return 494 }' \
  "$postwrite_validation"
sabotage nonbearer-consumption \
  '    if consumption.handle_lookup_only != 1 || consumption.authenticated_before_lookup != 1 || consumption.single_writer != 1 || consumption.atomic_consume != 1 || consumption.barrier_custody != 1 || consumption.replay_isolated != 1 || consumption.filesystem_authority_absent != 1 { return 495 }' \
  "$bearer_consume"
sabotage crash-revocation \
  '    if revocation.deadline_bound != 1 || revocation.deadline_live != 1 || revocation.policy_failure_closed != 1 || revocation.broker_loss_revokes != 1 || revocation.kernel_loss_revokes != 1 || revocation.guardian_loss_revokes != 1 || revocation.quarantine_on_uncertainty != 1 { return 496 }' \
  "$unfenced_crash"
sabotage affirmative-extinction \
  '    if ((transition.operation == 1 || transition.operation == 2) && (extinction.state_absence_observed != 0 || extinction.generation_retired != 0 || extinction.authority_revoked != 0 || extinction.terminal_receipt_bound != 0 || extinction.silence_rejected != 1)) || ((transition.operation == 3 || transition.operation == 4) && (extinction.state_absence_observed != 1 || extinction.generation_retired != 1 || extinction.authority_revoked != 1 || extinction.terminal_receipt_bound != 1 || extinction.silence_rejected != 1)) { return 497 }' \
  "$silent_extinction"
sabotage outcome-closure \
  '    if ((transition.operation == 1 || transition.operation == 2) && (outcome.obligation_bound != 0 || outcome.outcome_complete != 0 || outcome.tree_quiescent != 0 || outcome.open_effects_zero != 0 || outcome.terminal_receipts_complete != 0 || outcome.journal_committed_before_remove != 0 || outcome.incomplete_materialized != 0 || outcome.success_absent != 0)) || (transition.operation == 3 && (outcome.obligation_bound != 1 || outcome.outcome_complete != 1 || outcome.tree_quiescent != 1 || outcome.open_effects_zero != 1 || outcome.terminal_receipts_complete != 1 || outcome.journal_committed_before_remove != 1 || outcome.incomplete_materialized != 0 || outcome.success_absent != 0)) || (transition.operation == 4 && ((transition.current_state == 1 && outcome.obligation_bound != 0) || ((transition.current_state == 2 || transition.current_state == 3) && outcome.obligation_bound != 1) || outcome.outcome_complete != 0 || outcome.journal_committed_before_remove != 1 || outcome.incomplete_materialized != 1 || outcome.success_absent != 1)) { return 498 }' \
  "$open_outcome"
sabotage prohibited-authority \
  '    if authority.producer_sounio != 1 || authority.expected_result_sounio != 1 || authority.python_absent != 1 || authority.rust_absent != 1 || authority.review_unpromoted != 1 || authority.parity_unpromoted != 1 || authority.bearer_authority_absent != 1 { return 499 }' \
  "$python_oracle"
sabotage provenance-binding \
  '    if evidence.provenance_complete != 1 || evidence.receipt_bound != 1 || !loom_kernel_exec_grant_cell_digest_nonzero(bindings.grant_identity_hash) || !loom_kernel_exec_grant_cell_digest_nonzero(bindings.command_environment_hash) || !loom_kernel_exec_grant_cell_digest_nonzero(bindings.peer_vector_hash) || !loom_kernel_exec_grant_cell_digest_nonzero(bindings.transition_journal_hash) || !loom_kernel_exec_grant_cell_digest_nonzero(bindings.source_semantics_toolchain_hash) || !loom_kernel_exec_grant_cell_digest_nonzero(bindings.result_receipt_hash) { return 500 }' \
  "$unbound_result"
sabotage sabotage-completeness \
  '    if evidence.sabotage_count != 11 || evidence.sabotage_required != 11 { return 501 }' \
  "$incomplete_sabotage"

printf '%s\n' \
  'sounio-loom-kernel-exec-grant-cell-authority-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9030 cases=21 positive=ALLOWx4 current_material=DENY491 parent=DENY491 identity=DENY492 peer=DENY493 prewrite=DENY494 nonbearer=DENY495 revocation=DENY496 extinction=DENY497 outcome=DENY498 python_oracle=DENY499 provenance=DENY500 sabotage=DENY501 malformed=DENY424 causal_sabotage=ALLOWx11 python_executed=false material_grant=false same_uid_peer_isolation=false'
