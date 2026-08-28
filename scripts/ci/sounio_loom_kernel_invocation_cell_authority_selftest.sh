#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-invocation-cell.XXXXXX")"
RUNTIME="$TEST_ROOT/kernel-invocation-cell-authority"
MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_invocation_cell_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-invocation-cell-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_invocation_cell_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_KERNEL_INVOCATION_CELL_SELFTEST PASS cases=17' ]] ||
  fail "unexpected Sounio selftest: $selftest"

parent_9028='1991017987 113822720 1367310835 4264184359 1117900107 2622180275 1259621157 4224578159'
parent_9025='3253784467 4165106381 4153681002 298013982 643434942 312724736 195896759 132696721'
parent_9023='2365323 2301161672 762924345 38070334 1558458629 1166539901 3590963442 1546541903'
one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
bindings="$parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $one"

prepare_join='1 1 1 1 1 1'
admit_join='2 2 1 1 1 1'
close_join='3 3 1 1 1 1'
abort_join='4 4 1 1 1 1'
capsule='1 1 5 6 7 1 1 0 0 1 1 1'
membrane='1 8 9 10 11 1 1 1 1'
scope='1 1 1 1 1 1'
coverage='1 100 1 50 1 1 1 1'
open_lifecycle='1 1 1 12 13 1 0 0 0'
close_lifecycle='1 1 1 12 13 1 0 1 0'
abort_lifecycle='1 1 1 12 13 1 0 0 1'
open_outcome='0 0 0 0 0 0 0 0 0 0'
close_outcome='14 1 1 1 1 1 0 0 0 0'
abort_outcome='14 0 1 1 1 1 1 1 1 1'
authority='1 1 1 1 1 1'
evidence='1 1 10 10'

valid_prepare="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
valid_admit="9029 3 1 $admit_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
valid_close="9029 3 1 $close_join $capsule $membrane $scope $coverage $close_lifecycle $close_outcome $authority $evidence $bindings"
valid_abort="9029 3 1 $abort_join $capsule $membrane $scope $coverage $abort_lifecycle $abort_outcome $authority $evidence $bindings"
wrong_stage="9029 2 1 $prepare_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
wrong_parent_hash="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $one $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $one"
current_material="9029 3 1 1 1 0 0 1 0 $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
borrowed_capsule="9029 3 1 $prepare_join 0 1 5 6 7 1 1 0 0 1 1 1 $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
wrong_actor="9029 3 1 $prepare_join $capsule 1 8 9 10 11 0 1 1 1 $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
out_of_scope="9029 3 1 $prepare_join $capsule $membrane 1 1 0 1 1 1 $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"
missing_deadline="9029 3 1 $prepare_join $capsule $membrane $scope 0 100 1 50 1 1 1 1 $open_lifecycle $open_outcome $authority $evidence $bindings"
replayed_cell="9029 3 1 $prepare_join $capsule $membrane $scope $coverage 1 1 1 12 13 1 1 0 0 $open_outcome $authority $evidence $bindings"
live_descendant="9029 3 1 $close_join $capsule $membrane $scope $coverage $close_lifecycle 14 1 0 1 1 1 0 0 0 0 $authority $evidence $bindings"
python_oracle="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome 0 0 1 1 0 1 $evidence $bindings"
missing_provenance="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $zero"
incomplete_sabotage="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $open_lifecycle $open_outcome $authority 1 1 9 10 $bindings"
malformed_flag="9029 3 1 $prepare_join 1 1 5 6 7 2 1 0 0 1 1 1 $membrane $scope $coverage $open_lifecycle $open_outcome $authority $evidence $bindings"

assert_output() {
  local label="$1" frame="$2" expected="$3"
  local actual
  actual="$(printf '%s\n' "$frame" | "$RUNTIME" || true)"
  [[ "$actual" == "$expected" ]] || fail "$label: $actual"
}

allow='SOUNIO_KERNEL_INVOCATION_CELL_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output valid-prepare "$valid_prepare" "$allow"
assert_output valid-admit "$valid_admit" "$allow"
assert_output valid-close "$valid_close" "$allow"
assert_output valid-abort "$valid_abort" "$allow"
assert_output wrong-stage "$wrong_stage" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=405 reason=wrong-stage-or-parent-freeze stage=SOUNIO_EXECUTABLE'
assert_output wrong-parent-hash "$wrong_parent_hash" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=405 reason=wrong-stage-or-parent-freeze stage=SEMANTICS_FROZEN'
assert_output current-material "$current_material" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=481 reason=parent-semantic-join-incomplete stage=SEMANTICS_FROZEN'
assert_output borrowed-capsule "$borrowed_capsule" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=482 reason=capsule-custody-incomplete stage=SEMANTICS_FROZEN'
assert_output wrong-actor "$wrong_actor" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=483 reason=membrane-actor-incomplete stage=SEMANTICS_FROZEN'
assert_output out-of-scope "$out_of_scope" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=484 reason=command-scope-incomplete stage=SEMANTICS_FROZEN'
assert_output missing-deadline "$missing_deadline" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=485 reason=deadline-coverage-incomplete stage=SEMANTICS_FROZEN'
assert_output replayed-cell "$replayed_cell" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=486 reason=one-shot-lifecycle-incomplete stage=SEMANTICS_FROZEN'
assert_output live-descendant "$live_descendant" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=487 reason=outcome-closure-incomplete stage=SEMANTICS_FROZEN'
assert_output python-oracle "$python_oracle" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=488 reason=authority-laundering stage=SEMANTICS_FROZEN'
assert_output missing-provenance "$missing_provenance" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=489 reason=provenance-incomplete stage=SEMANTICS_FROZEN'
assert_output incomplete-sabotage "$incomplete_sabotage" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=490 reason=sabotage-incomplete stage=SEMANTICS_FROZEN'
assert_output malformed-flag "$malformed_flag" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=424 reason=malformed-frame stage=SEMANTICS_FROZEN'
assert_output wrong-action "${valid_prepare/9029 /9028 }" \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=405 reason=wrong-stage-or-parent-freeze stage=SEMANTICS_FROZEN'
assert_output malformed '9029 3' \
  'SOUNIO_KERNEL_INVOCATION_CELL_DENY code=424 reason=malformed-frame stage=INVALID'

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

sabotage parent-semantic-join \
  '    if join.parent_9028_allow != 1 || join.parent_9025_allow != 1 || join.material_observation_joined != 1 { return 481 }' \
  "$current_material"
sabotage capsule-custody \
  '    if capsule.capsule_bound != 1 || capsule.identity_equal != 1 || capsule.lease_generation <= 0 || capsule.broker_epoch <= 0 || capsule.custody_generation <= 0 || capsule.broker_holds_pidfd != 1 || capsule.pidfd_cloexec != 1 || capsule.lane_holds_pidfd != 0 || capsule.resident_holds_pidfd != 0 || capsule.grant_bound != 1 || capsule.capsule_non_authorizing != 1 || capsule.barrier_custody != 1 { return 482 }' \
  "$borrowed_capsule"
sabotage membrane-actor \
  '    if membrane.membrane_bound != 1 || membrane.membrane_generation <= 0 || membrane.closure_generation <= 0 || membrane.resident_policy_generation <= 0 || membrane.event_sequence <= 0 || membrane.actor_equal != 1 || membrane.ancestry_bound != 1 || membrane.pre_effect_stopped != 1 || membrane.event_equal != 1 { return 483 }' \
  "$wrong_actor"
sabotage command-scope \
  '    if scope.command_bound != 1 || scope.worktree_bound != 1 || scope.claim_scope_bound != 1 || scope.target_bound != 1 || scope.target_identity_bound != 1 || scope.operation_bound != 1 { return 484 }' \
  "$out_of_scope"
sabotage deadline-coverage \
  '    if coverage.deadline_bound != 1 || coverage.deadline_tick <= 0 || coverage.budget_bound != 1 || coverage.budget_remaining <= 0 || coverage.effect_coverage_attested != 1 || coverage.architecture_attested != 1 || coverage.unknown_effect_kernel_denied != 1 || coverage.unsupported_effect_absent != 1 { return 485 }' \
  "$missing_deadline"
sabotage one-shot-lifecycle \
  '    if lifecycle.one_shot != 1 || lifecycle.sequence_monotonic != 1 || lifecycle.nontransferable != 1 || lifecycle.grant_generation <= 0 || lifecycle.cell_generation <= 0 || lifecycle.fresh_grant != 1 || lifecycle.already_consumed != 0 || (join.operation == 1 && (join.lifecycle_state != 1 || lifecycle.cell_closed != 0 || lifecycle.cell_poisoned != 0)) || (join.operation == 2 && (join.lifecycle_state != 2 || lifecycle.cell_closed != 0 || lifecycle.cell_poisoned != 0)) || (join.operation == 3 && (join.lifecycle_state != 3 || lifecycle.cell_closed != 1 || lifecycle.cell_poisoned != 0)) || (join.operation == 4 && (join.lifecycle_state != 4 || lifecycle.cell_closed != 0 || lifecycle.cell_poisoned != 1)) { return 486 }' \
  "$replayed_cell"
sabotage outcome-closure \
  '    if ((join.operation == 1 || join.operation == 2) && (outcome.outcome_generation != 0 || outcome.outcome_complete != 0 || outcome.tree_quiescent != 0 || outcome.open_effects_zero != 0 || outcome.terminal_receipts_complete != 0 || outcome.termination_complete != 0 || outcome.crash_poisoned != 0 || outcome.quarantine_bound != 0 || outcome.typed_abort_reason != 0 || outcome.unresolved_uncertainty != 0)) || (join.operation == 3 && (outcome.outcome_generation <= 0 || outcome.outcome_complete != 1 || outcome.tree_quiescent != 1 || outcome.open_effects_zero != 1 || outcome.terminal_receipts_complete != 1 || outcome.termination_complete != 1 || outcome.crash_poisoned != 0 || outcome.quarantine_bound != 0 || outcome.typed_abort_reason != 0 || outcome.unresolved_uncertainty != 0)) || (join.operation == 4 && (outcome.outcome_generation <= 0 || outcome.outcome_complete != 0 || outcome.tree_quiescent != 1 || outcome.open_effects_zero != 1 || outcome.terminal_receipts_complete != 1 || outcome.termination_complete != 1 || outcome.crash_poisoned != 1 || outcome.quarantine_bound != 1 || outcome.typed_abort_reason != 1 || outcome.unresolved_uncertainty != 1)) { return 487 }' \
  "$live_descendant"
sabotage prohibited-producer \
  '    if authority.producer_sounio != 1 || authority.python_absent != 1 || authority.rust_absent != 1 || authority.review_only_unpromoted != 1 || authority.expected_result_sounio != 1 || authority.parent_receipt_unpromoted != 1 { return 488 }' \
  "$python_oracle"
sabotage provenance-binding \
  '    if evidence.provenance_complete != 1 || evidence.receipt_bound != 1 || !loom_kernel_invocation_cell_digest_nonzero(bindings.capsule_hash) || !loom_kernel_invocation_cell_digest_nonzero(bindings.membrane_hash) || !loom_kernel_invocation_cell_digest_nonzero(bindings.command_hash) || !loom_kernel_invocation_cell_digest_nonzero(bindings.worktree_hash) || !loom_kernel_invocation_cell_digest_nonzero(bindings.claim_scope_hash) || !loom_kernel_invocation_cell_digest_nonzero(bindings.deadline_event_hash) || !loom_kernel_invocation_cell_digest_nonzero(bindings.source_semantics_hash) || !loom_kernel_invocation_cell_digest_nonzero(bindings.toolchain_hardware_hash) || !loom_kernel_invocation_cell_digest_nonzero(bindings.outcome_result_hash) { return 489 }' \
  "$missing_provenance"
sabotage sabotage-completeness \
  '    if evidence.sabotage_count != 10 || evidence.sabotage_required != 10 { return 490 }' \
  "$incomplete_sabotage"

printf '%s\n' \
  'sounio-loom-kernel-invocation-cell-authority-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9029 cases=19 positive=ALLOWx4 current_material=DENY481 capsule=DENY482 membrane=DENY483 scope=DENY484 coverage=DENY485 lifecycle=DENY486 outcome=DENY487 python_oracle=DENY488 provenance=DENY489 sabotage=DENY490 malformed=DENY424 causal_sabotage=ALLOWx10 material_invocation=false parity_open=false claim_ready=false'
