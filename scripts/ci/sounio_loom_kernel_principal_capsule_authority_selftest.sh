#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-principal-capsule.XXXXXX")"
RUNTIME="$TEST_ROOT/kernel-principal-capsule-authority"
MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_principal_capsule_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_principal_capsule_authority_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-principal-capsule-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_KERNEL_PRINCIPAL_CAPSULE_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_principal_capsule_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_SELFTEST PASS cases=19' ]] ||
  fail "unexpected Sounio selftest: $selftest"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
bindings="$one $one $one $one $one $one $one $one $one $one $one $one $one"
mint_order='1 1 1 2 3 1 1'
recovery_order='2 1 1 5 5 0 0'
identity='100 200 300 4 5 6 1 1 1'
isolation='1 1 1 1 1 1 1'
privilege='1 1 1 1 1 1 1 1'
custody='1 1 0 0 0 0 1'
mint_grant='1 1 1 1 1 1 1 1 1'
recovery_grant='1 0 0 0 0 0 0 0 1'
mint_recovery='0 3 3 5 5 0 0 0 0 0'
recovery='1 3 4 5 6 1 1 1 1 1'
evidence='9 9 1'

valid_mint="9028 3 1 $mint_order $identity $isolation $privilege $custody $mint_grant $mint_recovery $evidence $bindings"
valid_recovery="9028 3 1 $recovery_order $identity $isolation $privilege $custody $recovery_grant $recovery $evidence $bindings"
wrong_stage="9028 2 1 $mint_order $identity $isolation $privilege $custody $mint_grant $mint_recovery $evidence $bindings"
bad_parent="9028 3 1 1 1 0 2 3 1 1 $identity $isolation $privilege $custody $mint_grant $mint_recovery $evidence $bindings"
pid_reuse="9028 3 1 $mint_order 100 200 300 4 5 6 0 1 1 $isolation $privilege $custody $mint_grant $mint_recovery $evidence $bindings"
outer_namespace="9028 3 1 $mint_order $identity 0 1 1 1 1 1 1 $privilege $custody $mint_grant $mint_recovery $evidence $bindings"
regainable="9028 3 1 $mint_order $identity $isolation 1 1 1 0 1 1 1 1 $custody $mint_grant $mint_recovery $evidence $bindings"
raw_pidfd="9028 3 1 $mint_order $identity $isolation $privilege 1 1 0 0 0 1 1 $mint_grant $mint_recovery $evidence $bindings"
bearer_capsule="9028 3 1 $mint_order $identity $isolation $privilege $custody 0 1 1 1 1 1 1 1 1 $mint_recovery $evidence $bindings"
stale_recovery="9028 3 1 $recovery_order $identity $isolation $privilege $custody $recovery_grant 1 3 3 5 6 1 1 1 1 1 $evidence $bindings"
unbound="9028 3 1 $mint_order $identity $isolation $privilege $custody $mint_grant $mint_recovery $evidence $one $one $one $one $one $one $one $one $one $one $one $one $zero"
incomplete_sabotage="9028 3 1 $mint_order $identity $isolation $privilege $custody $mint_grant $mint_recovery 8 9 1 $bindings"
malformed_flag="9028 3 1 $mint_order $identity $isolation $privilege 2 1 0 0 0 0 1 $mint_grant $mint_recovery $evidence $bindings"
current_material="9028 3 1 1 0 0 2 3 1 1 $identity $isolation $privilege $custody $mint_grant $mint_recovery $evidence $bindings"

assert_output() {
  local label="$1" frame="$2" expected="$3"
  local actual
  actual="$(printf '%s\n' "$frame" | "$RUNTIME" || true)"
  [[ "$actual" == "$expected" ]] || fail "$label: $actual"
}

assert_output valid-mint "$valid_mint" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output valid-recovery "$valid_recovery" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output wrong-stage "$wrong_stage" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=405 reason=wrong-stage-or-parent stage=SOUNIO_EXECUTABLE'
assert_output parent-order "$bad_parent" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=472 reason=parent-launch-order-incomplete stage=SEMANTICS_FROZEN'
assert_output kernel-identity "$pid_reuse" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=473 reason=kernel-identity-incomplete stage=SEMANTICS_FROZEN'
assert_output isolation "$outer_namespace" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=474 reason=isolation-vector-incomplete stage=SEMANTICS_FROZEN'
assert_output privilege "$regainable" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=475 reason=privilege-posture-incomplete stage=SEMANTICS_FROZEN'
assert_output custody "$raw_pidfd" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=476 reason=pidfd-custody-incomplete stage=SEMANTICS_FROZEN'
assert_output grant "$bearer_capsule" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=477 reason=grant-fence-incomplete stage=SEMANTICS_FROZEN'
assert_output recovery "$stale_recovery" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=478 reason=recovery-lineage-incomplete stage=SEMANTICS_FROZEN'
assert_output provenance "$unbound" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=479 reason=provenance-incomplete stage=SEMANTICS_FROZEN'
assert_output sabotage "$incomplete_sabotage" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=480 reason=sabotage-incomplete stage=SEMANTICS_FROZEN'
assert_output malformed-flag "$malformed_flag" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=424 reason=malformed-frame stage=SEMANTICS_FROZEN'
assert_output current-material "$current_material" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=472 reason=parent-launch-order-incomplete stage=SEMANTICS_FROZEN'
assert_output wrong-action "${valid_mint/9028 /9029 }" \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=405 reason=wrong-stage-or-parent stage=SEMANTICS_FROZEN'
assert_output malformed '9028 3' \
  'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_DENY code=424 reason=malformed-frame stage=INVALID'

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
  [[ "$actual" == 'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
    fail "$label sabotage did not admit its unchanged witness: $actual"
}

sabotage parent-launch-order \
  '    if order.parent_9026_allow != 1 || order.parent_9027_allow != 1 || (order.operation == 1 && (order.lease_start_state != 2 || order.lease_target_state != 3 || order.preexec_barrier_closed != 1 || order.journal_transition_uncommitted != 1)) || (order.operation == 2 && ((order.lease_start_state != 3 && order.lease_start_state != 5) || order.lease_target_state != 5 || order.preexec_barrier_closed != 0 || order.journal_transition_uncommitted != 0)) { return 472 }' \
  "$bad_parent"
sabotage pid-reuse-identity \
  '    if identity.host_pid <= 1 || identity.start_time <= 0 || identity.pidfd_identity <= 0 || identity.broker_epoch <= 0 || identity.lease_generation <= 0 || identity.custody_generation <= 0 || identity.pidfd_peer_matches != 1 || identity.start_time_matches != 1 || identity.vector_complete != 1 { return 473 }' \
  "$pid_reuse"
sabotage namespace-cgroup-isolation \
  '    if isolation.namespaces_distinct != 1 || isolation.maps_exact != 1 || isolation.maps_disjoint != 1 || isolation.cgroup_exact != 1 || isolation.resources_enforced != 1 || isolation.descendants_contained != 1 || isolation.cgroup_escape_denied != 1 { return 474 }' \
  "$outer_namespace"
sabotage privilege-posture \
  '    if privilege.ids_exact != 1 || privilege.supplementary_groups_empty != 1 || privilege.capabilities_empty != 1 || privilege.no_new_privileges != 1 || privilege.seccomp_installed != 1 || privilege.nondumpable != 1 || privilege.setid_path_absent != 1 || privilege.broker_descriptors_absent != 1 { return 475 }' \
  "$regainable"
sabotage broker-only-pidfd-custody \
  '    if custody.broker_holds_pidfd != 1 || custody.pidfd_cloexec != 1 || custody.lane_holds_pidfd != 0 || custody.resident_holds_pidfd != 0 || custody.raw_pid_authority_exported != 0 || custody.raw_pidfd_authority_exported != 0 || custody.broker_lookup_required != 1 { return 476 }' \
  "$raw_pidfd"
sabotage non-bearer-grant-fence \
  '    if grant.capsule_non_authorizing != 1 || grant.review_only_enforced != 1 || (order.operation == 1 && (grant.execgrant_bound != 1 || grant.single_use != 1 || grant.unconsumed != 1 || grant.peer_bound != 1 || grant.ancestry_bound != 1 || grant.effect_bound != 1 || grant.command_bound != 1)) || (order.operation == 2 && (grant.execgrant_bound != 0 || grant.single_use != 0 || grant.unconsumed != 0 || grant.peer_bound != 0 || grant.ancestry_bound != 0 || grant.effect_bound != 0 || grant.command_bound != 0)) { return 477 }' \
  "$bearer_capsule"
sabotage quarantine-recovery-lineage \
  '    if (order.operation == 1 && recovery.recovery_mode != 0) || (order.operation == 2 && (recovery.recovery_mode != 1 || recovery.current_broker_epoch <= recovery.previous_broker_epoch || recovery.current_custody_generation <= recovery.previous_custody_generation || recovery.lineage_equal != 1 || recovery.prior_grants_revoked != 1 || recovery.old_custody_extinct != 1 || recovery.remains_quarantined != 1 || recovery.barrier_release_denied != 1)) { return 478 }' \
  "$stale_recovery"
sabotage provenance-binding \
  '    if evidence.receipt_bound != 1 || !loom_kernel_principal_capsule_digest_nonzero(bindings.launch_receipt_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.custody_receipt_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.grant_receipt_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.recovery_receipt_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.capsule_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.source_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.semantics_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.toolchain_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.hardware_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.command_hash) || !loom_kernel_principal_capsule_digest_nonzero(bindings.result_hash) { return 479 }' \
  "$unbound"
sabotage sabotage-completeness \
  '    if evidence.sabotage_count != 9 || evidence.sabotage_required != 9 { return 480 }' \
  "$incomplete_sabotage"

printf '%s\n' \
  'sounio-loom-kernel-principal-capsule-authority-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9028 cases=16 positive=ALLOWx2 current_material=DENY472 order=DENY472 identity=DENY473 isolation=DENY474 privilege=DENY475 custody=DENY476 grant=DENY477 recovery=DENY478 provenance=DENY479 sabotage=DENY480 malformed=DENY424 causal_sabotage=ALLOWx9 bare_pid_authority=forbidden raw_pidfd_authority=forbidden material_capsule=false'
