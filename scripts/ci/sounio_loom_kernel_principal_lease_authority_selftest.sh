#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-principal-lease.XXXXXX")"
RUNTIME="$TEST_ROOT/kernel-principal-lease-authority"
MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_principal_lease_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_principal_lease_authority_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-principal-lease-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_KERNEL_PRINCIPAL_LEASE_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_principal_lease_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_KERNEL_PRINCIPAL_LEASE_SELFTEST PASS cases=18' ]] ||
  fail "unexpected Sounio selftest: $selftest"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
all_bindings="$one $one $one $one $one $one $one $one $one $one $one"
broker='1 1 1 1 1 1'
allocator='1 1 1 1 1 1 1 2 40 41'
launch_lifecycle='1 0 3 1 1 1 0 0 0'
recycle_lifecycle='2 5 0 0 0 0 0 1 1'
launch='1 1 1 1 1 1'
privilege='1 1 1 1 1 1'
recovery='1 1 1 1 1 1'
fresh='0 1 0 0 0 0 0'
reused='1 0 1 1 1 1 1'
evidence='7 7 1'
absent_six='0 0 0 0 0 0'
absent_seven='0 0 0 0 0 0 0'
absent_evidence='0 0 0'

valid_launch="9027 3 1 $broker $allocator $launch_lifecycle $launch $privilege $recovery $fresh $evidence $all_bindings"
valid_recycle="9027 3 1 $broker $allocator $recycle_lifecycle $launch $privilege $recovery $reused $evidence $all_bindings"
wrong_stage="9027 2 1 $broker $allocator $launch_lifecycle $launch $privilege $recovery $fresh $evidence $all_bindings"
user_broker="9027 3 1 1 1 1 0 1 1 $allocator $launch_lifecycle $launch $privilege $recovery $fresh $evidence $all_bindings"
collision="9027 3 1 $broker 1 1 1 0 1 1 1 2 40 41 $launch_lifecycle $launch $privilege $recovery $fresh $evidence $all_bindings"
skipped_lifecycle="9027 3 1 $broker $allocator 1 0 3 0 0 0 0 0 0 $launch $privilege $recovery $fresh $evidence $all_bindings"
bad_parent_certificate="9027 3 1 $broker $allocator $launch_lifecycle 0 1 1 1 1 1 $privilege $recovery $fresh $evidence $all_bindings"
can_regain_privilege="9027 3 1 $broker $allocator $launch_lifecycle $launch 1 1 0 1 1 1 $recovery $fresh $evidence $all_bindings"
unfenced_crash="9027 3 1 $broker $allocator $launch_lifecycle $launch $privilege 1 1 0 1 1 1 $fresh $evidence $all_bindings"
stale_reuse="9027 3 1 $broker $allocator $launch_lifecycle $launch $privilege $recovery 1 0 0 1 1 1 1 $evidence $all_bindings"
incomplete_sabotage="9027 3 1 $broker $allocator $launch_lifecycle $launch $privilege $recovery $fresh 6 7 1 $all_bindings"
unbound="9027 3 1 $broker $allocator $launch_lifecycle $launch $privilege $recovery $fresh $evidence $one $one $one $one $one $zero $one $one $one $one $one"
bad_generation="9027 3 1 $broker 1 1 1 1 1 1 2 2 41 41 $launch_lifecycle $launch $privilege $recovery $fresh $evidence $all_bindings"
current_material="9027 3 1 $absent_six $allocator $launch_lifecycle $absent_six $absent_six $absent_six $absent_seven $absent_evidence $one $zero $zero $zero $zero $zero $zero $zero $zero $zero $zero"

assert_output() {
  local label="$1" frame="$2" expected="$3"
  local actual
  actual="$(printf '%s\n' "$frame" | "$RUNTIME" || true)"
  [[ "$actual" == "$expected" ]] || fail "$label: $actual"
}

assert_output valid-launch "$valid_launch" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output valid-recycle "$valid_recycle" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output wrong-stage "$wrong_stage" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=405 reason=wrong-stage-or-parent stage=SOUNIO_EXECUTABLE'
assert_output broker "$user_broker" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=463 reason=host-broker-boundary-incomplete stage=SEMANTICS_FROZEN'
assert_output allocator "$collision" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=464 reason=allocator-lease-incomplete stage=SEMANTICS_FROZEN'
assert_output lifecycle "$skipped_lifecycle" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=465 reason=lifecycle-transition-invalid stage=SEMANTICS_FROZEN'
assert_output parent-certificate "$bad_parent_certificate" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=466 reason=principal-certificate-incomplete stage=SEMANTICS_FROZEN'
assert_output privilege "$can_regain_privilege" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=467 reason=privilege-drop-incomplete stage=SEMANTICS_FROZEN'
assert_output recovery "$unfenced_crash" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=468 reason=crash-recovery-incomplete stage=SEMANTICS_FROZEN'
assert_output extinction "$stale_reuse" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=469 reason=affirmative-extinction-incomplete stage=SEMANTICS_FROZEN'
assert_output sabotage "$incomplete_sabotage" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=470 reason=sabotage-incomplete stage=SEMANTICS_FROZEN'
assert_output provenance "$unbound" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=471 reason=provenance-incomplete stage=SEMANTICS_FROZEN'
assert_output invalid-generation "$bad_generation" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=424 reason=malformed-frame stage=SEMANTICS_FROZEN'
assert_output current-material "$current_material" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=463 reason=host-broker-boundary-incomplete stage=SEMANTICS_FROZEN'
assert_output wrong-action "${valid_launch/9027 /9028 }" \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=405 reason=wrong-stage-or-parent stage=SEMANTICS_FROZEN'
assert_output malformed '9027 3' \
  'SOUNIO_KERNEL_PRINCIPAL_LEASE_DENY code=424 reason=malformed-frame stage=INVALID'

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
  [[ "$actual" == 'SOUNIO_KERNEL_PRINCIPAL_LEASE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
    fail "$label sabotage did not admit its unchanged witness: $actual"
}

sabotage host-broker-boundary \
  '    if broker.host_owned != 1 || broker.service_manager_supervised != 1 || broker.root_owned_socket != 1 || broker.user_invocation_denied != 1 || broker.policy_only != 1 || broker.frozen_policy_bound != 1 { return 463 }' \
  "$user_broker"
sabotage allocator-disjointness \
  '    if allocator.ranges_disjoint != 1 { return 464 }' \
  "$collision"
sabotage lifecycle-edges \
  '    if (lifecycle.operation == 1 && (lifecycle.start_state != 0 || lifecycle.end_state != 3 || lifecycle.free_to_reserved != 1 || lifecycle.reserved_to_mapped != 1 || lifecycle.mapped_to_launched != 1)) || (lifecycle.operation == 2 && ((lifecycle.start_state != 3 && lifecycle.start_state != 5) || lifecycle.end_state != 0 || (lifecycle.start_state == 3 && lifecycle.launched_to_draining != 1) || lifecycle.draining_to_quarantined != 1 || lifecycle.quarantined_to_free != 1)) { return 465 }' \
  "$skipped_lifecycle"
sabotage irreversible-drop \
  '    if privilege.irreversible_drop != 1 || privilege.no_new_privileges != 1 || privilege.privilege_regain_denied != 1 || privilege.lane_setuid_absent != 1 || privilege.capabilities_empty != 1 || privilege.broker_descriptors_closed != 1 { return 467 }' \
  "$can_regain_privilege"
sabotage crash-recovery \
  '    if recovery.broker_crash_fail_closed != 1 || recovery.orphan_scan_complete != 1 || recovery.generation_fenced != 1 || recovery.grants_revoked != 1 || recovery.incomplete_outcomes_materialized != 1 || recovery.forced_quarantine != 1 { return 468 }' \
  "$unfenced_crash"
sabotage affirmative-extinction \
  '    if (extinction.reuse_mode == 0 && extinction.never_used_receipt != 1) || (extinction.reuse_mode == 1 && (extinction.process_extinct != 1 || extinction.namespace_extinct != 1 || extinction.authority_extinct != 1 || extinction.quarantine_receipt != 1 || extinction.receipt_bound != 1)) || (lifecycle.operation == 2 && extinction.reuse_mode != 1) { return 469 }' \
  "$stale_reuse"
sabotage sabotage-completeness \
  '    if evidence.sabotage_count != 7 || evidence.sabotage_required != 7 { return 470 }' \
  "$incomplete_sabotage"

printf '%s\n' \
  'sounio-loom-kernel-principal-lease-authority-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9027 cases=18 positive=ALLOWx2 current_material=DENY463 broker=DENY463 allocator=DENY464 lifecycle=DENY465 parent=DENY466 privilege=DENY467 recovery=DENY468 extinction=DENY469 sabotage=DENY470 provenance=DENY471 malformed=DENY424 causal_sabotage=ALLOWx7 material_broker=false'
