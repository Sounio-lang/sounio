#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_activation_capsule_authority.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_peer_activation_capsule_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_peer_activation_capsule_authority_main.sio"
PARENT_9025="$ROOT_DIR/tools/loom/kernel_peer_material_judgment_v13.freeze.v1"
PARENT_9030="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"

fail() {
  printf 'sounio-loom-kernel-peer-activation-capsule-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}
file_hash() {
  sha256sum "$1" | cut -d ' ' -f 1
}

[[ -x "$BUILD" ]] || fail 'build script is absent or not executable'
[[ -f "$MODULE" && ! -L "$MODULE" ]] || fail 'action 9031 module is absent or linked'
[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] || fail 'action 9031 entrypoint is absent or linked'
[[ "$(file_hash "$PARENT_9025")" == f7adafcd1c79364b75ebe48b66999ec2d7b82a12d6b8e45d9c1cc4637a4ca9ca ]] ||
  fail 'action 9025 material judgment parent drifted'
[[ "$(file_hash "$PARENT_9030")" == 8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051 ]] ||
  fail 'action 9030 authority parent drifted'
grep -Fxq 'action_9025_allow=true' "$PARENT_9025" || fail 'action 9025 material judgment is not ALLOW'
grep -Fxq 'same_uid_peer_isolation=true' "$PARENT_9025" || fail 'same-UID peer isolation is not frozen true'
grep -Fxq 'material_execution=true' "$PARENT_9025" || fail 'material execution observation is not frozen true'
grep -Fxq 'stage=SEMANTICS_FROZEN' "$PARENT_9030" || fail 'action 9030 semantics are not frozen'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-activation-capsule-test.XXXXXX")"
trap 'rm -rf "$work"' EXIT
runtime_a="$work/runtime-a"
runtime_b="$work/runtime-b"
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT="$runtime_a" "$BUILD" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT="$runtime_b" "$BUILD" >/dev/null
[[ "$(file_hash "$runtime_a")" == "$(file_hash "$runtime_b")" ]] ||
  fail 'source-fresh action 9031 builds are nondeterministic'

selftest_a="$(printf '0\n' | "$runtime_a")"
selftest_b="$(printf '0\n' | "$runtime_b")"
[[ "$selftest_a" == "$selftest_b" ]] || fail 'Sounio-owned selftest output is nondeterministic'
[[ "$selftest_a" == 'SOUNIO_KERNEL_PEER_ACTIVATION_CAPSULE_SELFTEST PASS cases=13' ]] ||
  fail "Sounio-owned selftest failed: $selftest_a"

fixtures_a="$(printf '1\n' | "$runtime_a")"
fixtures_b="$(printf '1\n' | "$runtime_b")"
[[ "$fixtures_a" == "$fixtures_b" ]] || fail 'Sounio-owned fixture bundle is nondeterministic'
[[ "$(printf '%s\n' "$fixtures_a" | grep -c '^CASE ')" == 16 ]] ||
  fail 'Sounio-owned fixture count drifted'

case_line() {
  local label="$1"
  printf '%s\n' "$fixtures_a" | sed -n "/^CASE label=${label} /p"
}
case_frame() {
  local line
  line="$(case_line "$1")"
  [[ -n "$line" ]] || fail "Sounio fixture is absent: $1"
  printf '%s\n' "${line#* FRAME }"
}
case_code() {
  local line rest
  line="$(case_line "$1")"
  [[ -n "$line" ]] || fail "Sounio fixture is absent: $1"
  rest="${line#* EXPECT code=}"
  printf '%s\n' "${rest%% *}"
}
output_code() {
  local output="$1" rest
  rest="${output#* code=}"
  printf '%s\n' "${rest%% *}"
}

while IFS= read -r line; do
  [[ "$line" == CASE\ label=* ]] || continue
  label="${line#CASE label=}"
  label="${label%% *}"
  frame="${line#* FRAME }"
  expected="$(case_code "$label")"
  output="$(printf '%s\n' "$frame" | "$runtime_a" || true)"
  [[ "$(printf '%s\n' "$output" | wc -l)" == 1 ]] || fail "$label emitted an unbounded result"
  [[ "$(output_code "$output")" == "$expected" ]] ||
    fail "$label expected Sounio code $expected but observed: $output"
done <<< "$fixtures_a"

allow_code="$(case_code seal)"
[[ "$allow_code" == "$(case_code consume)" && "$allow_code" == "$(case_code extinguish)" && "$allow_code" == "$(case_code poison)" ]] ||
  fail 'Sounio positive fixtures disagree on the allow code'

python_sentinel="$work/python3"
python_executed="$work/python-executed"
printf '#!/bin/sh\nprintf prohibited > %s\n' "$python_executed" > "$python_sentinel"
chmod 0755 "$python_sentinel"
python_output="$(printf '%s\n' "$(case_frame python_oracle)" | "$runtime_a" || true)"
if [[ "$(output_code "$python_output")" == "$allow_code" ]]; then
  "$python_sentinel"
fi
[[ ! -e "$python_executed" ]] || fail 'Python oracle crossed the action 9031 refusal'

sabotage() {
  local label="$1" rule="$2" witness="$3"
  local sabotaged_module="$work/$label.sio"
  local combined="$work/$label-combined.sio"
  local sabotaged_runtime="$work/$label-runtime"
  grep -Fqx "$rule" "$MODULE" || fail "$label rule is absent or changed"
  grep -Fvx "$rule" "$MODULE" > "$sabotaged_module"
  sed -n '1,$p' "$sabotaged_module" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$combined" \
    -o "$sabotaged_runtime" >/dev/null
  chmod 0755 "$sabotaged_runtime"
  local output
  output="$(printf '%s\n' "$(case_frame "$witness")" | "$sabotaged_runtime" || true)"
  [[ "$(output_code "$output")" == "$allow_code" ]] ||
    fail "$label sabotage did not admit its unchanged Sounio witness: $output"
}

sabotage parent-decision-chain \
  '    if parents.action_9025_allow != 1 || parents.action_9025_same_uid_peer_isolation != 1 || parents.action_9025_material_execution != 1 || (transition.operation == 1 && (parents.action_9030_issue_allow != 1 || parents.action_9030_consume_allow != 0 || parents.action_9030_close_allow != 0 || parents.action_9030_revoke_allow != 0)) || (transition.operation == 2 && (parents.action_9030_issue_allow != 1 || parents.action_9030_consume_allow != 1 || parents.action_9030_close_allow != 0 || parents.action_9030_revoke_allow != 0)) || (transition.operation == 3 && (parents.action_9030_issue_allow != 1 || parents.action_9030_consume_allow != 1 || parents.action_9030_close_allow != 1 || parents.action_9030_revoke_allow != 0)) || (transition.operation == 4 && (parents.action_9030_issue_allow != 1 || parents.action_9030_close_allow != 0 || parents.action_9030_revoke_allow != 1 || (transition.current_state == 1 && parents.action_9030_consume_allow != 0) || (transition.current_state == 2 && parents.action_9030_consume_allow != 1))) { return 502 }' \
  current_material
sabotage kernel-anchor \
  '    if kernel.boot_id_bound != 1 || kernel.bpf_object_bound != 1 || kernel.bpf_programs_bound != 1 || kernel.bpf_links_active != 1 || kernel.bpf_maps_bound != 1 || kernel.bpf_epoch_equal != 1 || kernel.resident_sounio_bound != 1 || kernel.guardian_executable_bound != 1 || kernel.mediation_hooks_attached != 3 || !loom_kernel_peer_activation_digest_nonzero(bindings.kernel_anchor_hash) { return 503 }' \
  kernel_drift
sabotage principal-anchor \
  '    if principal.so_peercred_bound != 1 || principal.pidfd_bound != 1 || principal.start_tick_bound != 1 || principal.namespaces_bound != 1 || principal.cgroup_bound != 1 || principal.harness_ancestry_bound != 1 || principal.command_environment_worktree_bound != 1 || principal.operation_equal != 1 || !loom_kernel_peer_activation_digest_nonzero(bindings.principal_anchor_hash) { return 504 }' \
  principal_drift
sabotage prewrite-lifecycle \
  '    if shape.prewrite_validated != 1 || shape.closed_domain != 1 || shape.legal_transition != 1 || shape.generations_monotonic != 1 || shape.old_state_receipt_bound != 1 || shape.proposed_receipt_bound != 1 || shape.future_terminal_obligation_bound != 1 || shape.no_mutation_on_deny != 1 || transition.broker_epoch <= 0 || transition.lease_generation <= 0 || transition.custody_generation <= 0 || transition.capsule_generation <= 0 || transition.grant_generation <= 0 || transition.request_sequence <= 0 || transition.deadline_tick <= 0 || transition.budget_remaining <= 0 || (transition.operation == 1 && (transition.current_state != 0 || transition.next_state != 1)) || (transition.operation == 2 && (transition.current_state != 1 || transition.next_state != 2)) || (transition.operation == 3 && (transition.current_state != 2 || transition.next_state != 3)) || (transition.operation == 4 && ((transition.current_state != 1 && transition.current_state != 2) || transition.next_state != 4)) || !loom_kernel_peer_activation_digest_nonzero(bindings.proposed_transition_hash) { return 505 }' \
  postwrite
sabotage nonbearer-custody \
  '    if custody.handle_lookup_only != 1 || custody.authenticated_before_lookup != 1 || custody.single_writer != 1 || custody.atomic_compare_exchange != 1 || custody.in_memory_only != 1 || custody.capsule_non_authorizing != 1 || custody.generation_fresh != 1 || custody.replay_isolated != 1 || !loom_kernel_peer_activation_digest_nonzero(bindings.capsule_identity_hash) { return 506 }' \
  bearer
sabotage affirmative-absence \
  '    if terminal.policy_failure_closed != 1 || terminal.boot_loss_poison != 1 || terminal.bpf_loss_poison != 1 || terminal.guardian_loss_poison != 1 || terminal.timeout_poison != 1 || terminal.silence_rejected != 1 || ((transition.operation == 1 || transition.operation == 2) && (terminal.effect_closure_bound != 0 || terminal.action_9030_terminal_bound != 0 || terminal.registry_absent != 0 || terminal.kernel_extinct != 0 || terminal.replay_refused != 0)) || ((transition.operation == 3 || transition.operation == 4) && (terminal.effect_closure_bound != 1 || terminal.action_9030_terminal_bound != 1 || terminal.registry_absent != 1 || terminal.kernel_extinct != 1 || terminal.replay_refused != 1)) { return 507 }' \
  silent_absence
sabotage authority-separation \
  '    if authority.producer_sounio != 1 || authority.expected_result_sounio != 1 || authority.python_absent != 1 || authority.rust_absent != 1 || authority.review_unpromoted != 1 || authority.parity_unpromoted != 1 || authority.disposable_oracle_absent != 1 { return 508 }' \
  python_oracle
sabotage provenance-binding \
  '    if evidence.provenance_complete != 1 || evidence.receipt_bound != 1 || !loom_kernel_peer_activation_digest_nonzero(bindings.source_semantics_toolchain_result_hash) { return 509 }' \
  unbound_result
sabotage sabotage-completeness \
  '    if evidence.sabotage_count != 9 || evidence.sabotage_required != 9 { return 510 }' \
  incomplete_sabotage

dependencies="$(ldd "$runtime_a" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eiq 'python|rust'; then
  fail "prohibited runtime dependency detected: $dependencies"
fi

printf 'sounio-loom-kernel-peer-activation-capsule-authority-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9031 cases=16 positive=ALLOWx4 current_material=DENY502 kernel=DENY503 principal=DENY504 lifecycle=DENY505 nonbearer=DENY506 absence_triplet=DENY507 authority_laundering=DENY508 provenance=DENY509 sabotage=DENY510 malformed=DENY424 causal_sabotage=ALLOWx9 builds=2 deterministic_binary=true deterministic_fixtures=true parent_9025_sha256=%s parent_9030_sha256=%s python_executed=false rust_executed=false operational_realization=false capsule_material=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$PARENT_9025")" "$(file_hash "$PARENT_9030")"
