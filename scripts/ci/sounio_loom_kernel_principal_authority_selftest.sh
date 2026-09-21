#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-principal.XXXXXX")"
RUNTIME="$TEST_ROOT/kernel-principal-authority"
MODULE="$ROOT_DIR/stdlib/coordination/loom_kernel_principal_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/kernel_principal_authority_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-principal-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_KERNEL_PRINCIPAL_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_principal_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_KERNEL_PRINCIPAL_SELFTEST PASS cases=17' ]] ||
  fail "unexpected Sounio selftest: $selftest"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
all_bindings="$one $one $one $one $one $one $one $one $one $one $one $one"
substrate='1 1 1 1 1 1'
mapping='1 1 1 1 1'
identities='1000 100000 100001 1000 100000 100001'
peer='1 1 1 1 1 1 1 1'
isolation='1 1 1 1 1 1'
grant='1 1 1 1 1'

valid="9026 3 1 $substrate $mapping $identities $peer $isolation $grant 5 5 1 $all_bindings"
wrong_stage="9026 2 1 $substrate $mapping $identities $peer $isolation $grant 5 5 1 $all_bindings"
unsupported="9026 3 1 0 1 1 1 1 1 $mapping $identities $peer $isolation $grant 5 5 1 $all_bindings"
no_allocation="9026 3 1 $substrate 0 0 0 0 1 1000 1000 1000 1000 1000 1000 $peer 0 0 0 0 0 0 $grant 0 5 1 $one $zero $zero $zero $one $zero $one $one $one $one $one $one"
bad_mapping="9026 3 1 $substrate 1 0 1 1 1 $identities $peer $isolation $grant 5 5 1 $all_bindings"
equal_principal="9026 3 1 $substrate $mapping 1000 1000 100001 1000 100000 100001 $peer $isolation $grant 5 5 1 $all_bindings"
bad_peer="9026 3 1 $substrate $mapping $identities 0 1 1 1 1 1 1 1 $isolation $grant 5 5 1 $all_bindings"
injectable="9026 3 1 $substrate $mapping $identities $peer 1 1 1 0 1 1 $grant 5 5 1 $all_bindings"
bad_grant="9026 3 1 $substrate $mapping $identities $peer $isolation 0 1 1 1 1 5 5 1 $all_bindings"
burnable="9026 3 1 $substrate $mapping $identities $peer $isolation 1 1 1 0 1 5 5 1 $all_bindings"
survives_crash="9026 3 1 $substrate $mapping $identities $peer $isolation 1 1 1 1 0 5 5 1 $all_bindings"
incomplete_sabotage="9026 3 1 $substrate $mapping $identities $peer $isolation $grant 4 5 1 $all_bindings"
unbound="9026 3 1 $substrate $mapping $identities $peer $isolation $grant 5 5 1 $one $one $one $one $one $zero $one $one $one $one $one $one"
current_material="$no_allocation"

assert_output() {
  local label="$1" frame="$2" expected="$3"
  local actual
  actual="$(printf '%s\n' "$frame" | "$RUNTIME" || true)"
  [[ "$actual" == "$expected" ]] || fail "$label: $actual"
}

assert_output valid "$valid" \
  'SOUNIO_KERNEL_PRINCIPAL_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output wrong-stage "$wrong_stage" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=405 reason=wrong-stage-or-parent stage=SOUNIO_EXECUTABLE'
assert_output unsupported "$unsupported" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=454 reason=kernel-substrate-incomplete stage=SEMANTICS_FROZEN'
assert_output allocation "$no_allocation" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=455 reason=subordinate-allocation-incomplete stage=SEMANTICS_FROZEN'
assert_output mapping "$bad_mapping" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=456 reason=namespace-mapping-invalid stage=SEMANTICS_FROZEN'
assert_output distinctness "$equal_principal" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=457 reason=kernel-principals-not-distinct stage=SEMANTICS_FROZEN'
assert_output peer "$bad_peer" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=458 reason=peer-binding-incomplete stage=SEMANTICS_FROZEN'
assert_output isolation "$injectable" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=459 reason=process-isolation-incomplete stage=SEMANTICS_FROZEN'
assert_output grant "$bad_grant" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=460 reason=grant-custody-incomplete stage=SEMANTICS_FROZEN'
assert_output nonburning "$burnable" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=460 reason=grant-custody-incomplete stage=SEMANTICS_FROZEN'
assert_output revocation "$survives_crash" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=460 reason=grant-custody-incomplete stage=SEMANTICS_FROZEN'
assert_output sabotage "$incomplete_sabotage" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=461 reason=sabotage-incomplete stage=SEMANTICS_FROZEN'
assert_output provenance "$unbound" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=462 reason=provenance-incomplete stage=SEMANTICS_FROZEN'
assert_output current-material "$current_material" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=455 reason=subordinate-allocation-incomplete stage=SEMANTICS_FROZEN'
assert_output wrong-action "${valid/9026 /9027 }" \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=405 reason=wrong-stage-or-parent stage=SEMANTICS_FROZEN'
assert_output malformed '9026 3' \
  'SOUNIO_KERNEL_PRINCIPAL_DENY code=424 reason=malformed-frame stage=INVALID'

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
  [[ "$actual" == 'SOUNIO_KERNEL_PRINCIPAL_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
    fail "$label sabotage did not admit its unchanged witness: $actual"
}

sabotage principal-distinctness \
  '    if mapping.outer_uid == mapping.principal_uid || mapping.outer_uid == mapping.sibling_uid || mapping.principal_uid == mapping.sibling_uid || mapping.outer_gid == mapping.principal_gid || mapping.outer_gid == mapping.sibling_gid || mapping.principal_gid == mapping.sibling_gid { return 457 }' \
  "$equal_principal"
sabotage injection-isolation \
  '    if isolation.non_dumpable != 1 || isolation.no_new_privileges != 1 || isolation.cap_sys_ptrace_absent != 1 || isolation.ptrace_denied != 1 || isolation.proc_mem_denied != 1 || isolation.cross_principal_signal_denied != 1 { return 459 }' \
  "$injectable"
sabotage nonburning-refusal \
  '    if grant.unauthorized_refusal_nonburning != 1 { return 460 }' \
  "$burnable"
sabotage crash-revocation \
  '    if grant.crash_revokes != 1 { return 460 }' \
  "$survives_crash"
sabotage sabotage-completeness \
  '    if grant.sabotage_count != 5 || grant.sabotage_required != 5 { return 461 }' \
  "$incomplete_sabotage"

printf '%s\n' \
  'sounio-loom-kernel-principal-authority-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9026 cases=17 positive=ALLOW current_material=DENY455 substrate=DENY454 allocation=DENY455 mapping=DENY456 distinctness=DENY457 peer=DENY458 isolation=DENY459 custody=DENY460 sabotage=DENY461 provenance=DENY462 malformed=DENY424 causal_sabotage=ALLOWx5 material_isolation=false'
