#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-effect-closure.XXXXXX")"
RUNTIME="$TEST_ROOT/effect-closure-authority"
MODULE="$ROOT_DIR/stdlib/coordination/loom_effect_closure_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/effect_closure_authority_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-effect-closure-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_EFFECT_CLOSURE_SELFTEST PASS cases=19' ]] ||
  fail "unexpected Sounio selftest: $selftest"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
all_bindings="$one $one $one $one $one $one $one $one $one $one $one"
closed_coverage='3 3 3 2 2 2 2 2 2 2 2 2'

valid="9025 3 1 1 1 1 1 1 1 1 12 12 1 $closed_coverage $all_bindings"
wrong_stage="9025 2 1 1 1 1 1 1 1 1 12 12 1 $closed_coverage $all_bindings"
orphan="9025 3 0 1 1 1 1 1 1 1 12 12 1 $closed_coverage $all_bindings"
decision_only="9025 3 1 1 1 1 1 1 1 1 12 12 1 3 3 1 2 2 2 2 2 2 2 2 2 $all_bindings"
short_universe="9025 3 1 1 1 1 1 1 1 1 11 12 1 $closed_coverage $all_bindings"
incomplete_sabotage="9025 3 1 1 1 1 1 1 1 1 12 11 1 $closed_coverage $all_bindings"
not_fail_closed="9025 3 1 1 1 1 0 1 1 1 12 12 1 $closed_coverage $all_bindings"
unsafe_path="9025 3 1 1 1 1 1 1 1 0 12 12 1 $closed_coverage $all_bindings"
peer_unsafe="9025 3 1 1 1 1 1 1 0 1 12 12 1 $closed_coverage $all_bindings"
unknown_mediated="9025 3 1 1 1 1 1 1 1 1 12 12 1 3 3 3 2 2 2 2 2 2 2 2 3 $all_bindings"
unsupported_arch="9025 3 1 1 1 0 1 1 1 1 12 12 1 $closed_coverage $all_bindings"
unbound_coverage="9025 3 1 1 1 1 1 1 1 1 12 12 1 $closed_coverage $one $one $one $one $one $zero $one $one $one $one $one"
current_material="9025 3 1 1 1 1 1 1 0 0 12 0 1 3 3 1 1 0 0 0 0 0 0 0 2 $one $one $one $one $one $one $zero $one $one $one $one"

assert_output() {
  local label="$1" frame="$2" expected="$3"
  local actual
  actual="$(printf '%s\n' "$frame" | "$RUNTIME" || true)"
  [[ "$actual" == "$expected" ]] || fail "$label: $actual"
}

assert_output valid "$valid" \
  'SOUNIO_EFFECT_CLOSURE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output wrong-stage "$wrong_stage" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=405 reason=wrong-stage stage=SOUNIO_EXECUTABLE'
assert_output orphan "$orphan" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=446 reason=parent-chain-invalid stage=SEMANTICS_FROZEN'
assert_output decision-only "$decision_only" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=447 reason=material-coverage-incomplete stage=SEMANTICS_FROZEN'
assert_output short-universe "$short_universe" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=447 reason=material-coverage-incomplete stage=SEMANTICS_FROZEN'
assert_output incomplete-sabotage "$incomplete_sabotage" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=448 reason=sabotage-incomplete stage=SEMANTICS_FROZEN'
assert_output fail-closed "$not_fail_closed" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=449 reason=revocation-or-race-unsafe stage=SEMANTICS_FROZEN'
assert_output path-race "$unsafe_path" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=449 reason=revocation-or-race-unsafe stage=SEMANTICS_FROZEN'
assert_output same-uid "$peer_unsafe" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=451 reason=same-uid-peer-isolation-absent stage=SEMANTICS_FROZEN'
assert_output unknown "$unknown_mediated" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=452 reason=unknown-effect-not-kernel-denied stage=SEMANTICS_FROZEN'
assert_output architecture "$unsupported_arch" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=453 reason=architecture-unsupported stage=SEMANTICS_FROZEN'
assert_output provenance "$unbound_coverage" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=450 reason=provenance-incomplete stage=SEMANTICS_FROZEN'
assert_output current-material "$current_material" \
  'SOUNIO_EFFECT_CLOSURE_DENY code=447 reason=material-coverage-incomplete stage=SEMANTICS_FROZEN'
assert_output malformed '9025 3' \
  'SOUNIO_EFFECT_CLOSURE_DENY code=424 reason=malformed-frame stage=INVALID'

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
  [[ "$actual" == 'SOUNIO_EFFECT_CLOSURE_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
    fail "$label sabotage did not admit its unchanged witness: $actual"
}

sabotage frozen-parent \
  '    if facts.parent_9023_frozen != 1 || facts.parent_9024_frozen != 1 || facts.resident_runtime_bound != 1 || !loom_effect_closure_digest_nonzero(bindings.parent_9023_manifest_hash) || !loom_effect_closure_digest_nonzero(bindings.parent_9024_manifest_hash) || !loom_effect_closure_digest_nonzero(bindings.resident_runtime_hash) { return 446 }' \
  "$orphan"
sabotage material-coverage \
  '    if facts.coverage_family_count != 12 || !loom_effect_closure_material_mode(coverage.executable_mode) || !loom_effect_closure_material_mode(coverage.process_mode) || !loom_effect_closure_material_mode(coverage.path_mode) || !loom_effect_closure_material_mode(coverage.descriptor_mode) || !loom_effect_closure_material_mode(coverage.mapped_storage_mode) || !loom_effect_closure_material_mode(coverage.async_io_mode) || !loom_effect_closure_material_mode(coverage.network_mode) || !loom_effect_closure_material_mode(coverage.unix_socket_mode) || !loom_effect_closure_material_mode(coverage.ipc_mode) || !loom_effect_closure_material_mode(coverage.device_mode) || !loom_effect_closure_material_mode(coverage.procfs_mode) { return 447 }' \
  "$decision_only"
sabotage sabotage-count \
  '    if facts.sabotage_family_count != 12 { return 448 }' \
  "$incomplete_sabotage"
sabotage same-uid \
  '    if facts.same_uid_peer_isolation != 1 { return 451 }' \
  "$peer_unsafe"
sabotage provenance \
  '    if facts.receipt_bound != 1 || !loom_effect_closure_digest_nonzero(bindings.source_hash) || !loom_effect_closure_digest_nonzero(bindings.semantics_hash) || !loom_effect_closure_digest_nonzero(bindings.coverage_manifest_hash) || !loom_effect_closure_digest_nonzero(bindings.sabotage_receipt_hash) || !loom_effect_closure_digest_nonzero(bindings.toolchain_hash) || !loom_effect_closure_digest_nonzero(bindings.hardware_hash) || !loom_effect_closure_digest_nonzero(bindings.command_hash) || !loom_effect_closure_digest_nonzero(bindings.result_hash) { return 450 }' \
  "$unbound_coverage"

printf '%s\n' \
  'sounio-loom-effect-closure-authority-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9025 cases=19 positive=ALLOW current_material=DENY447 decision_only=DENY447 sabotage_incomplete=DENY448 unsafe=DENY449 provenance=DENY450 same_uid=DENY451 unknown=DENY452 architecture=DENY453 malformed=DENY424 causal_sabotage=ALLOWx5'
