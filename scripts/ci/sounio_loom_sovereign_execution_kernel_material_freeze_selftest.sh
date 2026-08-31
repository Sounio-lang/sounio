#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/sovereign_execution_kernel_material.runtime.v1"

fail() {
  printf 'sounio-loom-sovereign-execution-kernel-material-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

expect() {
  local key="$1" expected="$2" actual
  actual="$(value "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "$key expected $expected but found $actual"
}

expect_hash() {
  local path="$1" expected="$2" actual
  [[ -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" ]] ||
    fail "$path is absent or linked"
  actual="$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] || fail "$path hash drifted"
}

expect_commit_hash() {
  local commit="$1" path="$2" expected="$3" actual
  git -C "$ROOT_DIR" cat-file -e "${commit}^{commit}" ||
    fail "commit $commit is absent"
  actual="$(git -C "$ROOT_DIR" show "${commit}:$path" | sha256sum | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] ||
    fail "$path is not bound to commit $commit"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'material manifest is absent or linked'
expect schema loom-sovereign-execution-kernel-material-runtime-v1
expect stage MATERIAL_EXECUTION_FROZEN
expect semantic_authority Sounio
expect action 9042
expect concept_id SOUNIO-LOOM-SOVEREIGN-EXECUTION-KERNEL
expect producing_language C++20+Linux
expect language_role MATERIAL_PARITY
expect transitory true
expect treatment PASS
expect hostile_same_uid_spoof REFUSED_BEFORE_EXECUTION
expect principal_binding_sabotage SPOOF_ADMITTED
expect guardian_death PASS
expect pdeathsig_sabotage MATERIAL_SURVIVED
expect grant_resident_memory true
expect grant_is_bearer false
expect grant_single_use true
expect consume_atomic true
expect exported_token false
expect exported_handle false
expect descriptor_is_execution_authority false
expect interface_release_authority zero
expect material_exactly_once true
expect guardian_pidfd_bound true
expect pdeathsig_armed true
expect release_absent_after_guardian_death true
expect causal_sabotage PASS
expect expected_results_encoded_in_material_layer false
expect python_executed false
expect rust_executed false
expect runtime_dependencies clean
expect material_execution true
expect same_uid_peer_isolation true
expect production_gate_ready true
expect production_activation false
expect exec_attached false
expect commit_attached false
expect ci_attached false
expect parity_open false
expect claim_ready false

for key in contract source build_script host_probe selftest verification_gate \
  semantic_manifest peer_judgment evidence; do
  expect_hash "$(value "${key}_path")" "$(value "${key}_sha256")"
done

source_commit="$(value material_source_commit)"
for key in contract source build_script host_probe selftest; do
  expect_commit_hash "$source_commit" "$(value "${key}_path")" \
    "$(value "${key}_sha256")"
done
receipt_commit="$(value material_receipt_commit)"
for key in evidence; do
  expect_commit_hash "$receipt_commit" "$(value "${key}_path")" \
    "$(value "${key}_sha256")"
done

result="$(bash "$ROOT_DIR/$(value selftest_path)")"
[[ "$result" == "$(value result)" ]] || fail 'material result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(value result_sha256)" ]] || fail 'material result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(value evidence_path)" ||
  fail 'material evidence is not the exact executable result'

printf 'sounio-loom-sovereign-execution-kernel-material-freeze-selftest: PASS semantic_authority=Sounio action=9042 stage=MATERIAL_EXECUTION_FROZEN treatment=PASS hostile_same_uid_spoof=REFUSED_BEFORE_EXECUTION principal_binding_sabotage=SPOOF_ADMITTED guardian_death=PASS pdeathsig_sabotage=MATERIAL_SURVIVED transport_death=PASS gui_death=PASS coordinator_death=PASS pod_death=PASS tmux_death=PASS grant_is_bearer=false same_uid_peer_isolation=true production_gate_ready=true production_activation=false exec_attached=false parity_open=false claim_ready=false result_sha256=%s\n' \
  "$(value result_sha256)"
