#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/native_hook_generation_reconcile.freeze.v1"

fail() {
  printf 'sounio-loom-native-hook-generation-reconcile-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

expect_value() {
  [[ "$(manifest_value "$1")" == "$2" ]] || fail "$1 diverged"
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" ]] ||
    fail "$path is absent or linked"
  [[ "$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "$path hash drifted"
}

expect_commit_hash() {
  local commit="$1" path="$2" expected="$3"
  git -C "$ROOT_DIR" cat-file -e "${commit}^{commit}" || fail "commit $commit is absent"
  [[ "$(git -C "$ROOT_DIR" show "${commit}:$path" | sha256sum | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "$path is not bound to commit $commit"
}

expect_value schema loom-native-hook-generation-reconcile-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value semantic_authority Sounio
expect_value action 9047
expect_value parent_action 9046-frozen
expect_value absence_triple record_identity+kernel_identity+causal_absence
expect_value load_bearing_rule pid_absent
expect_value python_oracle_attempt DENY687
expect_value fail_closed_required true
expect_value decision_receipt_required true
expect_value record_identity_bound_required true
expect_value kernel_identity_bound_required true
expect_value related_artifact_coverage_required true
expect_value quarantine_wal_required true
expect_value python_executed false
expect_value rust_executed false
expect_value disposable_oracle_executed false
expect_value parity_open false
expect_value native_entry_open false
expect_value quarantine_committed false
expect_value cutover_ready false
expect_value bridge_free_current false

for key in garden source entrypoint build_script selftest freeze_selftest \
  first_manifest first_evidence parent_9046_freeze toolchain_wrapper \
  toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" "$(manifest_value "${key}_sha256")"
done

expect_commit_hash "$(manifest_value garden_commit)" "$(manifest_value garden_path)" \
  "$(manifest_value garden_sha256)"
source_commit="$(manifest_value sounio_executable_commit)"
for key in source entrypoint build_script selftest; do
  expect_commit_hash "$source_commit" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done
first_commit="$(manifest_value first_receipt_commit)"
for key in first_manifest first_evidence; do
  expect_commit_hash "$first_commit" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

grep -qx 'action=9046' "$ROOT_DIR/$(manifest_value parent_9046_freeze_path)" ||
  fail 'parent action drifted'
grep -qx 'stage=SEMANTICS_FROZEN' "$ROOT_DIR/$(manifest_value parent_9046_freeze_path)" ||
  fail 'parent is not frozen'

actual_semantics="$(cat "$ROOT_DIR/$(manifest_value source_path)" \
  "$ROOT_DIR/$(manifest_value entrypoint_path)" | sha256sum | cut -d ' ' -f 1)"
[[ "$actual_semantics" == "$(manifest_value semantics_sha256)" ]] ||
  fail 'semantics hash drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook-generation-reconcile-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_NATIVE_HOOK_GENERATION_RECONCILE_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_generation_reconcile.sh" \
      >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" || fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
    "$(manifest_value executable_sha256)" ]] || fail 'Sounio executable hash drifted'

expect_runtime() {
  local frame="$1" expected="$2" observed
  observed="$(printf '%s\n' "$frame" | "$work/runtime-one")"
  [[ "$observed" == "$expected" ]] || fail "frozen decision diverged: $observed"
}

expect_refusal() {
  local frame="$1" expected="$2" observed code
  set +e
  observed="$(printf '%s\n' "$frame" | "$work/runtime-one")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == "$expected" ]] ||
    fail "frozen refusal diverged: code=$code result=$observed"
}

expect_runtime \
  "9047 1 $(manifest_value stage_word) $(manifest_value live_word) 0 0 0 1 2 3 4 $(manifest_value sabotage_count) $(manifest_value sabotage_required)" \
  "$(manifest_value live_decision)"
expect_runtime \
  "9047 1 $(manifest_value stage_word) $(manifest_value heartbeat_only_word) 0 0 0 1 2 3 4 $(manifest_value sabotage_count) $(manifest_value sabotage_required)" \
  "$(manifest_value heartbeat_only_decision)"
expect_runtime \
  "9047 1 $(manifest_value stage_word) $(manifest_value pid_absent_word) 2 2 3 1 2 3 4 $(manifest_value sabotage_count) $(manifest_value sabotage_required)" \
  "$(manifest_value quarantine_eligible_decision)"
expect_runtime \
  "9047 2 $(manifest_value stage_word) $(manifest_value pid_absent_word) 2 2 3 1 2 3 4 $(manifest_value sabotage_count) $(manifest_value sabotage_required)" \
  "$(manifest_value quarantine_ready_decision)"
expect_refusal \
  "9047 2 $(manifest_value stage_word) $(manifest_value python_oracle_attempt_word) 2 2 3 1 2 3 4 $(manifest_value sabotage_count) $(manifest_value sabotage_required)" \
  "$(manifest_value python_oracle_refusal)"
expect_refusal \
  "9047 2 $(manifest_value stage_word) $(manifest_value live_word) 2 2 3 1 2 3 4 $(manifest_value sabotage_count) $(manifest_value sabotage_required)" \
  "$(manifest_value causal_absence_refusal)"
expect_refusal \
  "9047 2 $(manifest_value stage_word) $(manifest_value transaction_incomplete_word) 2 2 3 1 2 3 4 $(manifest_value sabotage_count) $(manifest_value sabotage_required)" \
  "$(manifest_value transaction_refusal)"

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_native_hook_generation_reconcile_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'first result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
    "$(manifest_value result_sha256)" ]] || fail 'first result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value first_evidence_path)" ||
  fail 'first evidence is not the exact executable result'

freeze_result="sounio-loom-native-hook-generation-reconcile-freeze-selftest: PASS semantic_authority=Sounio action=9047 stage=SEMANTICS_FROZEN semantics_sha256=$(manifest_value semantics_sha256) executable_sha256=$(manifest_value executable_sha256) parent=9046-frozen absence_triple=RECORD+KERNEL+CAUSE load_bearing_rule=pid_absent python_oracle_attempt=DENY687 causal_sabotage=pid-absence-rule-removed python_executed=false rust_executed=false disposable_oracle_executed=false parity_open=false native_entry_open=false quarantine_committed=false cutover_ready=false bridge_free_current=false"
printf '%s\n' "$freeze_result" | cmp - "$ROOT_DIR/$(manifest_value freeze_evidence_path)" ||
  fail 'freeze evidence is not the exact frozen result'
printf '%s\n' "$freeze_result"
