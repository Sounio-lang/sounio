#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/causal_workflow_kernel.freeze.v1"

fail() {
  printf 'sounio-loom-causal-workflow-kernel-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  local key="$1" expected="$2" actual
  actual="$(manifest_value "$key")"
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

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] ||
  fail 'freeze manifest is absent or linked'
expect_value schema loom-causal-workflow-kernel-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9037
expect_value concept_id SOUNIO-LOOM-CAUSAL-WORKFLOW-KERNEL
expect_value state_count 12
expect_value refusal_range 580-589
expect_value causal_sabotage PASS
expect_value load_bearing_rule observed_predecessor_receipt_equals_current_journal_head
expect_value run_ticket_is_bearer false
expect_value run_ticket_is_execution_authority false
expect_value launch_authority action-9030
expect_value exactly_once_scope live-HostGuardian-generation
expect_value guardian_host_or_store_loss fail-closed
expect_value ocaml_journal_attached false
expect_value hostguardian_attachment false
expect_value controller_loss_measured false
expect_value pod_loss_measured false
expect_value dynamic_user_workflow_attached false
expect_value material_execution false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

for key in garden contract concept_registry source entrypoint build_script \
  selftest first_manifest first_evidence freeze_evidence parent_9030_manifest \
  parent_9031_manifest parent_9032_manifest parent_9033_manifest \
  parent_9034_manifest parent_9035_manifest parent_9036_manifest \
  toolchain_wrapper toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

garden_commit="$(manifest_value garden_commit)"
expect_commit_hash "$garden_commit" \
  "$(manifest_value garden_path)" "$(manifest_value garden_sha256)"

source_commit="$(manifest_value sounio_executable_commit)"
for key in garden contract source entrypoint build_script selftest; do
  expect_commit_hash "$source_commit" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

first_commit="$(manifest_value first_receipt_commit)"
for key in first_manifest first_evidence concept_registry; do
  expect_commit_hash "$first_commit" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

[[ "$(sha256sum "$ROOT_DIR/$(manifest_value canonical_source_path)" | cut -d ' ' -f 1)" == \
   "$(manifest_value canonical_source_sha256)" ]] ||
  fail 'canonical Sounio source drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-workflow-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_CAUSAL_WORKFLOW_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_kernel_fixture.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" ||
  fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] ||
  fail 'Sounio executable hash drifted'

final_frame="$(manifest_value wire_schema) $(manifest_value final_word0) $(manifest_value final_word1)"
recovery_frame="$(manifest_value wire_schema) $(manifest_value recovery_word0) $(manifest_value recovery_word1)"
final_output="$(printf '%s\n' "$final_frame" | "$work/runtime-one")"
recovery_output="$(printf '%s\n' "$recovery_frame" | "$work/runtime-one")"
[[ "${final_output%%$'\n'*}" == "$(manifest_value final_decision)" ]] ||
  fail 'final decision drifted'
[[ "${recovery_output%%$'\n'*}" == "$(manifest_value recovery_decision)" ]] ||
  fail 'recovery decision drifted'
[[ "$(printf '%s' "$final_output" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value final_output_sha256)" ]] || fail 'final output drifted'
[[ "$(printf '%s' "$recovery_output" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value recovery_output_sha256)" ]] || fail 'recovery output drifted'
[[ "$(printf '%s\n%s' "$final_output" "$recovery_output" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value semantics_sha256)" ]] || fail 'frozen semantics drifted'

for control in graph parent edge; do
  frame="$(manifest_value wire_schema) $(manifest_value "${control}_word0") $(manifest_value final_word1)"
  set +e
  output="$(printf '%s\n' "$frame" | "$work/runtime-one")"
  code=$?
  set -e
  [[ $code -eq 42 && "$output" == "$(manifest_value "${control}_decision")" ]] ||
    fail "$control control drifted"
done
for control in lineage ticket replay terminal; do
  frame="$(manifest_value wire_schema) $(manifest_value final_word0) $(manifest_value "${control}_word1")"
  set +e
  output="$(printf '%s\n' "$frame" | "$work/runtime-one")"
  code=$?
  set -e
  [[ $code -eq 42 && "$output" == "$(manifest_value "${control}_decision")" ]] ||
    fail "$control control drifted"
done
for control in recompile successor predecessor; do
  frame="$(manifest_value wire_schema) $(manifest_value recovery_word0) $(manifest_value "${control}_word1")"
  set +e
  output="$(printf '%s\n' "$frame" | "$work/runtime-one")"
  code=$?
  set -e
  [[ $code -eq 42 && "$output" == "$(manifest_value "${control}_decision")" ]] ||
    fail "$control control drifted"
done

MUTANT_MODULE="$work/mutant.sio"
sed 's/if observation.predecessor_receipt_equal != 1 {/if false {/' \
  "$ROOT_DIR/$(manifest_value source_path)" > "$MUTANT_MODULE"
MUTANT_RUNTIME="$work/mutant-runtime"
SOUNIO_LOOM_CAUSAL_WORKFLOW_MODULE="$MUTANT_MODULE" \
SOUNIO_LOOM_CAUSAL_WORKFLOW_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_kernel_fixture.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'causal mutant did not build'
mutant_frame="$(manifest_value wire_schema) $(manifest_value recovery_word0) $(manifest_value predecessor_word1)"
[[ "$(printf '%s\n' "$mutant_frame" | "$MUTANT_RUNTIME")" == "$recovery_output" ]] ||
  fail 'causal mutant did not admit the unchanged predecessor witness'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_causal_workflow_kernel_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'first result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'first result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value first_evidence_path)" ||
  fail 'first evidence is not the exact executable result'

freeze_result="sounio-loom-causal-workflow-kernel-freeze-selftest: PASS semantic_authority=Sounio action=9037 stage=SEMANTICS_FROZEN semantics_sha256=$(manifest_value semantics_sha256) executable_sha256=$(manifest_value executable_sha256) final=ADVANCE recovery=RECOVER refusals=DENY580-589 malformed=DENY424 causal_sabotage=PASS source=tests/verify-ir/call_b.sio expected_exit=0 expected_stdout=empty expected_stderr=empty launch_authority=action-9030 run_ticket_bearer=false run_ticket_execution_authority=false exactly_once_scope=live-HostGuardian-generation guardian_loss=fail-closed expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false ocaml_journal_attached=false hostguardian_attachment=false controller_loss_measured=false pod_loss_measured=false dynamic_user_workflow_attached=false material_execution=false production_activation=false parity_open=false claim_ready=false"
printf '%s\n' "$freeze_result" | cmp - "$ROOT_DIR/$(manifest_value freeze_evidence_path)" ||
  fail 'freeze evidence is not the exact frozen result'
printf '%s\n' "$freeze_result"
