#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/sovereign_change_kernel.freeze.v1"

fail() {
  printf 'sounio-loom-sovereign-change-kernel-freeze-selftest: FAIL: %s\n' "$*" >&2
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

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is absent or linked'
expect_value schema loom-sovereign-change-kernel-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value semantic_authority Sounio
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9043
expect_value concept_id SOUNIO-LOOM-SOVEREIGN-CHANGE-KERNEL
expect_value parent_action 9042-frozen+production-active
expect_value refusal_range 613-628
expect_value causal_sabotage PASS
expect_value load_bearing_rule ci_decision_consumed_not_reinterpreted
expect_value grant_resident_memory true
expect_value grant_is_bearer false
expect_value grant_single_use true
expect_value consume_atomic true
expect_value exported_token false
expect_value exported_handle false
expect_value descriptor_is_change_authority false
expect_value ci_policy consume-not-reinterpret
expect_value expected_results_encoded_in_material_layer false
expect_value python_executed false
expect_value rust_executed false
expect_value operational_attachment false
expect_value write_attached false
expect_value commit_attached false
expect_value ci_attached false
expect_value parity_open false
expect_value claim_ready false

for key in garden contract concept_registry concept_bindings source entrypoint \
  build_script selftest freeze_selftest first_manifest first_evidence \
  parent_9042_freeze parent_9042_product toolchain_wrapper toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

garden_commit="$(manifest_value garden_commit)"
expect_commit_hash "$garden_commit" "$(manifest_value garden_path)" \
  "$(manifest_value garden_sha256)"
source_commit="$(manifest_value sounio_executable_commit)"
for key in source entrypoint build_script selftest; do
  expect_commit_hash "$source_commit" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done
first_commit="$(manifest_value first_receipt_commit)"
for key in contract concept_registry concept_bindings first_manifest first_evidence; do
  expect_commit_hash "$first_commit" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

parent_freeze="$ROOT_DIR/$(manifest_value parent_9042_freeze_path)"
parent_product="$ROOT_DIR/$(manifest_value parent_9042_product_path)"
grep -qx 'action=9042' "$parent_freeze" || fail 'parent freeze action drifted'
grep -qx 'stage=SEMANTICS_FROZEN' "$parent_freeze" || fail 'parent is not frozen'
grep -qx 'production_activation=true' "$parent_product" ||
  fail 'parent production activation is absent'
grep -qx 'exec_attached=true' "$parent_product" ||
  fail 'parent execution attachment is absent'

actual_semantics="$(cat \
  "$ROOT_DIR/$(manifest_value source_path)" \
  "$ROOT_DIR/$(manifest_value entrypoint_path)" | sha256sum | cut -d ' ' -f 1)"
[[ "$actual_semantics" == "$(manifest_value semantics_sha256)" ]] ||
  fail 'frozen semantics hash drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-change-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_CHANGE_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_change_kernel.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" || fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'Sounio executable hash drifted'

frame_first_line() {
  printf '%s %s %s %s %s %s\n' \
    "$(manifest_value wire_schema)" \
    "$(manifest_value "$1_mode")" \
    "$(manifest_value stage_word)" \
    "$(manifest_value "$1_word")" \
    "$(manifest_value sabotage_count)" \
    "$(manifest_value sabotage_required)" | "$work/runtime-one" | sed -n '1p'
}

for decision in prepare consume commit ci production; do
  [[ "$(frame_first_line "$decision")" == \
     "$(manifest_value "${decision}_decision")" ]] ||
    fail "$decision decision drifted"
done

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_sovereign_change_kernel_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'first result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'first result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value first_evidence_path)" ||
  fail 'first evidence is not the exact executable result'

freeze_result="sounio-loom-sovereign-change-kernel-freeze-selftest: PASS semantic_authority=Sounio action=9043 stage=SEMANTICS_FROZEN semantics_sha256=$(manifest_value semantics_sha256) executable_sha256=$(manifest_value executable_sha256) prepare=CHANGE_PREPARED consume=CHANGE_CONSUMED commit=COMMIT_ADMIT ci=CI_ADMIT production=PRODUCTION_GATE_READY parent=9042-frozen+production-active grant=resident-memory+non-bearer+single-use+atomic intent=event+patch+worktree+HEAD+index+file-set ci_policy=consume-not-reinterpret causal_sabotage=PASS expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false operational_attachment=false write_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false"
printf '%s\n' "$freeze_result" | cmp - \
  "$ROOT_DIR/$(manifest_value freeze_evidence_path)" ||
  fail 'freeze evidence is not the exact frozen result'
printf '%s\n' "$freeze_result"
