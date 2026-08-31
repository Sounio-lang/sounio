#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/native_hook_cutover.freeze.v1"

fail() {
  printf 'sounio-loom-native-hook-cutover-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  [[ -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" ]] || fail "$path is absent or linked"
  [[ "$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "$path hash drifted"
}

expect_commit_hash() {
  local commit="$1" path="$2" expected="$3"
  git -C "$ROOT_DIR" cat-file -e "${commit}^{commit}" || fail "commit $commit is absent"
  [[ "$(git -C "$ROOT_DIR" show "${commit}:$path" | sha256sum | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "$path is not bound to commit $commit"
}

expect_value schema loom-native-hook-cutover-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value semantic_authority Sounio
expect_value action 9045
expect_value parent_action 9044-frozen
expect_value load_bearing_rule python_bridge_absent
expect_value fail_closed_required true
expect_value decision_receipt_required true
expect_value provider_config_native_required true
expect_value python_bridge_absent_required true
expect_value rust_bridge_absent_required true
expect_value disposable_oracle_absent_required true
expect_value python_executed false
expect_value rust_executed false
expect_value disposable_oracle_executed false
expect_value python_bridge_absent_from_package false
expect_value native_configs_promoted false
expect_value parity_open false
expect_value four_provider_canary false
expect_value claim_ready false

for key in garden source entrypoint build_script selftest first_manifest \
  first_evidence parent_9044_freeze toolchain_wrapper toolchain_compiler; do
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

grep -qx 'action=9044' "$ROOT_DIR/$(manifest_value parent_9044_freeze_path)" ||
  fail 'parent action drifted'
grep -qx 'stage=SEMANTICS_FROZEN' "$ROOT_DIR/$(manifest_value parent_9044_freeze_path)" ||
  fail 'parent is not frozen'

actual_semantics="$(cat "$ROOT_DIR/$(manifest_value source_path)" \
  "$ROOT_DIR/$(manifest_value entrypoint_path)" | sha256sum | cut -d ' ' -f 1)"
[[ "$actual_semantics" == "$(manifest_value semantics_sha256)" ]] || fail 'semantics hash drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-native-hook-cutover-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_NATIVE_HOOK_CUTOVER_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_cutover.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" || fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == "$(manifest_value executable_sha256)" ]] ||
  fail 'Sounio executable hash drifted'

expect_runtime() {
  local frame="$1" expected="$2" observed
  observed="$(printf '%s\n' "$frame" | "$work/runtime-one")"
  [[ "$observed" == "$expected" ]] || fail "frozen decision diverged: $observed"
}

base="$(manifest_value session_word)"
pretool="$(manifest_value pretool_word)"
stage="$(manifest_value stage_word)"
s0="$(manifest_value semantic_hash0)"
s1="$(manifest_value semantic_hash1)"
oh="$(manifest_value ocaml_hash_probe)"
ch="$(manifest_value config_hash_probe)"
sc="$(manifest_value sabotage_count)"
sr="$(manifest_value sabotage_required)"
hook_expected="$(manifest_value hook_decision)"
expect_runtime "9045 1 $stage 1 1 1 $base 0 $s0 $s1 $oh $ch $sc $sr" "$hook_expected"
expect_runtime "9045 1 $stage 2 1 1 $base 0 $s0 $s1 $oh $ch $sc $sr" "$hook_expected"
expect_runtime "9045 1 $stage 3 2 1 $base 0 $s0 $s1 $oh $ch $sc $sr" "$hook_expected"
expect_runtime "9045 1 $stage 4 3 1 $base 0 $s0 $s1 $oh $ch $sc $sr" "$hook_expected"
expect_runtime "9045 1 $stage 1 1 3 $pretool 0 $s0 $s1 $oh $ch $sc $sr" "$hook_expected"
expect_runtime "9045 2 $stage 3 2 1 $(manifest_value provider_canary_word) 4 $s0 $s1 $oh $ch $sc $sr" \
  "$(manifest_value canary_decision)"
expect_runtime "9045 3 $stage 0 0 0 $(manifest_value claim_word) $(manifest_value four_provider_mask) $s0 $s1 $oh $ch $sc $sr" \
  "$(manifest_value claim_decision)"

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_native_hook_cutover_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'first result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == "$(manifest_value result_sha256)" ]] ||
  fail 'first result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value first_evidence_path)" ||
  fail 'first evidence is not the exact executable result'

freeze_result="sounio-loom-native-hook-cutover-freeze-selftest: PASS semantic_authority=Sounio action=9045 stage=SEMANTICS_FROZEN semantics_sha256=$(manifest_value semantics_sha256) executable_sha256=$(manifest_value executable_sha256) parent=9044-frozen providers=codex,claude,cursor,grok provider_dialects=LOAD_BEARING python_absence=LOAD_BEARING causal_sabotage=PASS python_executed=false rust_executed=false disposable_oracle_executed=false parity_open=false four_provider_canary=false claim_ready=false"
printf '%s\n' "$freeze_result" | cmp - "$ROOT_DIR/$(manifest_value freeze_evidence_path)" ||
  fail 'freeze evidence is not the exact frozen result'
printf '%s\n' "$freeze_result"
