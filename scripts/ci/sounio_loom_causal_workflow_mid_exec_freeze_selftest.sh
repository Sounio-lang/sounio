#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/causal_workflow_mid_exec.freeze.v1"

fail() {
  printf 'sounio-loom-causal-workflow-mid-exec-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  local key="$1" expected="$2"
  [[ "$(manifest_value "$key")" == "$expected" ]] ||
    fail "$key expected $expected"
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
  actual="$(git -C "$ROOT_DIR" show "${commit}:${path}" | sha256sum | cut -d ' ' -f 1)" ||
    fail "$path is absent from build commit $commit"
  [[ "$actual" == "$expected" ]] ||
    fail "$path in build commit $commit does not match its frozen hash"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] ||
  fail 'freeze manifest is absent or linked'
expect_value schema loom-causal-workflow-mid-exec-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9037
expect_value subordinate_contract mid-exec-v1
expect_value concept_id SOUNIO-LOOM-CAUSAL-WORKFLOW-KERNEL
expect_value source_tree_state COMMITTED_SOURCE_INPUTS
expect_value release_stage RELEASE_ADMISSION
expect_value claim_stage CLAIM_CONTINUITY
expect_value load_bearing_release_rule exact_exec_started_witness_identity
expect_value load_bearing_claim_rule digest_bound_barrier_nonce_equality
expect_value causal_sabotage PASS
expect_value exact_counts compile+ticket+launch+result+attestation:1
expect_value material_execution false
expect_value pod_loss_measured false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

build_head="$(manifest_value build_head_commit)"
git -C "$ROOT_DIR" cat-file -e "${build_head}^{commit}" 2>/dev/null ||
  fail 'build_head_commit is not a local commit'
git -C "$ROOT_DIR" merge-base --is-ancestor "$build_head" HEAD ||
  fail 'build_head_commit is not an ancestor of the verifying checkout'
for key in contract source entrypoint build_script selftest; do
  expect_commit_hash "$build_head" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

for key in contract source entrypoint build_script selftest first_manifest \
  first_evidence freeze_evidence parent_9037_manifest toolchain_wrapper \
  toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-mid-exec-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_CAUSAL_MID_EXEC_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_mid_exec_fixture.sh" >/dev/null
done
runtime="$work/runtime-one"
cmp "$runtime" "$work/runtime-two" || fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$runtime" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'Sounio executable hash drifted'

release_frame="$(manifest_value wire_schema) $(manifest_value release_stage_word) $(manifest_value release_word0) $(manifest_value release_word1)"
claim_frame="$(manifest_value wire_schema) $(manifest_value claim_stage_word) $(manifest_value claim_word0) $(manifest_value claim_word1)"
release="$(printf '%s\n' "$release_frame" | "$runtime")"
claim="$(printf '%s\n' "$claim_frame" | "$runtime")"
[[ "${release%%$'\n'*}" == "$(manifest_value release_decision)" ]] ||
  fail 'release decision drifted'
[[ "${claim%%$'\n'*}" == "$(manifest_value claim_decision)" ]] ||
  fail 'claim decision drifted'
[[ "$(printf '%s' "$release" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value release_output_sha256)" ]] || fail 'release output drifted'
[[ "$(printf '%s' "$claim" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value claim_output_sha256)" ]] || fail 'claim output drifted'
[[ "$(printf '%s\n%s' "$release" "$claim" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value semantics_sha256)" ]] || fail 'two-stage semantics drifted'

for control in release_replacement_pid claim_replacement_pid \
  claim_replacement_invocation claim_replacement_start_tick \
  release_unbound_successor release_duplicate_launch claim_duplicate_launch \
  claim_count_drift claim_barrier_nonce release_future_completion \
  claim_completion_absent; do
  frame="$(manifest_value wire_schema) $(manifest_value "${control}_stage") $(manifest_value "${control}_word0") $(manifest_value "${control}_word1")"
  set +e
  observed="$(printf '%s\n' "$frame" | "$runtime")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == "$(manifest_value "${control}_decision")" ]] ||
    fail "$control refusal drifted"
done

release_mutant_module="$work/release-mutant.sio"
sed 's/if observation.material_pid_equal != 1 { return 593 }/if false { return 593 }/' \
  "$ROOT_DIR/$(manifest_value source_path)" > "$release_mutant_module"
release_mutant_runtime="$work/release-mutant-runtime"
SOUNIO_LOOM_CAUSAL_MID_EXEC_MODULE="$release_mutant_module" \
SOUNIO_LOOM_CAUSAL_MID_EXEC_OUTPUT="$release_mutant_runtime" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_mid_exec_fixture.sh" >/dev/null 2>&1 || true
[[ -x "$release_mutant_runtime" ]] || fail 'release PID mutant did not build'
release_mutant_frame="$(manifest_value wire_schema) $(manifest_value release_replacement_pid_stage) $(manifest_value release_replacement_pid_word0) $(manifest_value release_replacement_pid_word1)"
[[ "$(printf '%s\n' "$release_mutant_frame" | "$release_mutant_runtime")" == "$release" ]] ||
  fail 'release PID mutant did not admit replacement witness'

claim_mutant_module="$work/claim-mutant.sio"
sed 's/if observation.barrier_nonce_equal != 1 { return 600 }/if false { return 600 }/' \
  "$ROOT_DIR/$(manifest_value source_path)" > "$claim_mutant_module"
claim_mutant_runtime="$work/claim-mutant-runtime"
SOUNIO_LOOM_CAUSAL_MID_EXEC_MODULE="$claim_mutant_module" \
SOUNIO_LOOM_CAUSAL_MID_EXEC_OUTPUT="$claim_mutant_runtime" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_mid_exec_fixture.sh" >/dev/null 2>&1 || true
[[ -x "$claim_mutant_runtime" ]] || fail 'claim barrier-nonce mutant did not build'
claim_mutant_frame="$(manifest_value wire_schema) $(manifest_value claim_barrier_nonce_stage) $(manifest_value claim_barrier_nonce_word0) $(manifest_value claim_barrier_nonce_word1)"
[[ "$(printf '%s\n' "$claim_mutant_frame" | "$claim_mutant_runtime")" == "$claim" ]] ||
  fail 'claim barrier-nonce mutant did not admit stale barrier witness'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_causal_workflow_mid_exec_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'first result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'first result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value first_evidence_path)" ||
  fail 'first evidence is not the exact executable result'

freeze_result="sounio-loom-causal-workflow-mid-exec-freeze-selftest: PASS semantic_authority=Sounio action=9037 subordinate_contract=mid-exec-v1 stage=SEMANTICS_FROZEN semantics_sha256=$(manifest_value semantics_sha256) executable_sha256=$(manifest_value executable_sha256) release=ADMIT claim=CONTINUITY release_replacement_pid=DENY593 claim_replacement_pid=DENY593 replacement_invocation=DENY592 replacement_start_tick=DENY594 unbound_successor=DENY597 duplicate_launch=DENY599 count_drift=DENY599 barrier_nonce=DENY600 release_future_completion=DENY601 claim_completion_absent=DENY598 causal_sabotage=PASS exact_counts=compile+ticket+launch+result+attestation:1 material_execution=false pod_loss_measured=false production_activation=false parity_open=false claim_ready=false"
printf '%s\n' "$freeze_result" | cmp - "$ROOT_DIR/$(manifest_value freeze_evidence_path)" ||
  fail 'freeze evidence is not the exact frozen result'
printf '%s\n' "$freeze_result"
