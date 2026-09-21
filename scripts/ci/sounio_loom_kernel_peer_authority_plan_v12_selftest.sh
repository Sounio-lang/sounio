#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-kernel-peer-plan-v12-test.XXXXXX")"
PLAN_A="$TEST_ROOT/kernel-peer-plan-a"
PLAN_B="$TEST_ROOT/kernel-peer-plan-b"
ACTION_9025="$TEST_ROOT/action-9025-authority"
SOURCE="$ROOT_DIR/tools/loom/kernel_peer_authority_plan_v12_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-peer-authority-plan-v12-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

file_hash() {
  sha256sum "$1" | cut -d ' ' -f 1
}

SOUNIO_LOOM_KERNEL_PEER_PLAN_V12_OUTPUT="$PLAN_A" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_authority_plan_v12.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_PLAN_V12_OUTPUT="$PLAN_B" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_authority_plan_v12.sh" >/dev/null
[[ "$(file_hash "$PLAN_A")" == "$(file_hash "$PLAN_B")" ]] ||
  fail 'Sounio V12 rebuild is nondeterministic'

bundle="$TEST_ROOT/bundle"
(ulimit -f 2048; timeout 10s "$PLAN_A" >"$bundle") ||
  fail 'V12 semantic bundle timed out or crossed its output bound'
[[ "$(wc -c <"$bundle")" -le 65536 ]] || fail 'V12 semantic bundle is oversized'
[[ "$(file_hash "$bundle")" == 94d22ea974168f41200684c60da5e673a4afebb7e34f0b4cd8f228d0303e7b97 ]] ||
  fail 'V12 semantic bundle hash drifted'

grep -Fxq 'PEER_TRUTH same_kuid_required=true real_uid_equal=true effective_uid_equal=true saved_uid_equal=true filesystem_uid_equal=true attacker_syscalls_open=true receiver_side_required=true distinct_kuid_is_not_same_uid_proof=true caller_seccomp_is_not_receiver_proof=true pid_secrecy_is_not_isolation=true' "$bundle" ||
  fail 'V12 peer-truth rule drifted'
grep -Fxq 'HASH_BINDING fields=invariant_sha256+delta_sha256+attempt_sha256+target_sha256+extinction_sha256 causal_pair=TREATMENT+MEDIATOR_REMOVED only_delta=mediator_presence+policy_hash' "$bundle" ||
  fail 'V12 causal binding drifted'
grep -Fxq 'OBSERVATION_TYPES values=REFUSED_BEFORE_EFFECT+CROSSED_NAMED_RULE+EFFECT_COMPLETED+EXPERIMENT_UNAVAILABLE crossed_counts_as_coverage=false unavailable_counts_as_isolation=false syscall_error_alone_is_refusal=false syscall_success_alone_is_completion=false' "$bundle" ||
  fail 'V12 observation semantics drifted'
grep -Fxq 'EXPECTED_TOTALS refused=26 completed=14 unavailable=10 crossed=0 treatment_refused=10 mediator_removed_completed=10 distinct_refused=10 caller_seccomp_unavailable=10 dumpable_completed=4 dumpable_refused=6' "$bundle" ||
  fail 'V12 expected totals drifted'
grep -Fxq 'MATERIAL_ACCEPTANCE same_kuid_pair_observed=true attacker_syscalls_open=true receiver_mediator_active=true all_operations_refused_before_effect=true mediator_removed_completes_named_effects=true distinct_kuid_not_counted_as_same_uid_proof=true caller_seccomp_not_counted_as_receiver_proof=true dumpable_only_not_counted_as_complete=true all_epoch_objects_extinct=true' "$bundle" ||
  fail 'V12 material acceptance drifted'

awk '
  function value(key, i, a) {
    for (i = 1; i <= NF; i++) {
      split($i, a, "=")
      if (a[1] == key) return a[2]
    }
    return ""
  }
  /^OPERATION / {
    operation_id = value("index")
    if (operation_id !~ /^[0-9]+$/ || operation_id < 1 || operation_id > 10 || operation[operation_id]++) exit 10
    if (value("completion_witness") == "" || value("caller_surface") != "REQUIRED_OPEN") exit 11
    operations++
  }
  /^VERTEX / {
    vertex = value("vertex")
    operation_index = value("operation")
    expected = value("expected")
    rule = value("rule")
    role = value("proof_role")
    key = vertex "/" operation_index
    if (seen[key]++) exit 12
    if (operation_index !~ /^[0-9]+$/ || operation_index < 1 || operation_index > 10) exit 13
    if (value("invariant_sha256") != "REQUIRED" || value("delta_sha256") != "REQUIRED" || value("attempt_sha256") != "REQUIRED" || value("target_sha256") != "REQUIRED" || value("extinction_sha256") != "REQUIRED") exit 14
    if (vertex == "TREATMENT") {
      if (expected != "REFUSED_BEFORE_EFFECT" || rule != "RECEIVER_KERNEL_MEDIATOR" || role != "DECISIVE") exit 15
      treatment++
    } else if (vertex == "MEDIATOR_REMOVED") {
      if (expected != "EFFECT_COMPLETED" || rule != "NONE" || role != "DECISIVE") exit 16
      open++
    } else if (vertex == "DISTINCT_KUID_CONTROL") {
      if (expected != "REFUSED_BEFORE_EFFECT" || rule != "KERNEL_CREDENTIAL_CHECK" || role != "CONTROL_ONLY") exit 17
      distinct++
    } else if (vertex == "CALLER_SECCOMP_CONTROL") {
      if (expected != "EXPERIMENT_UNAVAILABLE" || rule != "CALLER_ATTACK_SYSCALL_FILTERED" || role != "INVALID_PROOF") exit 18
      caller++
    } else if (vertex == "DUMPABLE_ONLY_CONTROL") {
      if (rule != "DUMPABLE_ONLY" || role != "CONTROL_ONLY") exit 19
      if (expected == "EFFECT_COMPLETED") dumpable_completed++
      else if (expected == "REFUSED_BEFORE_EFFECT") dumpable_refused++
      else exit 20
    } else {
      exit 21
    }
    if (expected == "REFUSED_BEFORE_EFFECT") refused++
    else if (expected == "EFFECT_COMPLETED") completed++
    else if (expected == "EXPERIMENT_UNAVAILABLE") unavailable++
    else exit 22
    vertices++
  }
  END {
    if (operations != 10 || vertices != 50 || treatment != 10 || open != 10 || distinct != 10 || caller != 10 || dumpable_completed != 4 || dumpable_refused != 6 || refused != 26 || completed != 14 || unavailable != 10) exit 23
  }
' "$bundle" || fail 'typed V12 operation matrix diverged'

[[ "$(grep -c '^RECEIVER_PROPERTY ' "$bundle")" == 7 ]] ||
  fail 'receiver property count drifted'
[[ "$(grep -c '^PRINCIPAL_VERTEX ' "$bundle")" == 5 ]] ||
  fail 'principal vertex count drifted'
[[ "$(grep -c '^ACTION_CASE ' "$bundle")" == 3 ]] ||
  fail 'action case count drifted'

SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$ACTION_9025" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null
one='1 1 1 1 1 1 1 1'
bindings="$one $one $one $one $one $one $one $one $one $one $one"
coverage_modes='3 3 3 2 2 2 2 2 2 2 2 2'

while IFS= read -r line; do
  label="$(printf '%s\n' "$line" | sed -n 's/^ACTION_CASE label=\([^ ]*\).*/\1/p')"
  peer="$(printf '%s\n' "$line" | sed -n 's/.* same_uid_peer_isolation=\([^ ]*\).*/\1/p')"
  coverage="$(printf '%s\n' "$line" | sed -n 's/.* coverage_family_count=\([^ ]*\).*/\1/p')"
  expected_code="$(printf '%s\n' "$line" | sed -n 's/.* expected_code=\([^ ]*\).*/\1/p')"
  expected_reason="$(printf '%s\n' "$line" | sed -n 's/.* expected_reason=\([^ ]*\).*/\1/p')"
  [[ -n "$label" && -n "$peer" && -n "$coverage" && -n "$expected_code" && -n "$expected_reason" ]] ||
    fail "malformed Sounio action case: $line"
  frame="9025 3 1 1 1 1 1 1 $peer 1 $coverage 12 1 $coverage_modes $bindings"
  actual="$(printf '%s\n' "$frame" | "$ACTION_9025" || true)"
  actual_code="$(printf '%s\n' "$actual" | sed -n 's/.* code=\([^ ]*\).*/\1/p')"
  actual_reason="$(printf '%s\n' "$actual" | sed -n 's/.* reason=\([^ ]*\).*/\1/p')"
  [[ "$actual_code" == "$expected_code" && "$actual_reason" == "$expected_reason" ]] ||
    fail "$label disagreed with action 9025: $actual"
done < <(grep '^ACTION_CASE ' "$bundle")

boundary='BOUNDARY garden_v12=true sounio_executable_v12=false semantics_frozen_v12=false backend_discovery=false material_peer_matrix=false same_uid_peer_isolation=false action_9025_decision=DENY451 material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false'
grep -Fxq "$boundary" "$bundle" || fail 'V12 boundary drifted'

PREMATERIAL_COMMIT="${SOUNIO_LOOM_KERNEL_PEER_PLAN_V12_PREMATERIAL_COMMIT:-}"
if [[ -n "$PREMATERIAL_COMMIT" ]]; then
  git -C "$ROOT_DIR" cat-file -e "${PREMATERIAL_COMMIT}^{commit}" 2>/dev/null ||
    fail "prematerial commit is absent: $PREMATERIAL_COMMIT"
  if git -C "$ROOT_DIR" cat-file -e \
    "$PREMATERIAL_COMMIT:tools/loom/src/loom_kernel_peer_authority_v12.cpp" 2>/dev/null; then
    fail 'V12 material bytes existed in the frozen prematerial commit'
  fi
else
  [[ ! -e "$ROOT_DIR/tools/loom/src/loom_kernel_peer_authority_v12.cpp" ]] ||
    fail 'V12 material bytes appeared before semantic freeze'
fi

printf 'sounio-loom-kernel-peer-authority-plan-v12-selftest: PASS producer=Sounio role=SEMANTIC_POLICY_PLAN semantic_authority=Sounio action=9025 principal_vertices=5 operations=10 observations=50 decisive_pairs=10 receiver_properties=7 refused=26 completed=14 unavailable=10 treatment_refused=10 mediator_removed_completed=10 distinct_kuid_control=10 caller_seccomp_invalid_proof=10 dumpable_partial=4+6 complete_hypothesis=ALLOW current_v11=DENY451 coverage_missing=DENY447 deterministic=true output_bounded=true shell_expected_results=false python_executed=false rust_executed=false source_sha256=%s executable_sha256=%s bundle_sha256=%s garden_v12=true sounio_executable_v12=false semantics_frozen_v12=false backend_discovery=false native_v12_bytes_created=false material_peer_matrix=false same_uid_peer_isolation=false material_coverage=false complete_effects=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$SOURCE")" "$(file_hash "$PLAN_A")" "$(file_hash "$bundle")"
