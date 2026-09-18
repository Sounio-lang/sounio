#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-grant-controller.XXXXXX")"
CONTROLLER_ONE="$TEST_ROOT/controller-one"
CONTROLLER_TWO="$TEST_ROOT/controller-two"
FIXTURE="$TEST_ROOT/fixture"
BUNDLE="$TEST_ROOT/fixtures.v1"
RESIDENT="$TEST_ROOT/resident-v4"
RECEIPTS="$TEST_ROOT/resident.tsv"
AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
GENERATION=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-exec-grant-controller-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$AUTHORITY_MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "authority field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$AUTHORITY_MANIFEST")"
  printf '%s' "${line#*=}"
}

decision_code() {
  local decision token
  decision="$(manifest_value "$1")"
  for token in $decision; do
    if [[ "$token" == code=* ]]; then
      printf '%s' "${token#code=}"
      return 0
    fi
  done
  fail "authority decision $1 omitted code"
}

frame_for() {
  local label="$1" prefix="FIXTURE $1 " count line
  count="$(grep -c "^FIXTURE ${label} " "$BUNDLE" || true)"
  [[ "$count" == 1 ]] || fail "fixture $label occurs $count times"
  line="$(grep -m1 "^FIXTURE ${label} " "$BUNDLE")"
  printf '%s' "${line#"$prefix"}"
}

for output in "$CONTROLLER_ONE" "$CONTROLLER_TWO"; do
  SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_exec_grant_controller.sh" >/dev/null
done
cmp "$CONTROLLER_ONE" "$CONTROLLER_TWO" || fail 'two controller builds differ'

SOUNIO_LOOM_HOST_EXEC_QUORUM_FIXTURE_OUTPUT="$FIXTURE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_exec_quorum_fixture.sh" \
    >/dev/null
"$FIXTURE" > "$BUNDLE"
[[ "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)" == \
  523e132c4ab6a41ade56c2421472b092171627087fe4cf55ba4c74ac1f5d98fe ]] ||
  fail 'Sounio fixture bundle hash drifted'

SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_OUTPUT="$RESIDENT" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v4.sh" \
    >/dev/null

run_raw_controller() {
  local expected_parent="$1" generation="$2" input="$3" input_file status
  input_file="$(mktemp "$TEST_ROOT/controller-input.XXXXXX")"
  printf '%s\n' "$input" > "$input_file"
  set +e
  (
    set +e
    local actual_parent="$BASHPID"
    if [[ "$expected_parent" == actual ]]; then
      expected_parent="$actual_parent"
    fi
    SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_EXEC_GRANT_CONTROLLER=1 \
    SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_PARENT_PID="$expected_parent" \
    SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_GENERATION="$generation" \
    SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_DEADLINE_MS=15000 \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_RUNTIME="$RESIDENT" \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RECEIPTS" \
    "$CONTROLLER_ONE" < "$input_file"
    status=$?
    exit "$status"
  )
  status=$?
  set -e
  rm -f "$input_file"
  return "$status"
}

run_controller() {
  run_raw_controller actual "$GENERATION" "$1"
}

issue="$(frame_for issue)"
consume="$(frame_for consume)"
close="$(frame_for close)"
current="$(frame_for current)"
python="$(frame_for python)"
issue_code="$(decision_code issue_decision)"
consume_code="$(decision_code consume_decision)"
close_code="$(decision_code close_decision)"
current_code="$(decision_code current_decision)"
python_code="$(decision_code python_decision)"

happy="$(run_controller "$(printf 'ISSUE %s\nCONSUME %s\nCLOSE %s\n' \
  "$issue" "$consume" "$close")")"
[[ "$(wc -l <<< "$happy")" == 3 ]] || fail "happy path receipt count diverged: $happy"
issue_receipt="$(sed -n '1p' <<< "$happy")"
consume_receipt="$(sed -n '2p' <<< "$happy")"
close_receipt="$(sed -n '3p' <<< "$happy")"
[[ "$issue_receipt" == *"operation=ISSUE "*"code=$issue_code state=ISSUED "* &&
   "$issue_receipt" == *'resident_sequence=1 resident_poisoned=false quorum_ready=false controller_terminal=false'* ]] ||
  fail "issue receipt diverged: $issue_receipt"
[[ "$consume_receipt" == *"operation=CONSUME "*"code=$consume_code state=OUTCOME_PENDING "* &&
   "$consume_receipt" == *'resident_sequence=2 resident_poisoned=false quorum_ready=true controller_terminal=false'* ]] ||
  fail "consume receipt diverged: $consume_receipt"
[[ "$close_receipt" == *"operation=CLOSE "*"code=$close_code state=CLOSED "* &&
   "$close_receipt" == *'resident_sequence=3 resident_poisoned=false quorum_ready=false controller_terminal=true'* ]] ||
  fail "close receipt diverged: $close_receipt"
[[ "$happy" == *'single_resident_controller=true non_bearer_transport=pending material_grant=false material_execution=false barrier_release=false exec_attached=false parity_open=false claim_ready=false'* ]] ||
  fail 'controller promoted beyond evidence'

resident_pid="$(sed -n 's/.* resident_pid=\([0-9][0-9]*\) .*/\1/p' <<< "$issue_receipt")"
resident_generation="$(sed -n 's/.* resident_generation_sha256=\([0-9a-f]\{64\}\) .*/\1/p' <<< "$issue_receipt")"
[[ -n "$resident_pid" && -n "$resident_generation" ]] || fail 'issue receipt omitted resident identity'
for receipt in "$consume_receipt" "$close_receipt"; do
  [[ "$receipt" == *" resident_pid=$resident_pid "* &&
     "$receipt" == *" resident_generation_sha256=$resident_generation "* ]] ||
    fail 'controller changed resident within one grant generation'
done

treatment="$(run_controller "$(printf 'ISSUE %s\n' "$current")")"
[[ "$(wc -l <<< "$treatment")" == 1 &&
   "$treatment" == *"operation=ISSUE "*"code=$current_code state=VACANT "* &&
   "$treatment" == *'quorum_ready=false controller_terminal=true'* ]] ||
  fail "current treatment did not terminate without quorum: $treatment"

python_output="$(run_controller "$(printf 'ISSUE %s\n' "$python")")"
[[ "$python_output" == *"operation=ISSUE "*"code=$python_code state=VACANT "* &&
   "$python_output" == *'quorum_ready=false controller_terminal=true'* ]] ||
  fail "Python authority fixture did not terminate without quorum: $python_output"

refusal() {
  local label="$1" input="$2" expected="$3" output status
  set +e
  output="$(run_controller "$input" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 70 && "$output" == *"$expected"* &&
     "$output" != *'quorum_ready=true'* ]] ||
    fail "$label did not fail closed: status=$status output=$output"
}

refusal consume-before-issue "$(printf 'CONSUME %s\n' "$consume")" \
  'controller-protocol-state-mismatch'
refusal replay-issue "$(printf 'ISSUE %s\nISSUE %s\n' "$issue" "$issue")" \
  'controller-protocol-state-mismatch'
refusal multiline-command $'UNKNOWN\n' 'controller-protocol-state-mismatch'

set +e
wrong_parent="$(run_raw_controller 1 "$GENERATION" $'STOP\n' 2>&1)"
wrong_parent_status=$?
set -e
[[ $wrong_parent_status -eq 70 && "$wrong_parent" == *'controller-parent-mismatch'* ]] ||
  fail "wrong parent was admitted: $wrong_parent"

set +e
bad_generation="$(run_raw_controller actual 0 $'STOP\n' 2>&1)"
bad_generation_status=$?
set -e
[[ $bad_generation_status -eq 70 && "$bad_generation" == *'controller-generation-invalid'* ]] ||
  fail "invalid generation was admitted: $bad_generation"

set +e
public_output="$($CONTROLLER_ONE 2>&1)"
public_status=$?
set -e
[[ $public_status -eq 70 && "$public_output" == *'missing-controller-environment:SOUNIO_LOOM_EXEC_GRANT_CONTROLLER'* ]] ||
  fail 'controller exposed an unmarked public mode'

if rg -n 'DENY49[1-9]|DENY500|DENY501|ALLOW code=0 reason=allow|code=491|code=499' \
  "$ROOT_DIR/tools/loom/src/loom_exec_grant_controller.ml" >/dev/null; then
  fail 'controller copied a Sounio expected-result string'
fi
dependencies="$(ldd "$CONTROLLER_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'controller has a prohibited runtime dependency'
fi

printf 'sounio-loom-exec-grant-controller-selftest: PASS semantic_authority=Sounio operational_kernel=OCaml role=EFFECT_PARITY source_fixtures=Sounio fixture_bundle_sha256=523e132c4ab6a41ade56c2421472b092171627087fe4cf55ba4c74ac1f5d98fe lifecycle=ISSUE-CONSUME-CLOSE treatment=semantic-deny python=semantic-deny resident_identity=stable resident_sequence=1,2,3 controller_generation=bound transaction_digest=bound consume_quorum_ready=true issue_quorum_ready=false close_quorum_ready=false replay=refused out_of_order=refused wrong_parent=refused invalid_generation=refused public_mode=absent expected_results_encoded=false runtime_dependencies=clean controller_source_sha256=%s controller_binary_sha256=%s single_resident_controller=true non_bearer_transport=pending material_grant=false material_execution=false barrier_release=false exec_attached=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_grant_controller.ml" | cut -d ' ' -f 1)" \
  "$(sha256sum "$CONTROLLER_ONE" | cut -d ' ' -f 1)"
