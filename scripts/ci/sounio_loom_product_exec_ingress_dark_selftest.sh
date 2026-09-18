#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/product-exec-ingress-dark.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
AUTHORITY_RUNTIME="$TEST_ROOT/sounio-loom-language-authority-runtime"
RESIDENT_RUNTIME="$TEST_ROOT/sounio-loom-resident-membrane-runtime-v5"
AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"
PROJECTION="$ROOT_DIR/tools/loom/kernel_peer_activation_capsule.current.v1"
COORD_DIR="$TEST_ROOT/coord"
EXECUTION_LOG="$TEST_ROOT/execution-authority.tsv"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-product-exec-ingress-dark-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

projection_code() {
  local label="$1" line rest
  line="$(grep -m1 "^CASE label=${label} EXPECT code=" "$PROJECTION")"
  [[ -n "$line" ]] || fail "Sounio projection omitted $label"
  rest="${line#* EXPECT code=}"
  printf '%s' "${rest%% *}"
}

probe() {
  local tag="$1" mode="$2" event="$3"
  SOUNIO_COORD_DIR="$COORD_DIR" \
  SOUNIO_COORD_RUNTIME_MODE=local \
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_COORD_NATIVE_HOOK_SELFTEST=1 \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME="$AUTHORITY_RUNTIME" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_LOG="$TEST_ROOT/$tag.hook.tsv" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_RUNTIME="$RESIDENT_RUNTIME" \
  SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$TEST_ROOT/$tag.resident.tsv" \
  SOUNIO_LOOM_EXEC_INGRESS_DARK_LOG="$TEST_ROOT/$tag.ingress.tsv" \
  SOUNIO_LOOM_EXECUTION_AUTHORITY_LOG="$EXECUTION_LOG" \
    "$LOOM" exec-ingress-probe --root "$ROOT_DIR" --mode "$mode" \
      --event "$event"
}

expect_probe() {
  local tag="$1" mode="$2" event="$3" marker="$4"
  local output rc
  set +e
  output="$(probe "$tag" "$mode" "$event" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -eq 0 && "$output" == *"$marker"* ]] ||
    fail "$tag failed: rc=$rc output=$output"
  printf '%s' "$output"
}

frozen_executable_commit="$(sed -n 's/^sounio_executable_commit=//p' "$AUTHORITY_MANIFEST")"
[[ -n "$frozen_executable_commit" ]] ||
  fail 'language-authority manifest omitted its executable commit'
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$frozen_executable_commit" \
  bin/souc bin/souc-lean-single-x86_64 | tar -x -C "$TOOLCHAIN_ROOT"
SOUNIO_LOOM_LANGUAGE_AUTHORITY_SOUC="$TOOLCHAIN_ROOT/bin/souc" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$AUTHORITY_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CURRENT_OUTPUT="$PROJECTION" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_activation_capsule_current_frame.sh" \
  >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$RESIDENT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v5.sh" >/dev/null
(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"

sentinel="$TEST_ROOT/command-executed"
python_sentinel="$TEST_ROOT/python-executed"
rust_sentinel="$TEST_ROOT/rust-executed"
event="$TEST_ROOT/exec.json"
copied_event="$TEST_ROOT/copied.json"
python_event="$TEST_ROOT/python.json"
rust_event="$TEST_ROOT/rust.json"

printf '%s\n' \
  "{\"hook_event_name\":\"PreToolUse\",\"session_id\":\"exec-ingress-dark\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"/usr/bin/touch $sentinel\"}}" \
  >"$event"
cp "$event" "$copied_event"
printf '%s\n' \
  "{\"hook_event_name\":\"PreToolUse\",\"session_id\":\"exec-ingress-python\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"python3 -c 'open(\\\"$python_sentinel\\\",\\\"w\\\").write(\\\"BAD\\\")'\"}}" \
  >"$python_event"
printf '%s\n' \
  "{\"hook_event_name\":\"PreToolUse\",\"session_id\":\"exec-ingress-rust\",\"cwd\":\"$ROOT_DIR\",\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"rustc --version > $rust_sentinel\"}}" \
  >"$rust_event"

current_code="$(projection_code current_material)"
seal_code="$(projection_code seal)"

inherited_output="$(expect_probe inherited inherited "$event" \
  'mode=inherited hook_code=0 broker_code=0')"
[[ "$inherited_output" != *'sounio execution capability'* ]] ||
  fail 'inherited dark probe reached execution capability issuance'
grep -Fq $'decision=DENY\treason=descriptor-bound-action-9031\tauthorizing=false' \
  "$TEST_ROOT/inherited.ingress.tsv" ||
  fail 'descriptor-bound current projection was not a dark DENY'
grep -Fq $'descriptor_present=true\tdescriptor_bound=true\tdescriptor_transport=unix-stream-inherited\tdescriptor_is_bearer=false' \
  "$TEST_ROOT/inherited.ingress.tsv" ||
  fail 'inherited descriptor facts were not recorded'
grep -Fq $'peer_distinct_uid=false\t' "$TEST_ROOT/inherited.ingress.tsv" ||
  fail 'same-UID fixture exception was not explicit'
grep -Fq $'activation_code='"$current_code"$'\t' \
  "$TEST_ROOT/inherited.ingress.tsv" ||
  fail 'current Sounio projection code was not recorded'
grep -Fq $'decision_authority=Sounio\tsounio_evaluated=true\tsemantic_authority=Sounio\tproducing_language=Sounio\tlanguage_role=SEMANTIC_AUTHORITY\t' \
  "$TEST_ROOT/inherited.ingress.tsv" ||
  fail 'executed Sounio decision lost its authority label'

forged_output="$(expect_probe forged forged "$event" \
  'mode=forged hook_code=2 broker_code=90')"
[[ "$forged_output" == *'product-exec-ingress-peer-not-distinct'* ]] ||
  fail "same-UID self-broker refusal was unattributed: $forged_output"
grep -Fq $'decision=DENY\treason=peer-not-distinct\t' \
  "$TEST_ROOT/forged.ingress.tsv" ||
  fail 'same-UID self-broker refusal omitted its decision receipt'
grep -Fq $'descriptor_present=true\tdescriptor_bound=false\t' \
  "$TEST_ROOT/forged.ingress.tsv" ||
  fail 'same-UID self-broker was recorded as descriptor-bound'

fixture_escape_output="$(expect_probe fixture-escape fixture-escape "$event" \
  'mode=fixture-escape hook_code=2 broker_code=90')"
[[ "$fixture_escape_output" == \
   *'product-exec-ingress-same-uid-fixture-requires-probe-only'* ]] ||
  fail "same-UID fixture flag escaped probe-only custody: $fixture_escape_output"

missing_output="$(expect_probe missing missing "$event" \
  'mode=missing hook_code=2 broker_code=-1')"
[[ "$missing_output" == *'product-exec-ingress-descriptor-absent'* ]] ||
  fail "missing descriptor refusal was unattributed: $missing_output"
grep -Fq $'decision=DENY\treason=descriptor-absent\t' \
  "$TEST_ROOT/missing.ingress.tsv" ||
  fail 'missing descriptor refusal omitted its decision receipt'
grep -Fq $'decision_authority=OCaml-structural-precondition\tsounio_evaluated=false\tsemantic_authority=Sounio\tproducing_language=OCaml\tlanguage_role=OPERATIONAL_ATTACHMENT\t' \
  "$TEST_ROOT/missing.ingress.tsv" ||
  fail 'structural refusal was laundered into a Sounio decision'

copied_output="$(expect_probe copied-json missing "$copied_event" \
  'mode=missing hook_code=2 broker_code=-1')"
[[ "$copied_output" == *'product-exec-ingress-descriptor-absent'* ]] ||
  fail 'copied hook JSON was not refused by the ingress rule'

python_output="$(expect_probe python-oracle missing "$python_event" \
  'mode=missing hook_code=2 broker_code=-1')"
rust_output="$(expect_probe rust-oracle missing "$rust_event" \
  'mode=missing hook_code=2 broker_code=-1')"
[[ "$python_output" == *'product-exec-ingress-descriptor-absent'* && \
   "$rust_output" == *'product-exec-ingress-descriptor-absent'* && \
   ! -e "$python_sentinel" && ! -e "$rust_sentinel" ]] ||
  fail 'Python or Rust oracle attempt crossed the required ingress'

set +e
sabotage_output="$(SOUNIO_LOOM_ACTIVATION_DARK_LABEL=seal \
  probe sabotage inherited "$event" 2>&1)"
sabotage_rc=$?
set -e
[[ "$sabotage_rc" -eq 0 && \
   "$sabotage_output" == *'mode=inherited hook_code=2 broker_code=0'* && \
   "$sabotage_output" == *'product-exec-ingress-dark-unexpected-allow'* ]] ||
  fail "Sounio ALLOW sabotage did not stop at the product ingress: $sabotage_output"
grep -Fq $'decision=ALLOW\treason=descriptor-bound-action-9031\t' \
  "$TEST_ROOT/sabotage.ingress.tsv" ||
  fail 'causal sabotage did not record the Sounio ALLOW'
grep -Fq $'activation_code='"$seal_code"$'\t' \
  "$TEST_ROOT/sabotage.ingress.tsv" ||
  fail 'causal sabotage recorded the wrong Sounio projection code'

[[ ! -e "$sentinel" && ! -e "$EXECUTION_LOG" ]] ||
  fail 'a command or legacy execution-authority path ran during dark attachment'

observe_line="$(grep -n -m1 'Loom_exec_ingress.observe' \
  "$ROOT_DIR/tools/loom/src/loom_hook.ml" | cut -d: -f1)"
issue_line="$(grep -n -m1 'Loom_exec.authorize_and_issue' \
  "$ROOT_DIR/tools/loom/src/loom_hook.ml" | cut -d: -f1)"
[[ "$observe_line" =~ ^[0-9]+$ && "$issue_line" =~ ^[0-9]+$ && \
   "$observe_line" -lt "$issue_line" ]] ||
  fail 'ExecIngress no longer precedes grant issuance in the native hook'

if rg -n 'DENY50[2-9]|DENY510|ALLOW code=0 reason=allow' \
  "$ROOT_DIR/tools/loom/src/loom_exec_ingress.ml" \
  "$ROOT_DIR/tools/loom/src/loom_hook.ml" \
  "$ROOT_DIR/tools/loom/src/loom_membrane.ml" >/dev/null; then
  fail 'OCaml copied a Sounio semantic expected-result string'
fi

printf '%s\n' \
  "sounio-loom-product-exec-ingress-dark-selftest: PASS semantic_authority=Sounio action=9031 operational_attachment=OCaml product_path=native-agent-hook hook_order=ExecIngress-before-ExecGrant descriptor_transport=unix-stream-inherited descriptor_is_bearer=false one_shot_challenge=event+command-hash current_material=DENY${current_code}+PRODUCT_CONTINUES causal_sabotage=ALLOW${seal_code}+HOOK_REFUSAL+NO_GRANT same_uid_self_broker=refused same_uid_fixture_escape=refused copied_json_without_descriptor=refused_when-required python_oracle=refused-before-execution rust_oracle=refused-before-execution command_executed=false python_executed=false rust_executed=false descriptor_dark_attached=true distinct_uid_product_broker=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false"
