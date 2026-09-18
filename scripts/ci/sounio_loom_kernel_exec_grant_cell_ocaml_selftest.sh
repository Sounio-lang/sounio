#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/exec-grant-cell-ocaml.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
RUNTIME="$TEST_ROOT/resident-v4"
RECEIPTS="$TEST_ROOT/resident.tsv"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-exec-grant-cell-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v4.sh" >/dev/null
(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"

parent_9029='1636926980 3205986131 3323207532 3505413428 706242987 2411760920 1929815169 3727939342'
parent_9021='3497534264 556131944 3943529214 1565657389 3821375173 3204015455 2733765994 2625951936'
parent_9022='4125506095 3601417934 2711931735 20635855 2708941890 3284947684 758124027 2068177262'
one='1 1 1 1 1 1 1 1'
bindings="$parent_9029 $parent_9021 $parent_9022 $one $one $one $one $one $one"
issue_transition='1 0 1 2 3 4 5 6 7 100 50'
consume_transition='2 1 3 2 3 4 5 6 8 100 49'
close_transition='3 3 4 2 3 4 5 6 9 100 48'
revoke_transition='4 1 5 2 3 4 5 6 8 100 49'
issue_parents='1 1 0 0 0 1 0 0'
consume_parents='1 1 1 0 0 1 1 0'
close_parents='1 1 1 1 0 1 1 1'
revoke_parents='1 1 0 0 1 1 0 0'
identity='1 1 1 1 1 1 1 1'
peer='1 1 1 1 1 1 1 1 1'
wrong_peer='1 1 1 1 1 1 0 0 1'
shape='1 1 1 1 1 1 1 1'
consumption='1 1 1 1 1 1 1'
revocation='1 1 1 1 1 1 1'
live_extinction='0 0 0 0 1'
terminal_extinction='1 1 1 1 1'
live_outcome='0 0 0 0 0 0 0 0'
close_outcome='1 1 1 1 1 1 0 0'
revoke_outcome='0 0 0 0 0 1 1 1'
authority='1 1 1 1 1 1 1'
python_authority='0 0 0 1 1 1 1'
evidence='1 1 11 11'

valid_issue="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
valid_consume="9030 3 1 $consume_transition $consume_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
valid_close="9030 3 1 $close_transition $close_parents $identity $peer $shape $consumption $revocation $terminal_extinction $close_outcome $authority $evidence $bindings"
valid_revoke="9030 3 1 $revoke_transition $revoke_parents $identity $peer $shape $consumption $revocation $terminal_extinction $revoke_outcome $authority $evidence $bindings"
deny_consume="9030 3 1 $consume_transition $consume_parents $identity $wrong_peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
current_material="9030 3 1 $issue_transition 1 0 0 0 0 0 0 0 $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $authority $evidence $bindings"
python_oracle="9030 3 1 $issue_transition $issue_parents $identity $peer $shape $consumption $revocation $live_extinction $live_outcome $python_authority $evidence $bindings"

printf '%s\n' "$valid_issue" > "$TEST_ROOT/issue.frame"
printf '%s\n' "$valid_consume" > "$TEST_ROOT/consume.frame"
printf '%s\n' "$valid_close" > "$TEST_ROOT/close.frame"
printf '%s\n' "$valid_revoke" > "$TEST_ROOT/revoke.frame"
printf '%s\n' "$deny_consume" > "$TEST_ROOT/deny.frame"
printf '%s\n' "$current_material" > "$TEST_ROOT/current.frame"
printf '%s\n' "$python_oracle" > "$TEST_ROOT/python.frame"

probe() {
  local mode="$1" issue="$2"
  shift 2
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_RUNTIME="$RUNTIME" \
    SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RECEIPTS" \
    "$LOOM" exec-grant-cell-probe --root "$ROOT_DIR" --mode "$mode" \
      --issue "$issue" --deadline-ms 15000 "$@"
}

current_output="$(probe current "$TEST_ROOT/current.frame")"
[[ "$current_output" == *'codes=491 state=VACANT poisoned=false'* && \
  "$current_output" == *'material_grant=false'* ]] ||
  fail "current material was not refused without state advance: $current_output"

python_output="$(probe python "$TEST_ROOT/python.frame")"
[[ "$python_output" == *'codes=499 state=VACANT poisoned=false'* ]] ||
  fail "Python oracle data was not refused by Sounio: $python_output"

happy_output="$(probe happy "$TEST_ROOT/issue.frame" \
  --consume "$TEST_ROOT/consume.frame" --close "$TEST_ROOT/close.frame")"
[[ "$happy_output" == *'codes=0,0,0 state=CLOSED poisoned=false'* && \
  "$happy_output" == *'sequence=3 '* ]] ||
  fail "happy lifecycle did not close exactly once: $happy_output"

deny_output="$(probe deny-preserves "$TEST_ROOT/issue.frame" \
  --deny "$TEST_ROOT/deny.frame" --consume "$TEST_ROOT/consume.frame" \
  --close "$TEST_ROOT/close.frame")"
[[ "$deny_output" == *'codes=0,493,0,0 state=CLOSED poisoned=false'* && \
  "$deny_output" == *'deny_preserved=true'* ]] ||
  fail "semantic denial burned or mutated the grant: $deny_output"

revoke_output="$(probe revoke "$TEST_ROOT/issue.frame" \
  --revoke "$TEST_ROOT/revoke.frame")"
[[ "$revoke_output" == *'codes=0,0 state=REVOKED poisoned=false'* ]] ||
  fail "typed revocation did not retire the grant: $revoke_output"

for mode in replay mismatch timeout eof; do
  output="$(probe "$mode" "$TEST_ROOT/issue.frame" \
    --consume "$TEST_ROOT/consume.frame")"
  [[ "$output" == *'codes=0 state=POISONED poisoned=true control_refused=true reuse_refused=true'* ]] ||
    fail "$mode did not fail closed and refuse reuse: $output"
done

grep -Fq $'\tevent=EXEC_GRANT_CELL\t' "$RECEIPTS" ||
  fail 'exec-grant-cell receipt is missing'
grep -Fq $'\tparent_9030_manifest_sha256=8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051\t' \
  "$RECEIPTS" || fail 'receipt omitted frozen action 9030'
grep -Fq $'\tresident_manifest_sha256=f61c93a3aefdbab792ed757faddf778017d34e0fa6bed97c565b56fe3147d473\t' \
  "$RECEIPTS" || fail 'receipt omitted frozen resident v4 manifest'

tampered_manifest="$TEST_ROOT/kernel-exec-grant-cell.freeze.v1"
cp "$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1" \
  "$tampered_manifest"
printf '\n' >> "$tampered_manifest"
set +e
manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_RUNTIME="$RUNTIME" \
  SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_MANIFEST="$tampered_manifest" \
  "$LOOM" exec-grant-cell-probe --root "$ROOT_DIR" --mode current \
    --issue "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
manifest_rc=$?
set -e
[[ "$manifest_rc" -eq 1 && \
  "$manifest_output" == *'exec-grant-cell-manifest-hash-mismatch'* ]] ||
  fail "action 9030 manifest tamper did not fail before spawn: $manifest_output"

tampered_resident_manifest="$TEST_ROOT/resident-v4.runtime"
cp "$ROOT_DIR/tools/loom/resident_membrane.runtime.v4" \
  "$tampered_resident_manifest"
printf '\n' >> "$tampered_resident_manifest"
set +e
resident_manifest_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_RUNTIME="$RUNTIME" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_MANIFEST="$tampered_resident_manifest" \
  "$LOOM" exec-grant-cell-probe --root "$ROOT_DIR" --mode current \
    --issue "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
resident_manifest_rc=$?
set -e
[[ "$resident_manifest_rc" -eq 1 && \
  "$resident_manifest_output" == *'resident-runtime-v4-manifest-hash-mismatch'* ]] ||
  fail "resident v4 manifest tamper did not fail before spawn: $resident_manifest_output"

tampered_runtime="$TEST_ROOT/resident-v4-tampered"
cp "$RUNTIME" "$tampered_runtime"
printf '\n' >> "$tampered_runtime"
chmod 0755 "$tampered_runtime"
set +e
runtime_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_RUNTIME="$tampered_runtime" \
  "$LOOM" exec-grant-cell-probe --root "$ROOT_DIR" --mode current \
    --issue "$TEST_ROOT/current.frame" --deadline-ms 2000 2>&1)"
runtime_rc=$?
set -e
[[ "$runtime_rc" -eq 1 && \
  "$runtime_output" == *'resident-runtime-hash-mismatch'* ]] ||
  fail "resident v4 runtime tamper did not fail before spawn: $runtime_output"

if rg -n 'DENY49[1-9]|DENY500|DENY501|ALLOW code=0 reason=allow' \
  "$ROOT_DIR/tools/loom/src/loom_exec_grant_cell.ml" \
  "$ROOT_DIR/tools/loom/src/loom_resident.ml" \
  "$ROOT_DIR/tools/loom/src/loom.ml" >/dev/null; then
  fail 'OCaml copied a semantic expected-result string'
fi

printf '%s\n' \
  'sounio-loom-kernel-exec-grant-cell-ocaml-selftest: PASS semantic_authority=Sounio operational_kernel=OCaml resident=Sounio-v4 lifecycle=VACANT-ISSUED-OUTCOME_PENDING-CLOSED-REVOKED-POISONED happy=ALLOWx3 revoke=ALLOWx2 current_material=DENY491 python_oracle=DENY499 semantic_deny=DENY493+STATE_PRESERVED+RECOVERY_ALLOWED replay=POISON+REUSE_REFUSED mismatch=POISON+REUSE_REFUSED timeout=POISON+REUSE_REFUSED eof=POISON+REUSE_REFUSED receipts=hash-bound action_manifest_tamper=refused-before-spawn resident_manifest_tamper=refused-before-spawn runtime_tamper=refused-before-spawn ocaml_expected_results=absent material_grant=false material_coverage=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false'
