#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-transport-v5.XXXXXX")"
RUNTIME="$TEST_ROOT/resident-membrane-v5"
MEMBRANE_REFERENCE="$TEST_ROOT/subprocess-membrane"
RESIDENT_REFERENCE="$TEST_ROOT/resident-authority"
CLOSURE_REFERENCE="$TEST_ROOT/effect-closure-authority"
INVOCATION_REFERENCE="$TEST_ROOT/kernel-invocation-cell-authority"
GRANT_REFERENCE="$TEST_ROOT/kernel-exec-grant-cell-authority"
ACTIVATION_REFERENCE="$TEST_ROOT/kernel-peer-activation-capsule-authority"

cleanup() {
  if [[ -n "${resident_pid:-}" ]]; then
    kill "$resident_pid" 2>/dev/null || true
    wait "$resident_pid" 2>/dev/null || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-transport-v5-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v5.sh" >/dev/null
SOUNIO_LOOM_SUBPROCESS_MEMBRANE_OUTPUT="$MEMBRANE_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_subprocess_membrane.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_AUTHORITY_OUTPUT="$RESIDENT_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_authority.sh" >/dev/null
SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$CLOSURE_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$INVOCATION_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_invocation_cell_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$GRANT_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_exec_grant_cell_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CAPSULE_OUTPUT="$ACTIVATION_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_activation_capsule_authority.sh" >/dev/null

activation_fixtures="$(printf '1\n' | "$ACTIVATION_REFERENCE")"
activation_frame() {
  local label="$1"
  printf '%s\n' "$activation_fixtures" |
    sed -n "s/^CASE label=${label} EXPECT code=[0-9][0-9]* FRAME //p"
}

seal_activation="$(activation_frame seal)"
consume_activation="$(activation_frame consume)"
extinguish_activation="$(activation_frame extinguish)"
poison_activation="$(activation_frame poison)"
current_activation="$(activation_frame current_material)"
silent_activation="$(activation_frame silent_absence)"
python_activation="$(activation_frame python_oracle)"
for activation in "$seal_activation" "$consume_activation" "$extinguish_activation" \
  "$poison_activation" "$current_activation" "$silent_activation" "$python_activation"; do
  [[ -n "$activation" ]] || fail 'Sounio action 9031 fixture extraction failed'
done

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
start_frame="9024 3 1 1 1 0 0 0 0 1 1 1 0 $one $one $zero $zero $one"
request_frame="9024 3 2 1 1 1 0 1 0 1 1 1 0 $one $one $one $zero $one"
response_frame="9024 3 3 1 1 1 0 1 1 1 1 1 0 $one $one $one $one $one"
stop_frame="9024 3 4 1 1 1 1 0 0 1 1 1 0 $one $one $zero $zero $one"
valid_effect="9023 3 1 3 10 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero"
current_closure="9025 3 1 1 1 1 1 1 0 0 12 0 1 3 3 1 1 0 0 0 0 0 0 0 2 $one $one $one $one $one $one $zero $one $one $one $one"

parent_9028='1991017987 113822720 1367310835 4264184359 1117900107 2622180275 1259621157 4224578159'
parent_9025='3253784467 4165106381 4153681002 298013982 643434942 312724736 195896759 132696721'
parent_9023='2365323 2301161672 762924345 38070334 1558458629 1166539901 3590963442 1546541903'
invocation_bindings="$parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $one"
prepare_join='1 1 1 1 1 1'
capsule='1 1 5 6 7 1 1 0 0 1 1 1'
membrane='1 8 9 10 11 1 1 1 1'
scope='1 1 1 1 1 1'
coverage='1 100 1 50 1 1 1 1'
lifecycle='1 1 1 12 13 1 0 0 0'
invocation_outcome='0 0 0 0 0 0 0 0 0 0'
invocation_authority='1 1 1 1 1 1'
invocation_evidence='1 1 10 10'
valid_invocation="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $lifecycle $invocation_outcome $invocation_authority $invocation_evidence $invocation_bindings"
current_invocation="9029 3 1 1 1 0 0 1 0 $capsule $membrane $scope $coverage $lifecycle $invocation_outcome $invocation_authority $invocation_evidence $invocation_bindings"
python_invocation="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $lifecycle $invocation_outcome 0 0 1 1 0 1 $invocation_evidence $invocation_bindings"

parent_9029='1636926980 3205986131 3323207532 3505413428 706242987 2411760920 1929815169 3727939342'
parent_9021='3497534264 556131944 3943529214 1565657389 3821375173 3204015455 2733765994 2625951936'
parent_9022='4125506095 3601417934 2711931735 20635855 2708941890 3284947684 758124027 2068177262'
grant_bindings="$parent_9029 $parent_9021 $parent_9022 $one $one $one $one $one $one"
grant_transition='1 0 1 2 3 4 5 6 7 100 50'
grant_parents='1 1 0 0 0 1 0 0'
grant_identity='1 1 1 1 1 1 1 1'
grant_peer='1 1 1 1 1 1 1 1 1'
grant_shape='1 1 1 1 1 1 1 1'
grant_consumption='1 1 1 1 1 1 1'
grant_revocation='1 1 1 1 1 1 1'
grant_extinction='0 0 0 0 1'
grant_outcome='0 0 0 0 0 0 0 0'
grant_authority='1 1 1 1 1 1 1'
grant_evidence='1 1 11 11'
valid_grant="9030 3 1 $grant_transition $grant_parents $grant_identity $grant_peer $grant_shape $grant_consumption $grant_revocation $grant_extinction $grant_outcome $grant_authority $grant_evidence $grant_bindings"
current_grant="9030 3 1 $grant_transition 1 0 0 0 0 0 0 0 $grant_identity $grant_peer $grant_shape $grant_consumption $grant_revocation $grant_extinction $grant_outcome $grant_authority $grant_evidence $grant_bindings"
python_grant="9030 3 1 $grant_transition $grant_parents $grant_identity $grant_peer $grant_shape $grant_consumption $grant_revocation $grant_extinction $grant_outcome 0 0 0 1 1 1 1 $grant_evidence $grant_bindings"

coproc RESIDENT_PROCESS { "$RUNTIME"; }
resident_pid="$RESIDENT_PROCESS_PID"
exec {resident_read}<&"${RESIDENT_PROCESS[0]}"
exec {resident_write}>&"${RESIDENT_PROCESS[1]}"
start_time="$(sed -n 's/^[^)]*) //p' "/proc/$resident_pid/stat" | awk '{print $20}')"
[[ -n "$start_time" ]] || fail 'resident v5 process birth identity is unavailable'

resident_roundtrip() {
  local label="$1" route="$2" frame="$3" reference="$4" expected actual
  expected="$(printf '%s\n' "$frame" | "$reference" || true)"
  printf '%s\n' "$route" "$frame" >&"$resident_write"
  IFS= read -r -t 3 -u "$resident_read" actual || fail "$label produced no resident response"
  [[ "$actual" == "$expected" ]] ||
    fail "$label diverged from frozen Sounio: resident=$actual reference=$expected"
  printf -v "${label//-/_}_output" '%s' "$actual"
}

resident_roundtrip start 1 "$start_frame" "$RESIDENT_REFERENCE"
resident_roundtrip request 1 "$request_frame" "$RESIDENT_REFERENCE"
resident_roundtrip allow-effect 2 "$valid_effect" "$MEMBRANE_REFERENCE"
resident_roundtrip closure-current 3 "$current_closure" "$CLOSURE_REFERENCE"
resident_roundtrip invocation-prepare 4 "$valid_invocation" "$INVOCATION_REFERENCE"
resident_roundtrip invocation-current 4 "$current_invocation" "$INVOCATION_REFERENCE"
resident_roundtrip invocation-python 4 "$python_invocation" "$INVOCATION_REFERENCE"
resident_roundtrip grant-issue 5 "$valid_grant" "$GRANT_REFERENCE"
resident_roundtrip grant-current 5 "$current_grant" "$GRANT_REFERENCE"
resident_roundtrip grant-python 5 "$python_grant" "$GRANT_REFERENCE"
resident_roundtrip activation-seal 6 "$seal_activation" "$ACTIVATION_REFERENCE"
resident_roundtrip activation-consume 6 "$consume_activation" "$ACTIVATION_REFERENCE"
resident_roundtrip activation-extinguish 6 "$extinguish_activation" "$ACTIVATION_REFERENCE"
resident_roundtrip activation-poison 6 "$poison_activation" "$ACTIVATION_REFERENCE"
resident_roundtrip activation-current 6 "$current_activation" "$ACTIVATION_REFERENCE"
resident_roundtrip activation-silent 6 "$silent_activation" "$ACTIVATION_REFERENCE"
resident_roundtrip activation-python 6 "$python_activation" "$ACTIVATION_REFERENCE"
resident_roundtrip response 1 "$response_frame" "$RESIDENT_REFERENCE"
resident_roundtrip stop 1 "$stop_frame" "$RESIDENT_REFERENCE"

[[ "$closure_current_output" == *'code=447 '* ]] || fail 'action 9025 current material was not DENY447'
[[ "$invocation_current_output" == *'code=481 '* ]] || fail 'action 9029 current material was not DENY481'
[[ "$invocation_python_output" == *'code=488 '* ]] || fail 'action 9029 Python oracle was not DENY488'
[[ "$grant_current_output" == *'code=491 '* ]] || fail 'action 9030 current material was not DENY491'
[[ "$grant_python_output" == *'code=499 '* ]] || fail 'action 9030 Python oracle was not DENY499'
[[ "$activation_seal_output" == *'code=0 '* ]] || fail 'action 9031 SEAL was not ALLOW'
[[ "$activation_consume_output" == *'code=0 '* ]] || fail 'action 9031 CONSUME was not ALLOW'
[[ "$activation_extinguish_output" == *'code=0 '* ]] || fail 'action 9031 EXTINGUISH was not ALLOW'
[[ "$activation_poison_output" == *'code=0 '* ]] || fail 'action 9031 POISON was not ALLOW'
[[ "$activation_current_output" == *'code=502 '* ]] || fail 'action 9031 current material was not DENY502'
[[ "$activation_silent_output" == *'code=507 '* ]] || fail 'action 9031 silent absence was not DENY507'
[[ "$activation_python_output" == *'code=508 '* ]] || fail 'action 9031 Python oracle was not DENY508'

printf '%s\n' '9' >&"$resident_write"
IFS= read -r -t 3 -u "$resident_read" malformed_route || fail 'malformed route produced no response'
[[ "$malformed_route" == 'SOUNIO_RESIDENT_PROCESS_DENY code=424 reason=malformed-route stage=INVALID' ]] ||
  fail "malformed route was not refused: $malformed_route"

[[ -r "/proc/$resident_pid/stat" ]] || fail 'resident v5 process exited during the generation'
end_time="$(sed -n 's/^[^)]*) //p' "/proc/$resident_pid/stat" | awk '{print $20}')"
[[ "$end_time" == "$start_time" ]] || fail 'resident v5 process birth identity changed'

printf '%s\n' '0' >&"$resident_write"
exec {resident_write}>&-
wait "$resident_pid"
resident_pid=''
exec {resident_read}<&-

printf '%s\n' \
  'sounio-loom-resident-transport-v5-selftest: PASS semantic_authority=Sounio operational_realization=resident-Sounio actions=9023,9024,9025,9029,9030,9031 process_identity=stable exact_output_parity=19/19 activation=ALLOWx4 activation_current=DENY502 activation_silent=DENY507 activation_python=DENY508 invocation_current=DENY481 invocation_python=DENY488 grant_issue=ALLOW grant_current=DENY491 grant_python=DENY499 closure_current=DENY447 malformed_route=DENY424 ocaml_capsule_started=false capsule_material=false production_activation=false launch_open=false recycle_open=false same_uid_peer_isolation=true exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false python_executed=false rust_executed=false'
