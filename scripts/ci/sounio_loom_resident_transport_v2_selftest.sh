#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-transport-v2.XXXXXX")"
RUNTIME="$TEST_ROOT/resident-membrane-v2"
MEMBRANE_REFERENCE="$TEST_ROOT/subprocess-membrane"
RESIDENT_REFERENCE="$TEST_ROOT/resident-authority"
CLOSURE_REFERENCE="$TEST_ROOT/effect-closure-authority"

cleanup() {
  if [[ -n "${resident_pid:-}" ]]; then
    kill "$resident_pid" 2>/dev/null || true
    wait "$resident_pid" 2>/dev/null || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-transport-v2-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v2.sh" >/dev/null
SOUNIO_LOOM_SUBPROCESS_MEMBRANE_OUTPUT="$MEMBRANE_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_subprocess_membrane.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_AUTHORITY_OUTPUT="$RESIDENT_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_authority.sh" >/dev/null
SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$CLOSURE_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
start_frame="9024 3 1 1 1 0 0 0 0 1 1 1 0 $one $one $zero $zero $one"
request_frame="9024 3 2 1 1 1 0 1 0 1 1 1 0 $one $one $one $zero $one"
response_frame="9024 3 3 1 1 1 0 1 1 1 1 1 0 $one $one $one $one $one"
stop_frame="9024 3 4 1 1 1 1 0 0 1 1 1 0 $one $one $zero $zero $one"
replay_frame="9024 3 2 1 1 1 1 1 0 1 1 1 0 $one $one $one $zero $one"
valid_effect="9023 3 1 3 10 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero"
python_effect="9023 3 1 3 7 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero"
all_bindings="$one $one $one $one $one $one $one $one $one $one $one"
closed_coverage='3 3 3 2 2 2 2 2 2 2 2 2'
valid_closure="9025 3 1 1 1 1 1 1 1 1 12 12 1 $closed_coverage $all_bindings"
current_material_closure="9025 3 1 1 1 1 1 1 0 0 12 0 1 3 3 1 1 0 0 0 0 0 0 0 2 $one $one $one $one $one $one $zero $one $one $one $one"
same_uid_closure="9025 3 1 1 1 1 1 1 0 1 12 12 1 $closed_coverage $all_bindings"
unknown_closure="9025 3 1 1 1 1 1 1 1 1 12 12 1 3 3 3 2 2 2 2 2 2 2 2 3 $all_bindings"

coproc RESIDENT_PROCESS { "$RUNTIME"; }
resident_pid="$RESIDENT_PROCESS_PID"
exec {resident_read}<&"${RESIDENT_PROCESS[0]}"
exec {resident_write}>&"${RESIDENT_PROCESS[1]}"
start_time="$(sed -n 's/^[^)]*) //p' "/proc/$resident_pid/stat" | awk '{print $20}')"
[[ -n "$start_time" ]] || fail 'resident v2 process birth identity is unavailable'

resident_roundtrip() {
  local label="$1" route="$2" frame="$3" reference="$4" expected actual
  expected="$(printf '%s\n' "$frame" | "$reference" || true)"
  printf '%s\n' "$route" "$frame" >&"$resident_write"
  IFS= read -r -t 3 -u "$resident_read" actual || fail "$label produced no resident response"
  [[ "$actual" == "$expected" ]] ||
    fail "$label diverged from frozen Sounio: resident=$actual reference=$expected"
}

resident_roundtrip start 1 "$start_frame" "$RESIDENT_REFERENCE"
resident_roundtrip request 1 "$request_frame" "$RESIDENT_REFERENCE"
resident_roundtrip allow-effect 2 "$valid_effect" "$MEMBRANE_REFERENCE"
resident_roundtrip response 1 "$response_frame" "$RESIDENT_REFERENCE"
resident_roundtrip replay 1 "$replay_frame" "$RESIDENT_REFERENCE"
resident_roundtrip deny-python 2 "$python_effect" "$MEMBRANE_REFERENCE"
resident_roundtrip closure-allow 3 "$valid_closure" "$CLOSURE_REFERENCE"
resident_roundtrip closure-current-material 3 "$current_material_closure" "$CLOSURE_REFERENCE"
resident_roundtrip closure-same-uid 3 "$same_uid_closure" "$CLOSURE_REFERENCE"
resident_roundtrip closure-unknown 3 "$unknown_closure" "$CLOSURE_REFERENCE"
resident_roundtrip stop 1 "$stop_frame" "$RESIDENT_REFERENCE"

printf '%s\n' '9' >&"$resident_write"
IFS= read -r -t 3 -u "$resident_read" malformed_route || fail 'malformed route produced no response'
[[ "$malformed_route" == 'SOUNIO_RESIDENT_PROCESS_DENY code=424 reason=malformed-route stage=INVALID' ]] ||
  fail "malformed route was not refused: $malformed_route"

[[ -r "/proc/$resident_pid/stat" ]] || fail 'resident v2 process exited during the generation'
end_time="$(sed -n 's/^[^)]*) //p' "/proc/$resident_pid/stat" | awk '{print $20}')"
[[ "$end_time" == "$start_time" ]] || fail 'resident v2 process birth identity changed'

printf '%s\n' '0' >&"$resident_write"
exec {resident_write}>&-
wait "$resident_pid"
resident_pid=''
exec {resident_read}<&-

printf '%s\n' \
  'sounio-loom-resident-transport-v2-selftest: PASS semantic_authority=Sounio operational_realization=resident-Sounio actions=9023,9024,9025 process_identity=stable exact_output_parity=11/11 replay=DENY442 python=DENY410 closure_current=DENY447 closure_same_uid=DENY451 closure_unknown=DENY452 malformed_route=DENY424 ocaml_v2_started=false material_coverage=false membrane_v2_integration=false'
