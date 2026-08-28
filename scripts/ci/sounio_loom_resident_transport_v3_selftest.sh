#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-transport-v3.XXXXXX")"
RUNTIME="$TEST_ROOT/resident-membrane-v3"
MEMBRANE_REFERENCE="$TEST_ROOT/subprocess-membrane"
RESIDENT_REFERENCE="$TEST_ROOT/resident-authority"
CLOSURE_REFERENCE="$TEST_ROOT/effect-closure-authority"
CELL_REFERENCE="$TEST_ROOT/kernel-invocation-cell-authority"

cleanup() {
  if [[ -n "${resident_pid:-}" ]]; then
    kill "$resident_pid" 2>/dev/null || true
    wait "$resident_pid" 2>/dev/null || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-transport-v3-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_RESIDENT_MEMBRANE_V3_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v3.sh" >/dev/null
SOUNIO_LOOM_SUBPROCESS_MEMBRANE_OUTPUT="$MEMBRANE_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_subprocess_membrane.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_AUTHORITY_OUTPUT="$RESIDENT_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_authority.sh" >/dev/null
SOUNIO_LOOM_EFFECT_CLOSURE_OUTPUT="$CLOSURE_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_effect_closure_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$CELL_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_invocation_cell_authority.sh" >/dev/null

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
bindings="$parent_9028 $parent_9025 $parent_9023 $one $one $one $one $one $one $one $one $one"
prepare_join='1 1 1 1 1 1'
capsule='1 1 5 6 7 1 1 0 0 1 1 1'
membrane='1 8 9 10 11 1 1 1 1'
scope='1 1 1 1 1 1'
coverage='1 100 1 50 1 1 1 1'
lifecycle='1 1 1 12 13 1 0 0 0'
outcome='0 0 0 0 0 0 0 0 0 0'
authority='1 1 1 1 1 1'
evidence='1 1 10 10'
valid_cell="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $lifecycle $outcome $authority $evidence $bindings"
current_cell="9029 3 1 1 1 0 0 1 0 $capsule $membrane $scope $coverage $lifecycle $outcome $authority $evidence $bindings"
python_cell="9029 3 1 $prepare_join $capsule $membrane $scope $coverage $lifecycle $outcome 0 0 1 1 0 1 $evidence $bindings"

coproc RESIDENT_PROCESS { "$RUNTIME"; }
resident_pid="$RESIDENT_PROCESS_PID"
exec {resident_read}<&"${RESIDENT_PROCESS[0]}"
exec {resident_write}>&"${RESIDENT_PROCESS[1]}"
start_time="$(sed -n 's/^[^)]*) //p' "/proc/$resident_pid/stat" | awk '{print $20}')"
[[ -n "$start_time" ]] || fail 'resident v3 process birth identity is unavailable'

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
resident_roundtrip cell-prepare 4 "$valid_cell" "$CELL_REFERENCE"
resident_roundtrip cell-current 4 "$current_cell" "$CELL_REFERENCE"
resident_roundtrip cell-python 4 "$python_cell" "$CELL_REFERENCE"
resident_roundtrip response 1 "$response_frame" "$RESIDENT_REFERENCE"
resident_roundtrip stop 1 "$stop_frame" "$RESIDENT_REFERENCE"

[[ "$closure_current_output" == *'code=447 '* ]] || fail 'action 9025 current material was not DENY447'
[[ "$cell_current_output" == *'code=481 '* ]] || fail 'action 9029 current material was not DENY481'
[[ "$cell_python_output" == *'code=488 '* ]] || fail 'action 9029 Python oracle was not DENY488'

printf '%s\n' '9' >&"$resident_write"
IFS= read -r -t 3 -u "$resident_read" malformed_route || fail 'malformed route produced no response'
[[ "$malformed_route" == 'SOUNIO_RESIDENT_PROCESS_DENY code=424 reason=malformed-route stage=INVALID' ]] ||
  fail "malformed route was not refused: $malformed_route"

[[ -r "/proc/$resident_pid/stat" ]] || fail 'resident v3 process exited during the generation'
end_time="$(sed -n 's/^[^)]*) //p' "/proc/$resident_pid/stat" | awk '{print $20}')"
[[ "$end_time" == "$start_time" ]] || fail 'resident v3 process birth identity changed'

printf '%s\n' '0' >&"$resident_write"
exec {resident_write}>&-
wait "$resident_pid"
resident_pid=''
exec {resident_read}<&-

printf '%s\n' \
  'sounio-loom-resident-transport-v3-selftest: PASS semantic_authority=Sounio operational_realization=resident-Sounio actions=9023,9024,9025,9029 process_identity=stable exact_output_parity=9/9 cell_prepare=ALLOW cell_current=DENY481 cell_python=DENY488 closure_current=DENY447 malformed_route=DENY424 ocaml_invocation_started=false material_invocation=false material_coverage=false exec_attached=false commit_attached=false ci_attached=false'
