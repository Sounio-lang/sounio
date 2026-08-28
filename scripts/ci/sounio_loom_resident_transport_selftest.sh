#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-transport.XXXXXX")"
RUNTIME="$TEST_ROOT/resident-membrane"
MEMBRANE_REFERENCE="$TEST_ROOT/subprocess-membrane"
RESIDENT_REFERENCE="$TEST_ROOT/resident-authority"

cleanup() {
  if [[ -n "${resident_pid:-}" ]]; then
    kill "$resident_pid" 2>/dev/null || true
    wait "$resident_pid" 2>/dev/null || true
  fi
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-transport-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_RESIDENT_MEMBRANE_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane.sh" >/dev/null
SOUNIO_LOOM_SUBPROCESS_MEMBRANE_OUTPUT="$MEMBRANE_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_subprocess_membrane.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_AUTHORITY_OUTPUT="$RESIDENT_REFERENCE" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_authority.sh" >/dev/null

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
start_frame="9024 3 1 1 1 0 0 0 0 1 1 1 0 $one $one $zero $zero $one"
request_frame="9024 3 2 1 1 1 0 1 0 1 1 1 0 $one $one $one $zero $one"
response_frame="9024 3 3 1 1 1 0 1 1 1 1 1 0 $one $one $one $one $one"
stop_frame="9024 3 4 1 1 1 1 0 0 1 1 1 0 $one $one $zero $zero $one"
replay_frame="9024 3 2 1 1 1 1 1 0 1 1 1 0 $one $one $one $zero $one"
valid_effect="9023 3 1 3 10 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero"
python_effect="9023 3 1 3 7 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero"

coproc RESIDENT_PROCESS { "$RUNTIME"; }
resident_pid="$RESIDENT_PROCESS_PID"
exec {resident_read}<&"${RESIDENT_PROCESS[0]}"
exec {resident_write}>&"${RESIDENT_PROCESS[1]}"
start_time="$(sed -n 's/^[^)]*) //p' "/proc/$resident_pid/stat" | awk '{print $20}')"
[[ -n "$start_time" ]] || fail 'resident process birth identity is unavailable'

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
resident_roundtrip stop 1 "$stop_frame" "$RESIDENT_REFERENCE"

[[ -r "/proc/$resident_pid/stat" ]] || fail 'resident process exited during the generation'
end_time="$(sed -n 's/^[^)]*) //p' "/proc/$resident_pid/stat" | awk '{print $20}')"
[[ "$end_time" == "$start_time" ]] || fail 'resident process birth identity changed'

printf '%s\n' '0' >&"$resident_write"
exec {resident_write}>&-
wait "$resident_pid"
resident_pid=''
exec {resident_read}<&-

printf '%s\n' \
  'sounio-loom-resident-transport-selftest: PASS semantic_authority=Sounio operational_realization=resident-Sounio actions=9023,9024 process_identity=stable exact_output_parity=7/7 replay=DENY442 python=DENY410 ocaml_started=false performance_gate=false membrane_integration=false'
