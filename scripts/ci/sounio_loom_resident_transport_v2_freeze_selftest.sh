#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v2"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-resident-transport-v2-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-resident-transport-v2-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/resident-v2-one"
RUNTIME_TWO="$TEST_ROOT/resident-v2-two"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-transport-v2-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "record field $key occurs $count times in $path"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

stream_hash() {
  local sum
  sum="$(sha256sum)"
  printf '%s' "${sum%% *}"
}

[[ -f "$MANIFEST" ]] || fail 'resident v2 runtime manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'resident v2 runtime evidence is missing'
[[ "$(field schema)" == loom-resident-membrane-runtime-v2 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SOUNIO_RESIDENT_REALIZATION ]] || fail 'wrong runtime stage'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field actions)" == 9023,9024,9025 ]] || fail 'runtime actions differ'
[[ "$(field runtime_frozen)" == true ]] || fail 'runtime was not frozen'
[[ "$(field ocaml_v2_started)" == false ]] || fail 'freeze opened OCaml v2 realization'
[[ "$(field material_coverage)" == false ]] || fail 'freeze promoted material coverage'
[[ "$(field membrane_v2_integration)" == false ]] || fail 'freeze promoted native integration'
for surface in exec_attached commit_attached ci_attached; do
  [[ "$(field "$surface")" == false ]] || fail "$surface was promoted during freeze"
done

resident_commit="$(field sounio_resident_v2_commit)"
git -C "$ROOT_DIR" cat-file -e "${resident_commit}^{commit}" || fail 'resident Sounio v2 commit is absent'
dispatcher_path="$(field dispatcher_path)"
build_script_path="$(field build_script_path)"
gate_script_path="$(field gate_script_path)"
freeze_script_path="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$dispatcher_path")" == "$(field dispatcher_sha256)" ]] || fail 'dispatcher drifted'
[[ "$(file_hash "$ROOT_DIR/$build_script_path")" == "$(field build_script_sha256)" ]] || fail 'build script drifted'
[[ "$(file_hash "$ROOT_DIR/$gate_script_path")" == "$(field gate_script_sha256)" ]] || fail 'gate script drifted'
[[ "$(file_hash "$ROOT_DIR/$freeze_script_path")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze script drifted'
[[ "$(git -C "$ROOT_DIR" show "$resident_commit:$dispatcher_path" | stream_hash)" == "$(field dispatcher_sha256)" ]] ||
  fail 'committed dispatcher differs'
[[ "$(git -C "$ROOT_DIR" show "$resident_commit:$build_script_path" | stream_hash)" == "$(field build_script_sha256)" ]] ||
  fail 'committed build script differs'
[[ "$(git -C "$ROOT_DIR" show "$resident_commit:$gate_script_path" | stream_hash)" == "$(field gate_script_sha256)" ]] ||
  fail 'committed gate script differs'

parent_9023="$ROOT_DIR/$(field parent_9023_manifest_path)"
parent_9024="$ROOT_DIR/$(field parent_9024_manifest_path)"
parent_9025="$ROOT_DIR/$(field parent_9025_manifest_path)"
parent_resident_v1="$ROOT_DIR/$(field parent_resident_v1_manifest_path)"
for parent_name in parent_9023 parent_9024 parent_9025 parent_resident_v1; do
  parent_var="${parent_name}"
  parent_path="${!parent_var}"
  [[ "$(file_hash "$parent_path")" == "$(field "${parent_name}_sha256")" ]] ||
    fail "$parent_name drifted"
done
[[ "$(record_field "$parent_9023" stage)" == SEMANTICS_FROZEN ]] || fail '9023 is not frozen'
[[ "$(record_field "$parent_9024" stage)" == SEMANTICS_FROZEN ]] || fail '9024 is not frozen'
[[ "$(record_field "$parent_9025" stage)" == SEMANTICS_FROZEN ]] || fail '9025 is not frozen'
[[ "$(record_field "$parent_9023" action)" == 9023 ]] || fail 'wrong 9023 parent'
[[ "$(record_field "$parent_9024" action)" == 9024 ]] || fail 'wrong 9024 parent'
[[ "$(record_field "$parent_9025" action)" == 9025 ]] || fail 'wrong 9025 parent'
[[ "$(record_field "$parent_resident_v1" runtime_frozen)" == true ]] || fail 'resident v1 is not frozen'
[[ "$(record_field "$parent_9025" parent_9023_manifest_sha256)" == "$(field parent_9023_sha256)" ]] ||
  fail '9025 does not bind 9023'
[[ "$(record_field "$parent_9025" parent_9024_manifest_sha256)" == "$(field parent_9024_sha256)" ]] ||
  fail '9025 does not bind 9024'
[[ "$(record_field "$parent_9025" resident_runtime_manifest_sha256)" == "$(field parent_resident_v1_sha256)" ]] ||
  fail '9025 does not bind resident v1'

[[ "$(field route_9024)" == 1 ]] || fail '9024 route changed'
[[ "$(field route_9023)" == 2 ]] || fail '9023 route changed'
[[ "$(field route_9025)" == 3 ]] || fail '9025 route changed'
[[ "$(field route_stop)" == 0 ]] || fail 'stop route changed'
[[ "$(field max_frame_bytes)" == 65535 ]] || fail 'frame bound changed'
[[ "$(field framing)" == sounio-read-byte-newline ]] || fail 'framing changed'

wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$resident_commit" "$wrapper_path" "$compiler_path" |
  tar -x -C "$TOOLCHAIN_ROOT"
[[ "$(file_hash "$TOOLCHAIN_ROOT/$wrapper_path")" == "$(field toolchain_wrapper_sha256)" ]] ||
  fail 'frozen compiler wrapper drifted'
[[ "$(file_hash "$TOOLCHAIN_ROOT/$compiler_path")" == "$(field toolchain_compiler_sha256)" ]] ||
  fail 'frozen compiler binary drifted'

SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_OUTPUT="$RUNTIME_ONE" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V2_OUTPUT="$RUNTIME_TWO" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'resident Sounio v2 rebuilds differ'
[[ "$(file_hash "$RUNTIME_ONE")" == "$(field runtime_sha256)" ]] || fail 'resident v2 runtime hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_resident_transport_v2_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$gate_script_path")"
[[ "$result" == "$(field result)" ]] || fail 'resident v2 gate result differs'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'resident v2 gate hash differs'

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "runtime_manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind runtime v2 manifest'

printf '%s\n' \
  "sounio-loom-resident-transport-v2-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SOUNIO_RESIDENT_REALIZATION actions=9023,9024,9025 runtime_sha256=$(field runtime_sha256) manifest_sha256=$manifest_hash rebuilds=2 process_identity=stable exact_output_parity=11/11 closure_current=DENY447 closure_same_uid=DENY451 closure_unknown=DENY452 ocaml_v2_started=false material_coverage=false membrane_v2_integration=false exec_attached=false commit_attached=false ci_attached=false"
