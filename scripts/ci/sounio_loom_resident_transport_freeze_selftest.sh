#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-resident-transport-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-resident-transport-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/resident-one"
RUNTIME_TWO="$TEST_ROOT/resident-two"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-transport-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
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

[[ -f "$MANIFEST" ]] || fail 'resident runtime manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'resident runtime evidence is missing'
[[ "$(field schema)" == loom-resident-membrane-runtime-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SOUNIO_RESIDENT_REALIZATION ]] || fail 'wrong runtime stage'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field actions)" == 9023,9024 ]] || fail 'runtime actions differ'
[[ "$(field runtime_frozen)" == true ]] || fail 'runtime was not frozen'
[[ "$(field ocaml_resident_started)" == false ]] || fail 'freeze opened OCaml realization'
[[ "$(field performance_gate)" == false ]] || fail 'freeze promoted performance'
[[ "$(field membrane_integration)" == false ]] || fail 'freeze promoted integration'
for surface in exec_attached commit_attached ci_attached; do
  [[ "$(field "$surface")" == false ]] || fail "$surface was promoted during freeze"
done

resident_commit="$(field sounio_resident_commit)"
git -C "$ROOT_DIR" cat-file -e "${resident_commit}^{commit}" || fail 'resident Sounio commit is absent'
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

parent_9023="$ROOT_DIR/$(field parent_9023_manifest_path)"
parent_9024="$ROOT_DIR/$(field parent_9024_manifest_path)"
[[ "$(file_hash "$parent_9023")" == "$(field parent_9023_manifest_sha256)" ]] || fail '9023 parent drifted'
[[ "$(file_hash "$parent_9024")" == "$(field parent_9024_manifest_sha256)" ]] || fail '9024 parent drifted'
[[ "$(grep -m1 '^stage=' "$parent_9023" | cut -d= -f2)" == SEMANTICS_FROZEN ]] || fail '9023 is not frozen'
[[ "$(grep -m1 '^stage=' "$parent_9024" | cut -d= -f2)" == SEMANTICS_FROZEN ]] || fail '9024 is not frozen'
[[ "$(grep -m1 '^action=' "$parent_9023" | cut -d= -f2)" == 9023 ]] || fail 'wrong 9023 parent'
[[ "$(grep -m1 '^action=' "$parent_9024" | cut -d= -f2)" == 9024 ]] || fail 'wrong 9024 parent'

[[ "$(field route_9024)" == 1 ]] || fail '9024 route changed'
[[ "$(field route_9023)" == 2 ]] || fail '9023 route changed'
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

SOUNIO_LOOM_RESIDENT_MEMBRANE_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_OUTPUT="$RUNTIME_ONE" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_OUTPUT="$RUNTIME_TWO" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'resident Sounio rebuilds differ'
[[ "$(file_hash "$RUNTIME_ONE")" == "$(field runtime_sha256)" ]] || fail 'resident runtime hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_resident_transport_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$gate_script_path")"
[[ "$result" == "$(field result)" ]] || fail 'resident gate result differs'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'resident gate hash differs'

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "runtime_manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind runtime manifest'

printf '%s\n' \
  "sounio-loom-resident-transport-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SOUNIO_RESIDENT_REALIZATION actions=9023,9024 runtime_sha256=$(field runtime_sha256) manifest_sha256=$manifest_hash rebuilds=2 process_identity=stable exact_output_parity=7/7 ocaml_resident_started=false performance_gate=false membrane_integration=false exec_attached=false commit_attached=false ci_attached=false"
