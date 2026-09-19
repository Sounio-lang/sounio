#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v5"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-resident-transport-v5-20260829.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-resident-transport-v5-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/resident-v5-one"
RUNTIME_TWO="$TEST_ROOT/resident-v5-two"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-transport-v5-freeze-selftest: FAIL: %s\n' "$*" >&2
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

[[ -f "$MANIFEST" ]] || fail 'resident v5 runtime manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'resident v5 runtime evidence is missing'
[[ "$(field schema)" == loom-resident-membrane-runtime-v5 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SOUNIO_RESIDENT_REALIZATION ]] || fail 'wrong runtime stage'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field actions)" == 9023,9024,9025,9029,9030,9031 ]] || fail 'runtime actions differ'
[[ "$(field runtime_frozen)" == true ]] || fail 'runtime was not frozen'
[[ "$(field same_uid_peer_isolation)" == true ]] || fail 'frozen action 9031 lost its material peer-isolation parent'
for boundary in ocaml_capsule_started capsule_material production_activation launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  [[ "$(field "$boundary")" == false ]] || fail "$boundary was promoted during freeze"
done

resident_commit="$(field sounio_resident_v5_commit)"
git -C "$ROOT_DIR" cat-file -e "${resident_commit}^{commit}" || fail 'resident Sounio v5 commit is absent'
dispatcher_path="$(field dispatcher_path)"
build_script_path="$(field build_script_path)"
gate_script_path="$(field gate_script_path)"
freeze_script_path="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$dispatcher_path")" == "$(field dispatcher_sha256)" ]] || fail 'dispatcher drifted'
[[ "$(file_hash "$ROOT_DIR/$build_script_path")" == "$(field build_script_sha256)" ]] || fail 'build script drifted'
[[ "$(file_hash "$ROOT_DIR/$gate_script_path")" == "$(field gate_script_sha256)" ]] || fail 'gate script drifted'
[[ "$(file_hash "$ROOT_DIR/$freeze_script_path")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze selftest drifted'
[[ "$(git -C "$ROOT_DIR" show "$resident_commit:$dispatcher_path" | stream_hash)" == "$(field dispatcher_sha256)" ]] || fail 'committed dispatcher differs'
[[ "$(git -C "$ROOT_DIR" show "$resident_commit:$build_script_path" | stream_hash)" == "$(field build_script_sha256)" ]] || fail 'committed build script differs'
[[ "$(git -C "$ROOT_DIR" show "$resident_commit:$gate_script_path" | stream_hash)" == "$(field gate_script_sha256)" ]] || fail 'committed gate script differs'
[[ "$(git -C "$ROOT_DIR" show "$resident_commit:$freeze_script_path" | stream_hash)" == "$(field freeze_selftest_sha256)" ]] || fail 'committed freeze selftest differs'

for action in 9023 9024 9025 9029 9030 9031; do
  parent="$ROOT_DIR/$(field "parent_${action}_manifest_path")"
  [[ "$(file_hash "$parent")" == "$(field "parent_${action}_sha256")" ]] || fail "parent $action manifest drifted"
  [[ "$(record_field "$parent" action)" == "$action" ]] || fail "wrong parent action $action"
  [[ "$(record_field "$parent" stage)" == SEMANTICS_FROZEN ]] || fail "parent action $action is not frozen"
  [[ "$(record_field "$parent" producing_language)" == Sounio ]] || fail "parent action $action is not Sounio"
done
parent_v4="$ROOT_DIR/$(field parent_resident_v4_manifest_path)"
[[ "$(file_hash "$parent_v4")" == "$(field parent_resident_v4_sha256)" ]] || fail 'resident v4 manifest drifted'
[[ "$(record_field "$parent_v4" runtime_frozen)" == true ]] || fail 'resident v4 is not frozen'
[[ "$(record_field "$parent_v4" parent_9030_sha256)" == "$(field parent_9030_sha256)" ]] || fail 'resident v4 action 9030 binding differs'
[[ "$(record_field "$ROOT_DIR/$(field parent_9030_manifest_path)" parent_9029_manifest_sha256)" == "$(field parent_9029_sha256)" ]] || fail 'action 9030 parent 9029 binding differs'
[[ "$(record_field "$ROOT_DIR/$(field parent_9031_manifest_path)" parent_9030_manifest_sha256)" == "$(field parent_9030_sha256)" ]] || fail 'action 9031 parent 9030 binding differs'
[[ "$(record_field "$ROOT_DIR/$(field parent_9031_manifest_path)" parent_9025_manifest_sha256)" == "$(field parent_9025_v13_sha256)" ]] || fail 'action 9031 V13 material parent binding differs'
parent_9025_v13="$ROOT_DIR/$(field parent_9025_v13_manifest_path)"
[[ "$(file_hash "$parent_9025_v13")" == "$(field parent_9025_v13_sha256)" ]] || fail 'V13 action 9025 material manifest drifted'
[[ "$(record_field "$parent_9025_v13" action_9025_allow)" == true ]] || fail 'V13 action 9025 is not ALLOW'
[[ "$(record_field "$parent_9025_v13" same_uid_peer_isolation)" == true ]] || fail 'V13 same-UID peer isolation is not material'

wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$resident_commit" "$wrapper_path" "$compiler_path" | tar -x -C "$TOOLCHAIN_ROOT"
[[ "$(file_hash "$TOOLCHAIN_ROOT/$wrapper_path")" == "$(field toolchain_wrapper_sha256)" ]] || fail 'frozen compiler wrapper drifted'
[[ "$(file_hash "$TOOLCHAIN_ROOT/$compiler_path")" == "$(field toolchain_compiler_sha256)" ]] || fail 'frozen compiler binary drifted'

SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$RUNTIME_ONE" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V5_OUTPUT="$RUNTIME_TWO" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio resident v5 rebuilds differ'
[[ "$(file_hash "$RUNTIME_ONE")" == "$(field runtime_sha256)" ]] || fail 'rebuilt resident v5 hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_resident_transport_v5_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$gate_script_path")"
[[ "$result" == "$(field result)" ]] || fail 'gate result differs'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'gate result hash differs'

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the manifest hash'

printf '%s\n' \
  "sounio-loom-resident-transport-v5-freeze-selftest: PASS semantic_authority=Sounio operational_realization=resident-Sounio actions=9023,9024,9025,9029,9030,9031 manifest_sha256=$manifest_hash runtime_sha256=$(field runtime_sha256) rebuilds=2 process_identity=stable exact_output_parity=19/19 activation=ALLOWx4 activation_current=DENY502 activation_silent=DENY507 activation_python=DENY508 same_uid_peer_isolation=true ocaml_capsule_started=false capsule_material=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false python_executed=false rust_executed=false"
