#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/kernel_invocation_cell.material.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-kernel-invocation-cell-material-admission-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-invocation-material-freeze.XXXXXX")"
AUTHORITY="$TEST_ROOT/invocation-authority"
BROKER_ONE="$TEST_ROOT/principal-broker-one"
BROKER_TWO="$TEST_ROOT/principal-broker-two"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-invocation-cell-material-admission-freeze-selftest: FAIL: %s\n' "$*" >&2
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

[[ -f "$MANIFEST" ]] || fail 'material admission manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'material admission evidence is missing'
[[ "$(field schema)" == loom-kernel-invocation-cell-material-admission-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == MATERIAL_PARITY_FROZEN ]] || fail 'material parity is not frozen'
[[ "$(field producing_language)" == C++20 ]] || fail 'producer is not C++20'
[[ "$(field language_role)" == MATERIAL_PARITY ]] || fail 'wrong language role'
[[ "$(field semantic_authority)" == Sounio ]] || fail 'Sounio is not semantic authority'
[[ "$(field action)" == 9029 ]] || fail 'wrong semantic action'
[[ "$(field transitory_bootstrap)" == true ]] || fail 'C++ bootstrap is not marked transitory'
for boundary in expected_results_encoded_in_cpp material_invocation material_coverage same_uid_peer_isolation launch_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  [[ "$(field "$boundary")" == false ]] || fail "$boundary was promoted during material freeze"
done

implementation_commit="$(field implementation_commit)"
git -C "$ROOT_DIR" cat-file -e "${implementation_commit}^{commit}" || fail 'implementation commit is absent'
for pair in \
  broker_source_path:broker_source_sha256 \
  build_script_path:build_script_sha256 \
  gate_script_path:gate_script_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$implementation_commit:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the implementation commit"
done
for pair in \
  freeze_selftest_path:freeze_selftest_sha256 \
  contract_path:contract_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$(field "$hash_key")" ]] || fail "$path drifted"
done

broker_source="$ROOT_DIR/$(field broker_source_path)"
if grep -Eq 'code=481|parent-semantic-join-incomplete' "$broker_source"; then
  fail 'C++ source encodes the current-material expected result'
fi

parent_manifest="$ROOT_DIR/$(field parent_9029_manifest_path)"
[[ "$(file_hash "$parent_manifest")" == "$(field parent_9029_manifest_sha256)" ]] ||
  fail 'frozen action 9029 manifest drifted'
[[ "$(record_field "$parent_manifest" stage)" == SEMANTICS_FROZEN ]] || fail 'action 9029 is not frozen'
[[ "$(record_field "$parent_manifest" producing_language)" == Sounio ]] || fail 'action 9029 producer drifted'
[[ "$(record_field "$parent_manifest" language_role)" == SEMANTIC_AUTHORITY ]] || fail 'action 9029 role drifted'
[[ "$(record_field "$parent_manifest" action)" == 9029 ]] || fail 'parent action differs'
[[ "$(record_field "$parent_manifest" semantics_sha256)" == "$(field parent_9029_semantics_sha256)" ]] ||
  fail 'parent semantics hash differs'

SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_invocation_cell_authority.sh" >/dev/null
[[ "$(file_hash "$AUTHORITY")" == "$(field parent_9029_executable_sha256)" ]] ||
  fail 'source-built Sounio action 9029 executable differs'

SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_ONE" \
  bash "$ROOT_DIR/$(field build_script_path)" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_TWO" \
  bash "$ROOT_DIR/$(field build_script_path)" >/dev/null
cmp "$BROKER_ONE" "$BROKER_TWO" || fail 'two material broker rebuilds differ'
[[ "$(file_hash "$BROKER_ONE")" == "$(field broker_binary_sha256)" ]] ||
  fail 'material broker binary hash differs'

cxx="$(field cxx_path)"
[[ "$(file_hash "$cxx")" == "$(field cxx_sha256)" ]] || fail 'C++ compiler binary drifted'
[[ "$($cxx --version | sed -n '1p')" == "$(field cxx_version)" ]] || fail 'C++ compiler version drifted'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_kernel_invocation_cell_material_admission_selftest.sh' ]] ||
  fail 'unexpected material admission command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] ||
  fail 'command hash differs'
result="$(bash "$ROOT_DIR/$(field gate_script_path)")"
[[ "$result" == "$(field result)" ]] || fail 'material admission gate result differs'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] ||
  fail 'material admission result hash differs'
for binding in \
  "manifest_sha256=$(field parent_9029_manifest_sha256)" \
  "authority_sha256=$(field parent_9029_executable_sha256)" \
  "positive_frame_sha256=$(field positive_frame_sha256)" \
  "current_frame_sha256=$(field current_frame_sha256)" \
  "source_sha256=$(field broker_source_sha256)" \
  "binary_sha256=$(field broker_binary_sha256)"; do
  [[ "$result" == *"$binding"* ]] || fail "gate omitted $binding"
done

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fqx "manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind material manifest'
grep -Fqx "parent_9029_manifest_sha256=$(field parent_9029_manifest_sha256)" "$EVIDENCE" ||
  fail 'evidence does not bind frozen Sounio authority'
grep -Fqx "broker_binary_sha256=$(field broker_binary_sha256)" "$EVIDENCE" ||
  fail 'evidence does not bind material binary'

printf '%s\n' \
  "sounio-loom-kernel-invocation-cell-material-admission-freeze-selftest: PASS semantic_authority=Sounio material_parity=C++20 action=9029 manifest_sha256=$manifest_hash binary_sha256=$(field broker_binary_sha256) rebuilds=2 positive=ALLOW current_material=DENY481 expected_results_encoded_in_cpp=false launch_open=false material_invocation=false material_coverage=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false"
