#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/host_exec_quorum.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-host-exec-quorum-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-exec-quorum-freeze.XXXXXX")"
BROKER_ONE="$TEST_ROOT/broker-one"
BROKER_TWO="$TEST_ROOT/broker-two"
BARRIER_ONE="$TEST_ROOT/barrier-one"
BARRIER_TWO="$TEST_ROOT/barrier-two"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-host-exec-quorum-freeze-selftest: FAIL: %s\n' "$*" >&2
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

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'ExecQuorum manifest is missing or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'ExecQuorum evidence is missing or linked'
[[ "$(field schema)" == loom-host-exec-quorum-runtime-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == MATERIAL_PARITY_FROZEN ]] || fail 'wrong ExecQuorum stage'
[[ "$(field producing_language)" == C++20+Linux ]] || fail 'wrong material producer'
[[ "$(field language_role)" == MATERIAL_PARITY ]] || fail 'wrong material language role'
[[ "$(field semantic_authority)" == Sounio ]] || fail 'Sounio is not semantic authority'
[[ "$(field controller_language)" == OCaml ]] || fail 'OCaml controller is absent'
[[ "$(field controller_role)" == EFFECT_PARITY ]] || fail 'OCaml role drifted'
[[ "$(field action)" == 9030 ]] || fail 'wrong semantic action'
[[ "$(field single_resident_controller)" == true ]] || fail 'controller is not single-resident'
[[ "$(field non_bearer_transport)" == measured ]] || fail 'non-bearer transport is not measured'
[[ "$(field descriptor_barrier_causal)" == true ]] || fail 'descriptor causality is absent'
[[ "$(field material_threshold_measured)" == true ]] || fail 'material threshold is unmeasured'
[[ "$(field principal_distinct_uid)" == false ]] || fail 'local pod was laundered as distinct UID'
for boundary in same_uid_peer_isolation material_grant material_execution barrier_release launch_open exec_attached parity_open claim_ready; do
  [[ "$(field "$boundary")" == false ]] || fail "$boundary was promoted during freeze"
done

integration_commit="$(field integration_commit)"
freeze_commit="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${integration_commit}^{commit}" || fail 'integration commit is absent'
git -C "$ROOT_DIR" cat-file -e "${freeze_commit}^{commit}" || fail 'freeze gate commit is absent'
for pair in \
  broker_source_path:broker_source_sha256 \
  quorum_module_path:quorum_module_sha256 \
  barrier_source_path:barrier_source_sha256 \
  broker_build_script_path:broker_build_script_sha256 \
  barrier_build_script_path:barrier_build_script_sha256 \
  selftest_path:selftest_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$integration_commit:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the integration commit"
done
freeze_path="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$freeze_path")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze selftest drifted'
[[ "$(git -C "$ROOT_DIR" show "$freeze_commit:$freeze_path" | stream_hash)" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze selftest differs from its commit'

garden="$ROOT_DIR/$(field garden_path)"
authority="$ROOT_DIR/$(field authority_manifest_path)"
fixtures="$ROOT_DIR/$(field fixture_manifest_path)"
controller="$ROOT_DIR/$(field controller_manifest_path)"
resident="$ROOT_DIR/$(field resident_manifest_path)"
[[ "$(file_hash "$garden")" == "$(field garden_sha256)" ]] || fail 'Garden preregistration drifted'
[[ "$(file_hash "$authority")" == "$(field authority_manifest_sha256)" ]] || fail 'Sounio authority manifest drifted'
[[ "$(record_field "$authority" stage)" == SEMANTICS_FROZEN ]] || fail 'Sounio authority is not frozen'
[[ "$(record_field "$authority" language_role)" == SEMANTIC_AUTHORITY ]] || fail 'action 9030 lost semantic authority'
[[ "$(file_hash "$fixtures")" == "$(field fixture_manifest_sha256)" ]] || fail 'Sounio fixture manifest drifted'
[[ "$(record_field "$fixtures" producing_language)" == Sounio ]] || fail 'fixtures were not produced by Sounio'
[[ "$(file_hash "$controller")" == "$(field controller_manifest_sha256)" ]] || fail 'OCaml controller manifest drifted'
[[ "$(record_field "$controller" stage)" == EFFECT_PARITY_FROZEN ]] || fail 'OCaml controller is not frozen'
[[ "$(record_field "$controller" semantic_authority)" == Sounio ]] || fail 'controller lost its authority root'
[[ "$(file_hash "$resident")" == "$(field resident_manifest_sha256)" ]] || fail 'resident v4 manifest drifted'
[[ "$(record_field "$resident" runtime_frozen)" == true ]] || fail 'resident v4 is not frozen'

if grep -Eq 'DENY49[1-9]|DENY500|DENY501|ALLOW code=0 reason=allow|code=491|code=499' \
  "$ROOT_DIR/$(field quorum_module_path)"; then
  fail 'material quorum module encodes a Sounio expected result'
fi

cxx="$(field cxx_path)"
[[ "$(file_hash "$cxx")" == "$(field cxx_sha256)" ]] || fail 'C++ compiler drifted'
[[ "$($cxx --version | sed -n '1p')" == "$(field cxx_version)" ]] || fail 'C++ compiler version drifted'

for output in "$BROKER_ONE" "$BROKER_TWO"; do
  SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$output" \
    SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_CXX="$cxx" \
    bash "$ROOT_DIR/$(field broker_build_script_path)" >/dev/null
done
for output in "$BARRIER_ONE" "$BARRIER_TWO"; do
  SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_INTEGRATED_OUTPUT="$output" \
    SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_CXX="$cxx" \
    bash "$ROOT_DIR/$(field barrier_build_script_path)" >/dev/null
done
cmp "$BROKER_ONE" "$BROKER_TWO" || fail 'two broker rebuilds differ'
cmp "$BARRIER_ONE" "$BARRIER_TWO" || fail 'two integrated barrier rebuilds differ'
[[ "$(file_hash "$BROKER_ONE")" == "$(field broker_runtime_sha256)" ]] || fail 'broker runtime hash differs'
[[ "$(file_hash "$BARRIER_ONE")" == "$(field barrier_runtime_sha256)" ]] || fail 'barrier runtime hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_host_exec_quorum_selftest.sh' ]] || fail 'unexpected integrated gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'integrated command hash differs'
result="$(bash "$ROOT_DIR/$(field selftest_path)")"
[[ "$result" == "$(field result)" ]] || fail 'integrated causal gate result differs'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'integrated causal gate result hash differs'

legacy_result="$(bash "$ROOT_DIR/$(field legacy_selftest_path)")"
[[ "$legacy_result" == sounio-loom-kernel-principal-broker-selftest:\ PASS* ]] || fail 'legacy broker gate failed'
[[ "$legacy_result" == *'launch=closed recycle=closed'* &&
   "$legacy_result" == *'material_grant=false material_execution=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false' ]] ||
  fail 'legacy broker baseline opened during freeze'
[[ "$(printf '%s\n' "$legacy_result" | stream_hash)" == "$(field legacy_result_sha256)" ]] || fail 'legacy broker gate result hash differs'

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'ExecQuorum evidence hash drifted'
grep -Fxq "broker_runtime_sha256=$(field broker_runtime_sha256)" "$EVIDENCE" || fail 'evidence does not bind broker runtime'
grep -Fxq "barrier_runtime_sha256=$(field barrier_runtime_sha256)" "$EVIDENCE" || fail 'evidence does not bind barrier runtime'
grep -Fxq "result_sha256=$(field result_sha256)" "$EVIDENCE" || fail 'evidence does not bind integrated result'
grep -Fxq 'exact_write_sabotage=open' "$EVIDENCE" || fail 'evidence omits the causal control'
grep -Fxq 'principal_distinct_uid=false' "$EVIDENCE" || fail 'evidence launders pod identity'
grep -Fxq 'material_grant=false' "$EVIDENCE" || fail 'evidence overstates material grant'

manifest_hash="$(file_hash "$MANIFEST")"
printf 'sounio-loom-host-exec-quorum-freeze-selftest: PASS semantic_authority=Sounio controller=OCaml controller_role=EFFECT_PARITY material_layer=C++20+Linux material_role=MATERIAL_PARITY manifest_sha256=%s broker_runtime_sha256=%s barrier_runtime_sha256=%s deterministic_rebuilds=2+2 treatment=closed positive_semantics=ready positive_local=closed positive_local_reason=same-uid-principal exact_write_sabotage=open causal_matrix=frozen legacy_broker=green non_bearer_transport=measured same_uid_peer_isolation=false principal_distinct_uid=false descriptor_barrier_causal=true material_threshold_measured=true material_grant=false material_execution=false barrier_release=false launch_open=false exec_attached=false parity_open=false claim_ready=false\n' \
  "$manifest_hash" "$(field broker_runtime_sha256)" "$(field barrier_runtime_sha256)"
