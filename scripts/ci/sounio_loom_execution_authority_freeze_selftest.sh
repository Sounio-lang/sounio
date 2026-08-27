#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/execution_authority.freeze.v2"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-execution-authority-v2-20260827.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-execution-authority-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/execution-authority-one"
RUNTIME_TWO="$TEST_ROOT/execution-authority-two"
LANGUAGE_RUNTIME="$TEST_ROOT/language-authority"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-execution-authority-freeze-selftest: FAIL: %s\n' "$*" >&2
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

hash_u32_csv() {
  local hash="$1" values='' offset chunk value
  for offset in 0 8 16 24 32 40 48 56; do
    chunk="${hash:$offset:8}"
    value="$((16#$chunk))"
    if [[ -n "$values" ]]; then
      values="$values,$value"
    else
      values="$value"
    fi
  done
  printf '%s' "$values"
}

[[ -f "$MANIFEST" ]] || fail 'freeze manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'freeze evidence is missing'
[[ "$(field schema)" == loom-execution-authority-freeze-v2 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'manifest is not frozen'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field parity_open)" == false ]] || fail 'freeze manifest opened parity'
[[ "$(field claim_ready)" == false ]] || fail 'freeze manifest promoted a claim'
for surface in exec_attached child_exec_attached commit_attached ci_attached; do
  [[ "$(field "$surface")" == false ]] || fail "$surface was promoted during freeze"
done

garden_commit="$(field garden_commit)"
executable_commit="$(field sounio_executable_commit)"
git -C "$ROOT_DIR" cat-file -e "${garden_commit}^{commit}" || fail 'Garden commit is absent'
git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" || fail 'Sounio executable commit is absent'
[[ "$(git -C "$ROOT_DIR" rev-parse "${executable_commit}^")" == "$garden_commit" ]] ||
  fail 'Sounio executable commit does not immediately follow Garden'

source_path="$(field source_path)"
entrypoint_path="$(field entrypoint_path)"
source_hash="$(field source_sha256)"
entrypoint_hash="$(field entrypoint_sha256)"
[[ "$(file_hash "$ROOT_DIR/$source_path")" == "$source_hash" ]] || fail 'working source drifted'
[[ "$(file_hash "$ROOT_DIR/$entrypoint_path")" == "$entrypoint_hash" ]] || fail 'working entrypoint drifted'
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$source_path" | stream_hash)" == "$source_hash" ]] ||
  fail 'frozen commit source hash differs'
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$entrypoint_path" | stream_hash)" == "$entrypoint_hash" ]] ||
  fail 'frozen commit entrypoint hash differs'
[[ "$(hash_u32_csv "$source_hash")" == "$(field source_sha256_u32)" ]] || fail 'source limbs differ'

semantics_hash="$(sed -n '1,$p' "$ROOT_DIR/$source_path" "$ROOT_DIR/$entrypoint_path" | stream_hash)"
[[ "$semantics_hash" == "$(field semantics_sha256)" ]] || fail 'semantic bundle hash differs'
[[ "$(hash_u32_csv "$semantics_hash")" == "$(field semantics_sha256_u32)" ]] || fail 'semantics limbs differ'

wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
build_script_path="$(field build_script_path)"
gate_script="$ROOT_DIR/scripts/ci/sounio_loom_execution_authority_selftest.sh"
freeze_script_path="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$wrapper_path")" == "$(field toolchain_wrapper_sha256)" ]] || fail 'compiler wrapper drifted'
[[ "$(file_hash "$ROOT_DIR/$compiler_path")" == "$(field toolchain_compiler_sha256)" ]] || fail 'compiler binary drifted'
[[ "$(file_hash "$ROOT_DIR/$build_script_path")" == "$(field build_script_sha256)" ]] || fail 'build script drifted'
[[ "$(file_hash "$gate_script")" == "$(field command_script_sha256)" ]] || fail 'gate script drifted'
[[ "$(file_hash "$ROOT_DIR/$freeze_script_path")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze selftest drifted'

toolchain_hash="$({
  printf '%s\n' "engine=$(field toolchain_engine)"
  printf '%s\n' "wrapper=$wrapper_path"
  printf '%s\n' "wrapper_sha256=$(field toolchain_wrapper_sha256)"
  printf '%s\n' "compiler=$compiler_path"
  printf '%s\n' "compiler_sha256=$(field toolchain_compiler_sha256)"
} | stream_hash)"
[[ "$toolchain_hash" == "$(field toolchain_record_sha256)" ]] || fail 'toolchain record hash differs'
[[ "$(hash_u32_csv "$toolchain_hash")" == "$(field toolchain_record_sha256_u32)" ]] || fail 'toolchain limbs differ'

hardware_hash="$({
  printf '%s\n' "kernel=$(field hardware_kernel)"
  printf '%s\n' "architecture=$(field hardware_architecture)"
  printf '%s\n' "logical_cpus=$(field hardware_logical_cpus)"
  printf '%s\n' "cpu_model=$(field hardware_cpu_model)"
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail 'hardware record hash differs'
[[ "$(hash_u32_csv "$hardware_hash")" == "$(field hardware_record_sha256_u32)" ]] || fail 'hardware limbs differ'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_execution_authority_selftest.sh' ]] || fail 'unexpected gate command'
command_hash="$(printf '%s\n' "$command" | stream_hash)"
[[ "$command_hash" == "$(field command_sha256)" ]] || fail 'command hash differs'
[[ "$(hash_u32_csv "$command_hash")" == "$(field command_sha256_u32)" ]] || fail 'command limbs differ'

SOUNIO_LOOM_EXECUTION_AUTHORITY_OUTPUT="$RUNTIME_ONE" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_EXECUTION_AUTHORITY_OUTPUT="$RUNTIME_TWO" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio rebuilds differ'
[[ "$(file_hash "$RUNTIME_ONE")" == "$(field executable_sha256)" ]] || fail 'rebuilt executable hash differs'

gate_result="$(bash "$gate_script")"
[[ "$gate_result" == "$(field result)" ]] || fail 'gate result differs from frozen Sounio result'
result_hash="$(printf '%s\n' "$gate_result" | stream_hash)"
[[ "$result_hash" == "$(field result_sha256)" ]] || fail 'gate result hash differs'
[[ "$(hash_u32_csv "$result_hash")" == "$(field result_sha256_u32)" ]] || fail 'result limbs differ'

SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$LANGUAGE_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_language_authority.sh" >/dev/null
[[ "$(file_hash "$LANGUAGE_RUNTIME")" == "$(field freeze_authority_runtime_sha256)" ]] ||
  fail 'language-authority runtime drifted'
language_parent_semantics="$(grep -m1 '^semantics_sha256=' \
  "$ROOT_DIR/$(field freeze_authority_manifest)" | cut -d= -f2)"
[[ "$(field freeze_authority_semantics_sha256)" == "$language_parent_semantics" ]] ||
  fail 'language-authority parent semantics drifted'

source_u32="$(field source_sha256_u32)"; source_u32="${source_u32//,/ }"
semantics_u32="$(field semantics_sha256_u32)"; semantics_u32="${semantics_u32//,/ }"
toolchain_u32="$(field toolchain_record_sha256_u32)"; toolchain_u32="${toolchain_u32//,/ }"
hardware_u32="$(field hardware_record_sha256_u32)"; hardware_u32="${hardware_u32//,/ }"
command_u32="$(field command_sha256_u32)"; command_u32="${command_u32//,/ }"
result_u32="$(field result_sha256_u32)"; result_u32="${result_u32//,/ }"
zero='0 0 0 0 0 0 0 0'

freeze_frame="9020 2 3 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $source_u32 $semantics_u32 $zero $toolchain_u32 $hardware_u32 $command_u32 $result_u32 $zero"
freeze_decision="$(printf '%s\n' "$freeze_frame" | "$LANGUAGE_RUNTIME")"
[[ "$freeze_decision" == "$(field freeze_decision)" ]] || fail 'Sounio freeze decision differs'
[[ "$(printf '%s\n' "$freeze_decision" | stream_hash)" == "$(field freeze_decision_sha256)" ]] ||
  fail 'freeze decision hash differs'

consumer_frame="9021 3 4 4 9 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $source_u32 $semantics_u32 $semantics_u32 $toolchain_u32 $hardware_u32 $command_u32 $result_u32 $zero"
consumer_decision="$(printf '%s\n' "$consumer_frame" | "$RUNTIME_ONE")"
[[ "$consumer_decision" == "$(field consumer_decision)" ]] || fail 'Sounio consumer decision differs'
[[ "$(printf '%s\n' "$consumer_decision" | stream_hash)" == "$(field consumer_decision_sha256)" ]] ||
  fail 'consumer decision hash differs'

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "sha256 $manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the manifest hash'

printf '%s\n' \
  "sounio-loom-execution-authority-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN semantics_sha256=$semantics_hash manifest_sha256=$manifest_hash rebuilds=2 causal_sabotage=PASS exec_attached=false child_exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false"
