#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-language-authority-v1-20260827.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-language-authority-freeze.XXXXXX")"
RUNTIME="$TEST_ROOT/sounio-language-authority"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-language-authority-freeze-selftest: FAIL: %s\n' "$*" >&2
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

[[ -f "$MANIFEST" ]] || fail "freeze manifest is missing"
[[ -f "$EVIDENCE" ]] || fail "freeze evidence is missing"
[[ "$(field schema)" == loom-language-authority-freeze-v1 ]] || fail "unknown manifest schema"
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail "manifest is not frozen"
[[ "$(field producing_language)" == Sounio ]] || fail "producer is not Sounio"
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail "producer role is not semantic authority"
[[ "$(field parity_open)" == false ]] || fail "freeze manifest opened parity"
[[ "$(field claim_ready)" == false ]] || fail "freeze manifest promoted a claim"

garden_commit="$(field garden_commit)"
executable_commit="$(field sounio_executable_commit)"
git -C "$ROOT_DIR" cat-file -e "${garden_commit}^{commit}" || fail "Garden commit is absent"
git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" || fail "Sounio executable commit is absent"
[[ "$(git -C "$ROOT_DIR" rev-parse "${executable_commit}^")" == "$garden_commit" ]] ||
  fail "Sounio executable commit does not immediately follow Garden"

source_path="$(field source_path)"
entrypoint_path="$(field entrypoint_path)"
[[ "$source_path" == stdlib/coordination/loom_language_authority.sio ]] || fail "unexpected source path"
[[ "$entrypoint_path" == tools/loom/language_authority_main.sio ]] || fail "unexpected entrypoint path"
[[ "$(file_hash "$ROOT_DIR/$source_path")" == "$(field source_sha256)" ]] || fail "working source drifted"
[[ "$(file_hash "$ROOT_DIR/$entrypoint_path")" == "$(field entrypoint_sha256)" ]] || fail "working entrypoint drifted"
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$source_path" | stream_hash)" == "$(field source_sha256)" ]] ||
  fail "frozen commit source hash differs"
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$entrypoint_path" | stream_hash)" == "$(field entrypoint_sha256)" ]] ||
  fail "frozen commit entrypoint hash differs"

bundle_hash="$(sed -n '1,$p' "$ROOT_DIR/$source_path" "$ROOT_DIR/$entrypoint_path" | stream_hash)"
[[ "$bundle_hash" == "$(field semantics_sha256)" ]] || fail "semantic bundle hash differs"

wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
build_script_path="$(field build_script_path)"
[[ "$wrapper_path" == bin/souc ]] || fail "unexpected compiler wrapper"
[[ "$compiler_path" == bin/souc-lean-single-x86_64 ]] || fail "unexpected compiler binary"
[[ "$build_script_path" == scripts/dev/build_sounio_loom_language_authority.sh ]] || fail "unexpected build script"
[[ "$(file_hash "$ROOT_DIR/$wrapper_path")" == "$(field toolchain_wrapper_sha256)" ]] || fail "compiler wrapper drifted"
[[ "$(file_hash "$ROOT_DIR/$compiler_path")" == "$(field toolchain_compiler_sha256)" ]] || fail "compiler binary drifted"
[[ "$(file_hash "$ROOT_DIR/$build_script_path")" == "$(field build_script_sha256)" ]] || fail "build command implementation drifted"

toolchain_hash="$({
  printf '%s\n' "engine=$(field toolchain_engine)"
  printf '%s\n' "wrapper=$wrapper_path"
  printf '%s\n' "wrapper_sha256=$(field toolchain_wrapper_sha256)"
  printf '%s\n' "compiler=$compiler_path"
  printf '%s\n' "compiler_sha256=$(field toolchain_compiler_sha256)"
} | stream_hash)"
[[ "$toolchain_hash" == "$(field toolchain_record_sha256)" ]] || fail "toolchain record hash differs"

hardware_hash="$({
  printf '%s\n' "kernel=$(field hardware_kernel)"
  printf '%s\n' "architecture=$(field hardware_architecture)"
  printf '%s\n' "logical_cpus=$(field hardware_logical_cpus)"
  printf '%s\n' "cpu_model=$(field hardware_cpu_model)"
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail "hardware record hash differs"

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_language_authority_selftest.sh' ]] || fail "unexpected gate command"
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail "command hash differs"
gate_script="$ROOT_DIR/scripts/ci/sounio_loom_language_authority_selftest.sh"
[[ "$(file_hash "$gate_script")" == "$(field command_script_sha256)" ]] || fail "gate command implementation drifted"

SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
[[ "$(file_hash "$RUNTIME")" == "$(field executable_sha256)" ]] || fail "rebuilt executable hash differs"

gate_result="$(bash "$gate_script")"
[[ "$gate_result" == "$(field result)" ]] || fail "gate result differs from frozen Sounio result"
[[ "$(printf '%s\n' "$gate_result" | stream_hash)" == "$(field result_sha256)" ]] || fail "result hash differs"

source_u32="$(field source_sha256_u32)"
semantics_u32="$(field semantics_sha256_u32)"
toolchain_u32="$(field toolchain_record_sha256_u32)"
hardware_u32="$(field hardware_record_sha256_u32)"
command_u32="$(field command_sha256_u32)"
result_u32="$(field result_sha256_u32)"
source_u32="${source_u32//,/ }"
semantics_u32="${semantics_u32//,/ }"
toolchain_u32="${toolchain_u32//,/ }"
hardware_u32="${hardware_u32//,/ }"
command_u32="${command_u32//,/ }"
result_u32="${result_u32//,/ }"
zero='0 0 0 0 0 0 0 0'
freeze_frame="9020 2 3 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $source_u32 $semantics_u32 $zero $toolchain_u32 $hardware_u32 $command_u32 $result_u32 $zero"
freeze_decision="$(printf '%s\n' "$freeze_frame" | "$RUNTIME")"
[[ "$freeze_decision" == "$(field freeze_decision)" ]] || fail "Sounio freeze decision differs"
[[ "$(printf '%s\n' "$freeze_decision" | stream_hash)" == "$(field freeze_decision_sha256)" ]] ||
  fail "Sounio freeze decision hash differs"

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "  sha256 $manifest_hash" "$EVIDENCE" || fail "evidence does not bind the manifest hash"

printf '%s\n' \
  "sounio-loom-language-authority-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN semantics_sha256=$(field semantics_sha256) manifest_sha256=$manifest_hash parity_open=false claim_ready=false"
