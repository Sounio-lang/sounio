#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/lane_health.freeze.v1"
AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-lane-health-v1-20260827.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-lane-health-freeze.XXXXXX")"
RUNTIME_A="$TEST_ROOT/sounio-lane-health-a"
RUNTIME_B="$TEST_ROOT/sounio-lane-health-b"
AUTHORITY_RUNTIME="$TEST_ROOT/sounio-language-authority"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"
AUTHORITY_TOOLCHAIN_ROOT="$TEST_ROOT/authority-toolchain"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-lane-health-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_field() {
  local manifest="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$manifest" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times in $manifest"
  line="$(grep -m1 "^${key}=" "$manifest")"
  printf '%s' "${line#*=}"
}

field() {
  manifest_field "$MANIFEST" "$1"
}

authority_field() {
  manifest_field "$AUTHORITY_MANIFEST" "$1"
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

[[ -f "$MANIFEST" ]] || fail 'freeze manifest is missing'
[[ -f "$AUTHORITY_MANIFEST" ]] || fail 'language-authority manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'freeze evidence is missing'
[[ "$(field schema)" == loom-lane-health-freeze-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == SEMANTICS_FROZEN ]] || fail 'manifest is not frozen'
[[ "$(field producing_language)" == Sounio ]] || fail 'producer is not Sounio'
[[ "$(field language_role)" == SEMANTIC_AUTHORITY ]] || fail 'producer role is not semantic authority'
[[ "$(field parity_open)" == false ]] || fail 'freeze manifest opened parity'
[[ "$(field claim_ready)" == false ]] || fail 'freeze manifest promoted a claim'
[[ "$(file_hash "$AUTHORITY_MANIFEST")" == "$(field language_authority_manifest_sha256)" ]] ||
  fail 'language-authority parent manifest drifted'

garden_commit="$(field garden_commit)"
executable_commit="$(field sounio_executable_commit)"
git -C "$ROOT_DIR" cat-file -e "${garden_commit}^{commit}" || fail 'Garden commit is absent'
git -C "$ROOT_DIR" cat-file -e "${executable_commit}^{commit}" || fail 'Sounio executable commit is absent'
[[ "$(git -C "$ROOT_DIR" rev-parse "${executable_commit}^")" == "$garden_commit" ]] ||
  fail 'Sounio executable commit does not immediately follow Garden'

source_path="$(field source_path)"
entrypoint_path="$(field entrypoint_path)"
[[ "$source_path" == stdlib/coordination/loom_lane_health.sio ]] || fail 'unexpected source path'
[[ "$entrypoint_path" == tools/loom/lane_health_main.sio ]] || fail 'unexpected entrypoint path'
[[ "$(file_hash "$ROOT_DIR/$source_path")" == "$(field source_sha256)" ]] || fail 'working source drifted'
[[ "$(file_hash "$ROOT_DIR/$entrypoint_path")" == "$(field entrypoint_sha256)" ]] || fail 'working entrypoint drifted'
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$source_path" | stream_hash)" == "$(field source_sha256)" ]] ||
  fail 'frozen commit source hash differs'
[[ "$(git -C "$ROOT_DIR" show "$executable_commit:$entrypoint_path" | stream_hash)" == "$(field entrypoint_sha256)" ]] ||
  fail 'frozen commit entrypoint hash differs'
bundle_hash="$(sed -n '1,$p' "$ROOT_DIR/$source_path" "$ROOT_DIR/$entrypoint_path" | stream_hash)"
[[ "$bundle_hash" == "$(field semantics_sha256)" ]] || fail 'semantic bundle hash differs'

wrapper_path="$(field toolchain_wrapper_path)"
compiler_path="$(field toolchain_compiler_path)"
build_script_path="$(field build_script_path)"
[[ "$wrapper_path" == bin/souc ]] || fail 'unexpected compiler wrapper'
[[ "$compiler_path" == bin/souc-lean-single-x86_64 ]] || fail 'unexpected compiler binary'
[[ "$build_script_path" == scripts/dev/build_sounio_loom_lane_health.sh ]] || fail 'unexpected build script'
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$executable_commit" "$wrapper_path" "$compiler_path" |
  tar -x -C "$TOOLCHAIN_ROOT"
[[ "$(file_hash "$TOOLCHAIN_ROOT/$wrapper_path")" == "$(field toolchain_wrapper_sha256)" ]] ||
  fail 'frozen compiler wrapper drifted'
[[ "$(file_hash "$TOOLCHAIN_ROOT/$compiler_path")" == "$(field toolchain_compiler_sha256)" ]] ||
  fail 'frozen compiler binary drifted'
[[ "$(file_hash "$ROOT_DIR/$build_script_path")" == "$(field build_script_sha256)" ]] ||
  fail 'build command implementation drifted'

toolchain_hash="$({
  printf '%s\n' "engine=$(field toolchain_engine)"
  printf '%s\n' "wrapper=$wrapper_path"
  printf '%s\n' "wrapper_sha256=$(field toolchain_wrapper_sha256)"
  printf '%s\n' "compiler=$compiler_path"
  printf '%s\n' "compiler_sha256=$(field toolchain_compiler_sha256)"
} | stream_hash)"
[[ "$toolchain_hash" == "$(field toolchain_record_sha256)" ]] || fail 'toolchain record hash differs'

hardware_hash="$({
  printf '%s\n' "kernel=$(field hardware_kernel)"
  printf '%s\n' "architecture=$(field hardware_architecture)"
  printf '%s\n' "logical_cpus=$(field hardware_logical_cpus)"
  printf '%s\n' "cpu_model=$(field hardware_cpu_model)"
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail 'hardware record hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_lane_health_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
gate_script="$ROOT_DIR/scripts/ci/sounio_loom_lane_health_selftest.sh"
[[ "$(file_hash "$gate_script")" == "$(field command_script_sha256)" ]] || fail 'gate command implementation drifted'

SOUNIO_LOOM_LANE_HEALTH_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_LANE_HEALTH_OUTPUT="$RUNTIME_A" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
SOUNIO_LOOM_LANE_HEALTH_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_LANE_HEALTH_OUTPUT="$RUNTIME_B" \
  bash "$ROOT_DIR/$build_script_path" >/dev/null
[[ "$(file_hash "$RUNTIME_A")" == "$(field executable_sha256)" ]] || fail 'first executable hash differs'
[[ "$(file_hash "$RUNTIME_B")" == "$(field executable_sha256)" ]] || fail 'second executable hash differs'

gate_result="$(bash "$gate_script")"
[[ "$gate_result" == "$(field result)" ]] || fail 'gate result differs from frozen Sounio result'
[[ "$(printf '%s\n' "$gate_result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'result hash differs'

# Reconstruct the already-frozen language authority before asking it to admit
# this freeze. The live checkout compiler is not an approval oracle.
bash "$ROOT_DIR/scripts/ci/sounio_loom_language_authority_freeze_selftest.sh" >/dev/null
authority_commit="$(authority_field sounio_executable_commit)"
authority_wrapper="$(authority_field toolchain_wrapper_path)"
authority_compiler="$(authority_field toolchain_compiler_path)"
mkdir -p "$AUTHORITY_TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$authority_commit" "$authority_wrapper" "$authority_compiler" |
  tar -x -C "$AUTHORITY_TOOLCHAIN_ROOT"
SOUNIO_LOOM_LANGUAGE_AUTHORITY_SOUC="$AUTHORITY_TOOLCHAIN_ROOT/$authority_wrapper" \
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$AUTHORITY_RUNTIME" \
  bash "$ROOT_DIR/$(authority_field build_script_path)" >/dev/null

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
freeze_decision="$(printf '%s\n' "$freeze_frame" | "$AUTHORITY_RUNTIME")"
[[ "$freeze_decision" == "$(field freeze_decision)" ]] || fail 'Sounio freeze decision differs'
[[ "$(printf '%s\n' "$freeze_decision" | stream_hash)" == "$(field freeze_decision_sha256)" ]] ||
  fail 'Sounio freeze decision hash differs'

missing_source_frame="9020 2 3 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 $zero $semantics_u32 $zero $toolchain_u32 $hardware_u32 $command_u32 $result_u32 $zero"
set +e
missing_source_decision="$(printf '%s\n' "$missing_source_frame" | "$AUTHORITY_RUNTIME")"
missing_source_rc=$?
set -e
[[ "$missing_source_rc" -eq 115 && "$missing_source_decision" == *'reason=sounio-source-hash-missing'* ]] ||
  fail "source-less freeze was not refused: rc=$missing_source_rc decision=$missing_source_decision"

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "  sha256 $manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the manifest hash'

printf '%s\n' \
  "sounio-loom-lane-health-freeze-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY stage=SEMANTICS_FROZEN semantics_sha256=$(field semantics_sha256) manifest_sha256=$manifest_hash parity_open=false claim_ready=false"
