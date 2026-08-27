#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/lane_health.ocaml.v1"
FREEZE="$ROOT_DIR/tools/loom/lane_health.freeze.v1"
AUTHORITY_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-lane-health-ocaml-v1-20260827.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-lane-health-ocaml-receipt.XXXXXX")"
TOOLCHAIN_ROOT="$TEST_ROOT/toolchain"
AUTHORITY_TOOLCHAIN_ROOT="$TEST_ROOT/authority-toolchain"
SOUNIO_RUNTIME_A="$TEST_ROOT/sounio-parity-a"
SOUNIO_RUNTIME_B="$TEST_ROOT/sounio-parity-b"
AUTHORITY_RUNTIME="$TEST_ROOT/sounio-language-authority"
OCAML_RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-lane-health-ocaml-receipt-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_field() {
  local manifest="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$manifest" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times in $manifest"
  line="$(grep -m1 "^${key}=" "$manifest")"
  printf '%s' "${line#*=}"
}

field() { manifest_field "$MANIFEST" "$1"; }
freeze_field() { manifest_field "$FREEZE" "$1"; }
authority_field() { manifest_field "$AUTHORITY_MANIFEST" "$1"; }

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

[[ -f "$MANIFEST" && -f "$FREEZE" && -f "$AUTHORITY_MANIFEST" ]] || fail 'required manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'receipt evidence is missing'
[[ "$(field schema)" == loom-lane-health-ocaml-realization-v1 ]] || fail 'unknown receipt schema'
[[ "$(field producing_language)" == OCaml ]] || fail 'producer is not OCaml'
[[ "$(field language_role)" == OPERATIONAL_REALIZATION ]] || fail 'OCaml was assigned an authority role'
[[ "$(field operational_realization_admitted)" == true ]] || fail 'operational realization is not admitted'
[[ "$(field formal_parity_open)" == false ]] || fail 'receipt claimed formal parity'
[[ "$(field claim_ready)" == false ]] || fail 'receipt claimed product readiness'
[[ "$(file_hash "$FREEZE")" == "$(field frozen_parent_manifest_sha256)" ]] || fail 'frozen parent manifest drifted'
[[ "$(field frozen_semantics_sha256)" == "$(freeze_field semantics_sha256)" ]] || fail 'parent semantics hash differs'

freeze_commit="$(field freeze_receipt_commit)"
realization_commit="$(field realization_commit)"
[[ "$(git -C "$ROOT_DIR" rev-parse "${realization_commit}^")" == "$freeze_commit" ]] ||
  fail 'OCaml realization did not immediately follow the freeze receipt'

module_path="$(field ocaml_module_path)"
[[ "$(file_hash "$ROOT_DIR/$module_path")" == "$(field ocaml_module_sha256)" ]] || fail 'OCaml classifier drifted'
[[ "$(git -C "$ROOT_DIR" show "$realization_commit:$module_path" | stream_hash)" == "$(field ocaml_module_sha256)" ]] ||
  fail 'realization commit classifier hash differs'
for pair in \
  "loom_dispatch_path loom_dispatch_sha256" \
  "dune_file_path dune_file_sha256" \
  "sounio_parity_adapter_path sounio_parity_adapter_sha256" \
  "sounio_sha256_module_path sounio_sha256_module_sha256" \
  "parity_builder_path parity_builder_sha256" \
  "parity_gate_path parity_gate_sha256"; do
  set -- $pair
  path="$(field "$1")"
  expected="$(field "$2")"
  [[ "$(git -C "$ROOT_DIR" show "$realization_commit:$path" | stream_hash)" == "$expected" ]] ||
    fail "realization commit hash differs for $path"
done

toolchain_hash="$({
  printf '%s\n' 'language=OCaml'
  printf '%s\n' 'role=OPERATIONAL_REALIZATION'
  printf '%s\n' "ocamlopt_version=$(field toolchain_ocamlopt_version)"
  printf '%s\n' "ocamlopt_sha256=$(field toolchain_ocamlopt_sha256)"
  printf '%s\n' "dune_version=$(field toolchain_dune_version)"
  printf '%s\n' "dune_sha256=$(field toolchain_dune_sha256)"
  printf '%s\n' "cryptokit_version=$(field toolchain_cryptokit_version)"
} | stream_hash)"
[[ "$toolchain_hash" == "$(field toolchain_record_sha256)" ]] || fail 'OCaml toolchain record hash differs'

hardware_hash="$({
  printf '%s\n' "kernel=$(field hardware_kernel)"
  printf '%s\n' "architecture=$(field hardware_architecture)"
  printf '%s\n' "logical_cpus=$(field hardware_logical_cpus)"
  printf '%s\n' "cpu_model=$(field hardware_cpu_model)"
} | stream_hash)"
[[ "$hardware_hash" == "$(field hardware_record_sha256)" ]] || fail 'hardware record hash differs'

bash "$ROOT_DIR/scripts/ci/sounio_loom_lane_health_freeze_selftest.sh" >/dev/null
executable_commit="$(freeze_field sounio_executable_commit)"
wrapper_path="$(freeze_field toolchain_wrapper_path)"
compiler_path="$(freeze_field toolchain_compiler_path)"
mkdir -p "$TOOLCHAIN_ROOT"
git -C "$ROOT_DIR" archive "$executable_commit" "$wrapper_path" "$compiler_path" |
  tar -x -C "$TOOLCHAIN_ROOT"
SOUNIO_LOOM_LANE_HEALTH_PARITY_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_LANE_HEALTH_PARITY_OUTPUT="$SOUNIO_RUNTIME_A" \
  bash "$ROOT_DIR/$(field parity_builder_path)" >/dev/null
SOUNIO_LOOM_LANE_HEALTH_PARITY_SOUC="$TOOLCHAIN_ROOT/$wrapper_path" \
  SOUNIO_LOOM_LANE_HEALTH_PARITY_OUTPUT="$SOUNIO_RUNTIME_B" \
  bash "$ROOT_DIR/$(field parity_builder_path)" >/dev/null
[[ "$(file_hash "$SOUNIO_RUNTIME_A")" == "$(field sounio_parity_runtime_sha256)" ]] || fail 'first Sounio parity runtime differs'
[[ "$(file_hash "$SOUNIO_RUNTIME_B")" == "$(field sounio_parity_runtime_sha256)" ]] || fail 'second Sounio parity runtime differs'

dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
sounio_result="$($SOUNIO_RUNTIME_A)"
ocaml_result="$($OCAML_RUNTIME lane-health-parity)"
[[ "$(printf '%s\n' "$sounio_result" | stream_hash)" == "$(field sounio_result_sha256)" ]] || fail 'Sounio result hash differs'
[[ "$(printf '%s\n' "$ocaml_result" | stream_hash)" == "$(field ocaml_result_sha256)" ]] || fail 'OCaml result hash differs'
[[ "${sounio_result#SOUNIO_LANE_HEALTH_PARITY }" == "${ocaml_result#OCAML_LANE_HEALTH_PARITY }" ]] || fail 'current Sounio and OCaml outputs diverged'

command="$(field command)"
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
gate_result="$(bash "$ROOT_DIR/$(field parity_gate_path)")"
[[ "$gate_result" == "$(field result)" ]] || fail 'parity gate result differs'
[[ "$(printf '%s\n' "$gate_result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'parity gate result hash differs'

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

source_u32="$(field ocaml_module_sha256_u32)"
semantics_u32="$(field frozen_semantics_sha256_u32)"
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
authority_frame="9020 3 12 9 8 1 0 0 0 0 0 0 0 0 0 0 0 0 $source_u32 $semantics_u32 $semantics_u32 $toolchain_u32 $hardware_u32 $command_u32 $result_u32 $zero"
authority_decision="$(printf '%s\n' "$authority_frame" | "$AUTHORITY_RUNTIME")"
[[ "$authority_decision" == "$(field authority_decision)" ]] || fail 'Sounio authority decision differs'

wrong_parent='2 2 2 2 2 2 2 2'
wrong_parent_frame="9020 3 12 9 8 1 0 0 0 0 0 0 0 0 0 0 0 0 $source_u32 $semantics_u32 $wrong_parent $toolchain_u32 $hardware_u32 $command_u32 $result_u32 $zero"
set +e
wrong_parent_decision="$(printf '%s\n' "$wrong_parent_frame" | "$AUTHORITY_RUNTIME")"
wrong_parent_rc=$?
set -e
[[ "$wrong_parent_rc" -eq 117 && "$wrong_parent_decision" == *'reason=parent-semantics-hash-mismatch'* ]] ||
  fail "wrong-parent OCaml realization was not refused: rc=$wrong_parent_rc decision=$wrong_parent_decision"

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "  sha256 $manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the receipt manifest'
printf '%s\n' \
  "sounio-loom-lane-health-ocaml-receipt-selftest: PASS authority=Sounio realization=OCaml role=OPERATIONAL_REALIZATION domain=$(field domain_cases) digest_sha256=$(field decision_stream_sha256) wrong_parent=refused formal_parity_open=false claim_ready=false manifest_sha256=$manifest_hash"
