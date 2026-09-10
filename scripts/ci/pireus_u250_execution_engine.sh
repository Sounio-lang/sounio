#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_U250_EXECUTION_ENGINE_PARITY_RECEIPT_V1.md'
PARENT_REL='stdlib/hardware/pireus/u250_material_ingestion.sio'
PARENT_FREEZE_REL='tools/pireus/u250_material_ingestion.freeze.v1'
RAW_REL='docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt'
RECEIPT_REL='docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt'
TAMPERED_REL='tests/fixtures/pireus_u250_material_tampered_v1.txt'
MODULE_REL='stdlib/hardware/pireus/u250_execution_engine.sio'
EXAMPLE_REL='examples/pireus_u250_execution_engine.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_u250_execution_engine.sio'
SUPERSEDED_FREEZE_REL='tools/pireus/u250_execution_engine.freeze.v0'
FREEZE_REL='tools/pireus/u250_execution_engine.freeze.v1'
EVIDENCE_REL='tools/pireus/evidence/u250_execution_engine_v1.txt'

GARDEN="${ROOT}/${GARDEN_REL}"
PARENT="${ROOT}/${PARENT_REL}"
PARENT_FREEZE="${ROOT}/${PARENT_FREEZE_REL}"
RAW="${ROOT}/${RAW_REL}"
RECEIPT="${ROOT}/${RECEIPT_REL}"
TAMPERED="${ROOT}/${TAMPERED_REL}"
MODULE="${ROOT}/${MODULE_REL}"
EXAMPLE="${ROOT}/${EXAMPLE_REL}"
TEST="${ROOT}/${TEST_REL}"
SUPERSEDED_FREEZE="${ROOT}/${SUPERSEDED_FREEZE_REL}"
FREEZE="${ROOT}/${FREEZE_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
COMPILER="${ROOT}/bin/souc-lean-single-x86_64"

PARENT_MATERIAL_SEAL_COMMIT='da6a26d6cd76cec3011dbe3ad43bfe606d769050'
GARDEN_COMMIT='ade650349ef558bebd56572a4e61467dec5f50f1'
EXECUTABLE_COMMIT='93532b221cdb07caf4e1a245fae19f39c8b33df9'

GARDEN_SHA256='550127f59c59418eb2422a12270fa20c59d9771ac1227a6260e15176448b65d8'
PARENT_SHA256='dd24f9da944ecf5427491c5040442bb4f5fd1bd21a3c2394cbcdd585bc2469c2'
PARENT_FREEZE_SHA256='c4e2a0e0c1a4582f1192c185dc3d08ef837e3be19ac5ba982fa8a3327924f7d6'
PARENT_SEMANTICS_SHA256='536312cdd0d75fca14ae38d3322ceec2ce931d16853a0842c257e45f087a6794'
RAW_SHA256='6bea3b962c519dfe9a9878c008a6300b67b920f0a2b51ba9d89dbf180661e7df'
RECEIPT_SHA256='9889567b684fcc0213ed38a44041e8475c4c9a71722b7baa1c6c064e1f1d0d7a'
TAMPERED_SHA256='711c21a8b60e9c2717ca819b847b41779eafaab0d8f96924122146b76561164f'
MODULE_SHA256='2b39ae0d92d18fdf7da966264f44159a8a1ecfedd0ae7cb09e03333f4b5bebbd'
EXAMPLE_SHA256='00e843b1eb3990e05c3bf12521ea678456182fb00a1aec03ca0f6ba018a0e9eb'
TEST_SHA256='06ea37958e27628cda9a7027d2893da4593b5733870930d590a0dacfb3e42fa7'
SUPERSEDED_FREEZE_SHA256='4872efaa80f54aee282ec1e975586d487756025da08b062dcc6ed8795af2d829'
FREEZE_SHA256='b1ac42ba32baf967481c9eed888d8d3cb776a702a3eaf613dbf19af2e4994aab'
EVIDENCE_SHA256='c710dca7f089903ac11eb41a73a10d12093b845a1d68462db93be4463ec9350b'
SOURCE_MANIFEST_SHA256='a1785dfca0bf5ba66b742ad3aba85139a09f6fa3df9a7d217711562c45af3737'
SEMANTICS_SHA256='93c309d0c381464cc5d3e411a7227dd0b25174eb9e913c3c2fbaee7097d2c218'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
HARDWARE_SHA256='46262c5d0fc8df5734998677c2ad063c686fa2d1120fb8dc18dc5b382c7c4805'
COMMAND_SHA256='308b3d8f071d2e507c2dab88913051cd6e41afef9c3365bf53fd1dde09b5c602'
RESULT_SHA256='cf5c76e84a7ce2566e2b1ad7ad381a0f161f1e651b9090d5a8629b6bfb266262'
EXAMPLE_OUTPUT_SHA256='3da0c4803a2e4f60839a85b3b06ef25e2c7923349d5a0601ff7357addf9281fe'
TEST_OUTPUT_SHA256='a6c8f5c5e3fbaf352a969fe02215fbbacdb2df1776d92452e76ab7223272112c'
PREEXEC_FRAME_SHA256='af4b4e598008de76fd504771555b364507e08c536e4b6a3ae91b4d634adac8d9'
FREEZE_FRAME_SHA256='c57959593fe1da440bac85f01a0b1949b1b274721c464dd7ec5b2d9d6f0a832f'
PYTHON_FRAME_SHA256='75d4f9cd9fc17959d6b46009de668895e743afa681f62757415f627c7c76181d'
RUST_FRAME_SHA256='86c2da0d19772f1162628620d527e3e3a2cf01b341548f1b063b8fbdeb6737d2'
LLM_FRAME_SHA256='0abe8e7ef784969e019146f2ecc20435c218e2531b80567c96685d74f3c702fa'
CPP_FRAME_SHA256='a70a48c05e209e2209cb8e0ae945599d80c7c5a798a260e906f8e217e6ecbba7'
POLICY_MISSING_FRAME_SHA256='58ffa769ee0e3a8a4560f86e18fc01ae1fbc08a9efed7ae2830c294d536eb84e'
POLICY_TIMEOUT_FRAME_SHA256='ee77e0aee21a0014b858b6dffc47434b1c2724ba95072092077727bea49571a4'
PYTHON_TOOLCHAIN_SHA256='5c8cfd947420cd48743adb75469089a210d7782421a4e9e46bfc4c40021fb7cf'
PYTHON_COMMAND_SHA256='39d4704375b8db3fa1a86377da138b87363d09893df457b49e5f6f8a78809551'
RUST_TOOLCHAIN_SHA256='ae2ed5be700fb051fe99519bd8a9376ad8e4d5091c394ccbb9d8b57c36b7cd11'
RUST_COMMAND_SHA256='d38eb598379d9d455823308a7b90d494bbd7723845bb9eefc71bb4d1dd2c2059'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus U250 execution engine: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() { sha256sum "$1" | cut -d' ' -f1; }
sha_text() { printf '%s\n' "$1" | sha256sum | cut -d' ' -f1; }

require_hash() {
  local path="$1" expected="$2"
  [[ -f "${path}" ]] || fail "missing file: ${path}"
  [[ "$(sha_file "${path}")" == "${expected}" ]] ||
    fail "hash drift: ${path}"
}

require_line() {
  local path="$1" expected="$2"
  grep -Fqx -- "${expected}" "${path}" ||
    fail "missing exact line in ${path}: ${expected}"
}

sha_limbs() {
  local hex="$1" out='' i part
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

preexec_frame() {
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" "${ZERO}" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  local policy="${1:-1}"
  printf '9020 2 3 1 1 %s 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${policy}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

python_frame() {
  printf '9020 3 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${PYTHON_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

rust_frame() {
  printf '9020 3 4 8 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${RUST_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${RUST_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

llm_authority_frame() {
  printf '9020 3 5 6 1 1 1 1 0 1 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

cpp_authority_frame() {
  printf '9020 3 4 4 1 1 1 1 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

authorize() {
  local frame="$1" expected_sha="$2" expected_rc="$3" expected="$4"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "Guardian frame drift: ${expected_sha}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift: expected ${expected_rc}, got ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift: ${decision}"
}

[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
require_hash "${GARDEN}" "${GARDEN_SHA256}"
require_hash "${PARENT}" "${PARENT_SHA256}"
require_hash "${PARENT_FREEZE}" "${PARENT_FREEZE_SHA256}"
require_hash "${RAW}" "${RAW_SHA256}"
require_hash "${RECEIPT}" "${RECEIPT_SHA256}"
require_hash "${TAMPERED}" "${TAMPERED_SHA256}"
require_hash "${MODULE}" "${MODULE_SHA256}"
require_hash "${EXAMPLE}" "${EXAMPLE_SHA256}"
require_hash "${TEST}" "${TEST_SHA256}"
require_hash "${SUPERSEDED_FREEZE}" "${SUPERSEDED_FREEZE_SHA256}"
require_hash "${FREEZE}" "${FREEZE_SHA256}"
require_hash "${EVIDENCE}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${COMPILER}" "${COMPILER_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_MATERIAL_SEAL_COMMIT}" \
  "${GARDEN_COMMIT}" || fail 'parent material receipt does not precede Garden'
git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" HEAD ||
  fail 'Sounio executable commit is not an ancestor of HEAD'
[[ "$(git -C "${ROOT}" show "${GARDEN_COMMIT}:${GARDEN_REL}" | sha256sum | cut -d' ' -f1)" == "${GARDEN_SHA256}" ]] ||
  fail 'Garden commit hash drift'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == "${MODULE_SHA256}" ]] ||
  fail 'executable commit source hash drift'

actual_manifest="$(
  cd "${ROOT}"
  sha256sum "${MODULE_REL}" "${EXAMPLE_REL}" "${TEST_REL}" |
    sha256sum | cut -d' ' -f1
)"
[[ "${actual_manifest}" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'
actual_semantics="$(cat "${MODULE}" "${EXAMPLE}" "${TEST}" |
  sha256sum | cut -d' ' -f1)"
[[ "${actual_semantics}" == "${SEMANTICS_SHA256}" ]] ||
  fail 'semantics bundle drift'

toolchain_record="$(printf '%s\n' \
  'engine=lean_single' 'wrapper_path=bin/souc' \
  "wrapper_sha256=${WRAPPER_SHA256}" \
  'compiler_path=bin/souc-lean-single-x86_64' \
  "compiler_sha256=${COMPILER_SHA256}")"
[[ "$(sha_text "${toolchain_record}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'toolchain record drift'
hardware_record="$(printf '%s\n' \
  'hostname=sounio-workspace-control-0' 'os=Linux 7.0.2-5-pve' \
  'architecture=x86_64' 'cpu_model=INTEL(R) XEON(R) GOLD 6526Y' \
  'sockets=2' 'cores_per_socket=16' 'threads_per_core=2' \
  'logical_cpus=64')"
[[ "$(sha_text "${hardware_record}")" == "${HARDWARE_SHA256}" ]] ||
  fail 'hardware record drift'
command_record="$(printf '%s\n' \
  'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_u250_execution_engine.sio docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt' \
  'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_u250_execution_engine.sio docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt tests/fixtures/pireus_u250_material_tampered_v1.txt')"
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'command record drift'
result_record="$(printf '%s\n' \
  'stage=SOUNIO_EXECUTABLE' 'status=712091' \
  'status_name=ENGINE_INVENTORY_PARTIAL' 'canonical_target=true' \
  'engine_kind=FPGA' 'fabric=XCU250' 'interface=XRT_XDMA' \
  'declared_engine_count=2' 'observed_engine_count=1' \
  'unresolved_engine_count=1' 'memory_profile_count=1' \
  'graph_triple_count=326' 'blueprint_slot_count=2' \
  'isa_edge_count=0' 'operation_edge_count=0' \
  'machine_bridge_count=1' 'operation_capability_count=0' \
  'lowering_authorized=false' 'parent_material_parity_open=true' \
  'execution_engine_parity_open=false' 'cost_present=false' \
  'kernel_correctness_present=false' 'claim_ready=false' 'failures=0' \
  'negative_cases=15' 'parity_without_receipt=REFUSED' \
  'python_oracle=PREEXEC_REFUSED' \
  'python_process_launched=false')"
[[ "$(sha_text "${result_record}")" == "${RESULT_SHA256}" ]] ||
  fail 'result record drift'

require_line "${FREEZE}" 'stage=SEMANTICS_FROZEN'
require_line "${FREEZE}" 'producing_language=Sounio'
require_line "${FREEZE}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${FREEZE}" "parent_semantics_sha256=${PARENT_SEMANTICS_SHA256}"
require_line "${FREEZE}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${FREEZE}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${FREEZE}" "supersedes_freeze_sha256=${SUPERSEDED_FREEZE_SHA256}"
require_line "${FREEZE}" 'expected_status_name=ENGINE_INVENTORY_PARTIAL'
require_line "${FREEZE}" 'expected_observed_engine_count=1'
require_line "${FREEZE}" 'expected_unresolved_engine_count=1'
require_line "${FREEZE}" 'expected_isa_edge_count=0'
require_line "${FREEZE}" 'expected_operation_edge_count=0'
require_line "${FREEZE}" 'expected_lowering_authorized=false'
require_line "${FREEZE}" 'expected_execution_engine_parity_open=false'
require_line "${FREEZE}" 'expected_claim_ready=false'
require_line "${FREEZE}" 'parity_without_receipt=REFUSED'
require_line "${EVIDENCE}" "freeze_sha256=${FREEZE_SHA256}"
require_line "${EVIDENCE}" 'material_reexecution=false'
require_line "${EVIDENCE}" 'kernel_launch=false'

authorize "$(preexec_frame)" "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-u250-engine-v0.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" \
    "${RAW_REL}" "${RECEIPT_REL}"
) >"${work}/example.txt"
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}" \
    "${RAW_REL}" "${RECEIPT_REL}" "${TAMPERED_REL}"
) >"${work}/test.txt"
require_hash "${work}/example.txt" "${EXAMPLE_OUTPUT_SHA256}"
require_hash "${work}/test.txt" "${TEST_OUTPUT_SHA256}"
require_line "${work}/example.txt" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.u250-execution-engine.v0 stage=SOUNIO_EXECUTABLE'
require_line "${work}/example.txt" ' status=712091'
require_line "${work}/example.txt" ' observed=1'
require_line "${work}/example.txt" ' unresolved=1'
require_line "${work}/example.txt" ' operations=0'
require_line "${work}/example.txt" ' lowering_authorized=0'
require_line "${work}/example.txt" ' engine_parity_open=0'
require_line "${work}/example.txt" ' claim_ready=0'
require_line "${work}/example.txt" ' isa_edges=0'
require_line "${work}/example.txt" ' operation_edges=0'
require_line "${work}/test.txt" \
  'PIREUS_U250_EXECUTION_ENGINE_TEST_PASS status=ENGINE_INVENTORY_PARTIAL slots=2 observed=1 unresolved=1 memory_profiles=1 isa_edges=0 operation_edges=0 machine_bridge=1 cpu=REFUSED gpu=REFUSED xrt_as_isa=REFUSED fabric_as_operation=REFUSED shell_kernel=REFUSED memory_lowering=REFUSED operation_without_receipt=REFUSED cost=REFUSED cpp_authority=REFUSED llm_authority=REFUSED python_authority=REFUSED parity_prefreeze=REFUSED parity_without_receipt=REFUSED claim_ready=REFUSED parent_tamper=REFUSED'

authorize "$(freeze_frame 1)" "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize "$(python_frame)" "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
authorize "$(rust_frame)" "${RUST_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN'
authorize "$(llm_authority_frame)" "${LLM_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
authorize "$(cpp_authority_frame)" "${CPP_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=SEMANTICS_FROZEN'
authorize "$(freeze_frame 0)" "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize "$(freeze_frame 2)" "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'

printf 'PIREUS_U250_EXECUTION_ENGINE_GATE_PASS=true stage=SEMANTICS_FROZEN version=v1 target=AMD_ALVEO_U250 kind=FPGA fabric=XCU250 interface=XRT_XDMA slots=2 observed=1 unresolved=1 memory_profiles=1 isa_edges=0 operation_edges=0 lowering_authorized=false parent_material_parity_open=true engine_parity_open=false parity_without_receipt=REFUSED negatives=15 python_oracle=E110 rust_oracle=E110 python_process_launched=false rust_process_launched=false llm_authority=E113 cpp_authority=E113 policy_missing=E101 policy_timeout=E102 material_reexecution=false kernel_launch=false claim_ready=false\n'
