#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_U250_FPGA_KERNEL_ARTIFACT_V0.md'
PARENT_REL='stdlib/hardware/pireus/u250_execution_engine.sio'
PARENT_FREEZE_REL='tools/pireus/u250_execution_engine.freeze.v1'
RAW_REL='docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt'
RECEIPT_REL='docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt'
TAMPERED_REL='tests/fixtures/pireus_u250_material_tampered_v1.txt'
MODULE_REL='stdlib/hardware/pireus/u250_fpga_kernel_artifact.sio'
EXAMPLE_REL='examples/pireus_u250_fpga_kernel_artifact.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_u250_fpga_kernel_artifact.sio'
FREEZE_REL='tools/pireus/u250_fpga_kernel_artifact.freeze.v0'
EVIDENCE_REL='tools/pireus/evidence/u250_fpga_kernel_artifact_v0.txt'
HLS_SOURCE_REL='hardware/fpga/u250_catastrophe_scan/krnl_san_scan.cpp'
HLS_BUILD_REL='hardware/fpga/u250_catastrophe_scan/build_san_scan_xclbin.sh'
XRT_HOST_REL='hardware/fpga/u250_catastrophe_scan/host_san_scan.cpp'

GARDEN="${ROOT}/${GARDEN_REL}"
PARENT="${ROOT}/${PARENT_REL}"
PARENT_FREEZE="${ROOT}/${PARENT_FREEZE_REL}"
RAW="${ROOT}/${RAW_REL}"
RECEIPT="${ROOT}/${RECEIPT_REL}"
TAMPERED="${ROOT}/${TAMPERED_REL}"
MODULE="${ROOT}/${MODULE_REL}"
EXAMPLE="${ROOT}/${EXAMPLE_REL}"
TEST="${ROOT}/${TEST_REL}"
FREEZE="${ROOT}/${FREEZE_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
HLS_SOURCE="${ROOT}/${HLS_SOURCE_REL}"
HLS_BUILD="${ROOT}/${HLS_BUILD_REL}"
XRT_HOST="${ROOT}/${XRT_HOST_REL}"
COMPILER="${ROOT}/bin/souc-lean-single-x86_64"

PARENT_ENGINE_SEAL_COMMIT='327f0f3cded20b94faf5581a4b98fa7da58382af'
GARDEN_COMMIT='6f7d23129122c6163222b2050607e756d5cafb25'
EXECUTABLE_COMMIT='fa0eec43b531fe448f512195c06b7b85c460009d'

GARDEN_SHA256='bb49feca3c86a810eb7177889127f85f5f73d0012815ee8002393fc13bcd0f2a'
PARENT_SHA256='2b39ae0d92d18fdf7da966264f44159a8a1ecfedd0ae7cb09e03333f4b5bebbd'
PARENT_FREEZE_SHA256='b1ac42ba32baf967481c9eed888d8d3cb776a702a3eaf613dbf19af2e4994aab'
PARENT_SEMANTICS_SHA256='93c309d0c381464cc5d3e411a7227dd0b25174eb9e913c3c2fbaee7097d2c218'
RAW_SHA256='6bea3b962c519dfe9a9878c008a6300b67b920f0a2b51ba9d89dbf180661e7df'
RECEIPT_SHA256='9889567b684fcc0213ed38a44041e8475c4c9a71722b7baa1c6c064e1f1d0d7a'
TAMPERED_SHA256='711c21a8b60e9c2717ca819b847b41779eafaab0d8f96924122146b76561164f'
MODULE_SHA256='4fb3af027b717bd97ef224a39940404cdf9724d563d087e3842a386d7c43c465'
EXAMPLE_SHA256='539c0c0adc224574e1eb1b0746fdaf8fec1876151e09aa8c590e3d1c839bb90d'
TEST_SHA256='45ee275b67c417bed7def26fb17cb5b00f2f48cac3ce7cf60be777c146e62eb5'
FREEZE_SHA256='89278d99fab89bc2b582958a27d2806775b0b13a1e8f258550924fc20e3dc05e'
EVIDENCE_SHA256='17c5ca23e87177d94af079030d86c23faf9fe28b0e4dee732dc25fbedd881c37'
HLS_SOURCE_SHA256='26171968951bc9aa39a82bbadb4665306e5b80368de040e3dc52c49d68e89700'
HLS_BUILD_SHA256='cd27b7f960672ebe4b8ef716c4b36e20c631eea9ddf6272dda6260258526f3e4'
XRT_HOST_SHA256='d824a8d2ae08bdf87ae6af318b994348effa9d45c2d9dd8275d072ce1bc7ec62'
SOURCE_MANIFEST_SHA256='b61fd1f1ca989702bda1e4e4b56864cf6217b1087178067f6962462b8e2f61b3'
SEMANTICS_SHA256='e7d4a83e81c054a1d15808292d49fbcda6ea43a06dbf31469e7c4c81d51d3fe5'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'
TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
HARDWARE_SHA256='46262c5d0fc8df5734998677c2ad063c686fa2d1120fb8dc18dc5b382c7c4805'
COMMAND_SHA256='27241750fafc58992058578ded434f759985301d482bee7a3fe1746d8324158b'
RESULT_SHA256='b32256bcb32379193839b9826096b3a72f10acafce62e2c342721446435dcff4'
EXAMPLE_OUTPUT_SHA256='9b7340e96c2923dc877d081d0e7ddae2f6735affd81a705eed98b50b95f5a198'
TEST_OUTPUT_SHA256='d141dbabd4a7a06a6c4e9b0d3d9fe6fd970c17da9a4e2fb1bf4bd89d755d7753'
PREEXEC_FRAME_SHA256='f24a313f26c8cdc402d16a7e1d3dc4d2230834ef2cae2f704bfd93edc2826108'
FREEZE_FRAME_SHA256='8548b7c406118da142fa6b74a7cb45c0ccbe8f318a5e0c7a45252dcd238936ef'
PYTHON_FRAME_SHA256='81d0bae777d57b8ee4430a93e82123176530a79f0971ee78f7ebf4d0eea662bc'
RUST_FRAME_SHA256='99596c2a029b1406b3adca12e85145e2506bf50b45943bba727d1480eb9c8c76'
LLM_FRAME_SHA256='f275d7b73878b86648a72f67ddab481cc74c61b8c9b21b9d1e6d5a889e38eae4'
CPP_FRAME_SHA256='41f6ff7b82291364cc1a6f6305fe8d413c1c6c6a9911fb617819a0c36879d7b3'
POLICY_MISSING_FRAME_SHA256='843e5a5c9ada2433467fdcbf094d03ceef64a8d353ae43c7a3223c3a490ca677'
POLICY_TIMEOUT_FRAME_SHA256='c0d39344c2cb02db2ec9c64fd06a759516001fbe9e28b4aa10e1bf26954c1390'
PYTHON_TOOLCHAIN_SHA256='5c8cfd947420cd48743adb75469089a210d7782421a4e9e46bfc4c40021fb7cf'
PYTHON_COMMAND_SHA256='39d4704375b8db3fa1a86377da138b87363d09893df457b49e5f6f8a78809551'
RUST_TOOLCHAIN_SHA256='ae2ed5be700fb051fe99519bd8a9376ad8e4d5091c394ccbb9d8b57c36b7cd11'
RUST_COMMAND_SHA256='d38eb598379d9d455823308a7b90d494bbd7723845bb9eefc71bb4d1dd2c2059'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus U250 FPGA kernel artifact: FAIL: %s\n' "$*" >&2
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
require_hash "${FREEZE}" "${FREEZE_SHA256}"
require_hash "${EVIDENCE}" "${EVIDENCE_SHA256}"
require_hash "${HLS_SOURCE}" "${HLS_SOURCE_SHA256}"
require_hash "${HLS_BUILD}" "${HLS_BUILD_SHA256}"
require_hash "${XRT_HOST}" "${XRT_HOST_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${COMPILER}" "${COMPILER_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_ENGINE_SEAL_COMMIT}" \
  "${GARDEN_COMMIT}" || fail 'parent engine freeze does not precede Garden'
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
  'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_u250_fpga_kernel_artifact.sio docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt' \
  'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_u250_fpga_kernel_artifact.sio docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt tests/fixtures/pireus_u250_material_tampered_v1.txt')"
[[ "$(sha_text "${command_record}")" == "${COMMAND_SHA256}" ]] ||
  fail 'command record drift'
result_record="$(printf '%s\n' \
  'stage=SOUNIO_EXECUTABLE' 'status=713090' 'status_name=BLUEPRINT_ONLY' \
  'engine=712201' 'kernel_symbol=krnl_san_scan' \
  'artifact_format=XCLBIN' \
  'platform=xilinx_u250_gen3x16_xdma_4_1_202210_1' \
  'target_frequency_mhz=250' 'kernel_blueprint_count=1' \
  'abi_argument_count=8' 'input_buffer_count=2' \
  'scalar_input_count=3' 'output_buffer_count=3' \
  'm_axi_argument_count=5' 's_axilite_argument_count=3' \
  'graph_triple_count=420' 'artifact_edge_count=1' \
  'argument_edge_count=8' 'bits_edge_count=0' 'isa_edge_count=0' \
  'operation_edge_count=0' 'checked_source_present=true' \
  'build_recipe_present=true' 'bitstream_present=false' \
  'bitstream_digest_present=false' 'abi_parity_open=false' \
  'kernel_execution_observed=false' 'kernel_correctness_present=false' \
  'operation_capability_count=0' 'lowering_authorized=false' \
  'performance_present=false' 'artifact_parity_open=false' \
  'claim_ready=false' 'failures=0' 'negative_cases=23' \
  'legacy_python_golden=REFUSED' 'python_oracle=PREEXEC_REFUSED' \
  'python_process_launched=false')"
[[ "$(sha_text "${result_record}")" == "${RESULT_SHA256}" ]] ||
  fail 'result record drift'

require_line "${FREEZE}" 'stage=SEMANTICS_FROZEN'
require_line "${FREEZE}" 'producing_language=Sounio'
require_line "${FREEZE}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${FREEZE}" "parent_semantics_sha256=${PARENT_SEMANTICS_SHA256}"
require_line "${FREEZE}" "source_manifest_sha256=${SOURCE_MANIFEST_SHA256}"
require_line "${FREEZE}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${FREEZE}" 'legacy_material_role=HISTORICAL_NON_AUTHORITY'
require_line "${FREEZE}" 'legacy_python_golden_role=REVIEW_ONLY_REFUSED_AS_AUTHORITY'
require_line "${FREEZE}" 'expected_status_name=BLUEPRINT_ONLY'
require_line "${FREEZE}" 'expected_abi_argument_count=8'
require_line "${FREEZE}" 'expected_bits_edge_count=0'
require_line "${FREEZE}" 'expected_isa_edge_count=0'
require_line "${FREEZE}" 'expected_operation_edge_count=0'
require_line "${FREEZE}" 'expected_bitstream_present=false'
require_line "${FREEZE}" 'expected_abi_parity_open=false'
require_line "${FREEZE}" 'expected_kernel_execution_observed=false'
require_line "${FREEZE}" 'expected_kernel_correctness_present=false'
require_line "${FREEZE}" 'expected_lowering_authorized=false'
require_line "${FREEZE}" 'expected_artifact_parity_open=false'
require_line "${FREEZE}" 'expected_claim_ready=false'
require_line "${FREEZE}" 'legacy_python_golden=REFUSED'
require_line "${FREEZE}" 'parity_without_receipt=REFUSED'
require_line "${FREEZE}" 'abi_parity_without_receipt=REFUSED'
require_line "${EVIDENCE}" "freeze_sha256=${FREEZE_SHA256}"
require_line "${EVIDENCE}" 'bitstream_digest=ABSENT'
require_line "${EVIDENCE}" 'xclbin_build=false'
require_line "${EVIDENCE}" 'fpga_programming=false'
require_line "${EVIDENCE}" 'kernel_launch=false'

tracked_bits="$(git -C "${ROOT}" ls-files '*.xclbin' '*.xo')"
[[ -z "${tracked_bits}" ]] ||
  fail 'tracked xclbin/xo exists without a material receipt ingestor'

authorize "$(preexec_frame)" "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-u250-kernel-artifact-v0.XXXXXX")"
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
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.u250-fpga-kernel-artifact.v0 stage=SOUNIO_EXECUTABLE'
require_line "${work}/example.txt" ' status=713090'
require_line "${work}/example.txt" ' abi_arguments=8'
require_line "${work}/example.txt" ' bitstream_present=0'
require_line "${work}/example.txt" ' abi_parity_open=0'
require_line "${work}/example.txt" ' execution_observed=0'
require_line "${work}/example.txt" ' correctness_present=0'
require_line "${work}/example.txt" ' lowering_authorized=0'
require_line "${work}/example.txt" ' artifact_parity_open=0'
require_line "${work}/example.txt" ' claim_ready=0'
require_line "${work}/example.txt" ' bits_edges=0'
require_line "${work}/example.txt" ' isa_edges=0'
require_line "${work}/example.txt" ' operation_edges=0'
require_line "${work}/test.txt" \
  'PIREUS_U250_FPGA_KERNEL_ARTIFACT_TEST_PASS status=BLUEPRINT_ONLY kernel=krnl_san_scan abi_arguments=8 input_buffers=2 scalar_inputs=3 output_buffers=3 m_axi=5 s_axilite=3 bits_edges=0 isa_edges=0 operation_edges=0 outline=REFUSED source_implies_bits=REFUSED build_implies_bits=REFUSED unsealed_bits=REFUSED digest_without_bits=REFUSED receipt_without_ingestor=REFUSED execution_without_receipt=REFUSED python_authority=REFUSED rust_authority=REFUSED cpp_authority=REFUSED llm_authority=REFUSED legacy_python_golden=REFUSED xrt_as_isa=REFUSED artifact_as_operation=REFUSED correctness=REFUSED lowering=REFUSED performance=REFUSED parity_without_receipt=REFUSED abi_parity_without_receipt=REFUSED claim_ready=REFUSED parent_tamper=REFUSED'

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

printf 'PIREUS_U250_FPGA_KERNEL_ARTIFACT_GATE_PASS=true stage=SEMANTICS_FROZEN version=v0 engine=U250_SLOT_0 kernel=krnl_san_scan artifact=XCLBIN blueprint=1 abi_arguments=8 bits_edges=0 isa_edges=0 operation_edges=0 bitstream_present=false abi_parity_open=false execution_observed=false correctness_present=false lowering_authorized=false artifact_parity_open=false negatives=23 legacy_python_golden=REFUSED python_oracle=E110 rust_oracle=E110 python_process_launched=false rust_process_launched=false llm_authority=E113 cpp_authority=E113 policy_missing=E101 policy_timeout=E102 xclbin_build=false fpga_programming=false kernel_launch=false claim_ready=false\n'
