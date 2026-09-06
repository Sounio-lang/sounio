#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_U250_KERNEL_LAUNCH_RECEIPT_V0.md'
MODULE_REL='stdlib/hardware/pireus/u250_kernel_launch_receipt.sio'
EXAMPLE_REL='examples/pireus_u250_kernel_launch_receipt.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_u250_kernel_launch_receipt.sio'
FREEZE_REL='tools/pireus/u250_kernel_launch_receipt.freeze.v0'
CPP_REL='tools/pireus/u250_kernel_launch_probe.cpp'
RAW_REL='docs/research/evidence/pireus_u250_kernel_launch_probe_20260828.txt'
RECEIPT_REL='docs/research/evidence/pireus_u250_kernel_launch_receipt_20260828.txt'
TAMPERED_REL='tests/fixtures/pireus_u250_kernel_launch_tampered_v0.txt'
EVIDENCE_REL='tools/pireus/evidence/u250_kernel_launch_receipt_v0.txt'
CARD_RAW_REL='docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt'
CARD_RECEIPT_REL='docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt'
XCLBIN_RAW_REL='docs/research/evidence/pireus_u250_xclbin_material_probe_20260828.txt'

GARDEN="${ROOT}/${GARDEN_REL}"
MODULE="${ROOT}/${MODULE_REL}"
EXAMPLE="${ROOT}/${EXAMPLE_REL}"
TEST="${ROOT}/${TEST_REL}"
FREEZE="${ROOT}/${FREEZE_REL}"
CPP="${ROOT}/${CPP_REL}"
RAW="${ROOT}/${RAW_REL}"
RECEIPT="${ROOT}/${RECEIPT_REL}"
TAMPERED="${ROOT}/${TAMPERED_REL}"
EVIDENCE="${ROOT}/${EVIDENCE_REL}"
COMPILER="${ROOT}/bin/souc-lean-single-x86_64"

PARENT_GATE_COMMIT='5edc8f3716adc6f6c879ad86f0250bdbf6aef8b2'
GARDEN_COMMIT='53cfa1ead00baf9084c2d48f7b0baca414d1a498'
EXECUTABLE_COMMIT='916543c972d2cd28f376ef1b3f0d7b1d93434e7f'
FREEZE_COMMIT='3f41fdafae69e9263eeea43df16b61628167470f'
PROBE_SOURCE_COMMIT='a69e49a92d062194daf61787196ea016c5c1bf78'
MATERIAL_COMMIT='6475236fd2f541459a2712072ab25a1bd8675e6f'
RECEIPT_COMMIT='04a679e6fc9e803afe373ce319d240f3ce204f25'

GARDEN_SHA256='b4bfca6b2cad7cffe5d08bad54c88f64010b9f99c8c7aab84b0195c4cb7ab7f0'
MODULE_SHA256='06b65b2a20795cf83d34a47800acc8690e9b54401c50bdc6f16e94b3288dd414'
EXAMPLE_SHA256='a3c9060b0917bfb6eb6c17fff6e509cbfb72f39e515196fb452245a9a096db0b'
TEST_SHA256='209a2ed1c01665268c94b041d78576cf61535b6b615c8d8a3f78585c63a022d4'
FREEZE_SHA256='86ab7ef97e5e39a16442f83c40a4f580d8b72366e0da37a07dc48b2ff2f95ef5'
CPP_SHA256='d5333dad0e518c59676ea768e272ec6cba4d5009b30b7ec02754955e7968c56a'
CPP_BINARY_SHA256='a562a30e3d6c534e48448622fbd53c76c9ff9755b002953c7c0d49b291ce3a8e'
RAW_SHA256='5f36b51d25e9fe45e44a07762e9076e9c9d8c5aaeb36a71639564eb2842ab68d'
RECEIPT_SHA256='7ba3dddb55b2eea57981b1a416f70845d14eab55f1e3bb12d9f2508aeab8f98b'
TAMPERED_SHA256='7014a8d34d4ad3af106e6c7a3a06a930ea42d0565c5f24b1304a240eb9fd1216'
EVIDENCE_SHA256='9b18cc523a3162069f031a1852ca3875f7986c6c7c469c0160b7cdbbd7579c47'
SOURCE_MANIFEST_SHA256='8df664d97f0ec483b690420163bf86558ce2bf6ff2844e8ad37f892c1d497a8b'
SEMANTICS_SHA256='3ee82ed65bd9a33f7aef2d0f6a1b993fef29485bdb6bb449808be071480ff5a0'
PARENT_SEMANTICS_SHA256='8d89d94c4b808548a9b8827ac6f54c29d510a40674f2421e99b497f8a6f32f05'
PARENT_FREEZE_SHA256='662fe6e6f2dc0c5b27f227c26e5d92d1f0da3c588e7ef94accbf4b49b8abd2e5'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'

SOUNIO_TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
SOUNIO_HARDWARE_SHA256='46262c5d0fc8df5734998677c2ad063c686fa2d1120fb8dc18dc5b382c7c4805'
WAITING_COMMAND_SHA256='3fd275facda04accab400d58b8911c6003557cdf2cca52723a0dd3bdb1d44ca7'
WAITING_RESULT_SHA256='720cf6dadae8e2982e42ba7264adc04a4148f5bb32e37ee980bfa14016e27bdf'
WAITING_OUTPUT_SHA256='19d883aa755b578d20725fa8478222f409c15861a45ff7242cf050be7cbf83a5'
PARITY_COMMAND_SHA256='9126e041b95643b61d796a22e6f9cea96112ae484339478e092f4a153e69a41c'
PARITY_RESULT_SHA256='7603a47bfefc35f15cb9b8b256f5baabe0eb97b8e779212c5627dabb589914a9'
PARITY_OUTPUT_SHA256='f36cc51451f9996ed1ef8d9040b4c4e2a3aaa817ddea13182c54bd4ea481fe6a'
TEST_OUTPUT_SHA256='9c06e17794a9b24e83cbe6a18cb061c1ca1cb5e91349c3548dc833090eb5a717'

MATERIAL_TOOLCHAIN_SHA256='683759ae2d3f34e05a780a935399bb3b12c3acbf3d6a81d863620b572d231e3e'
MATERIAL_HARDWARE_SHA256='02bde35408178d3a691d69a8f4f10099c45ab23f468d673cb27f4fedc798554a'
MATERIAL_COMMAND_SHA256='5bb07529f0f6bb2536395957034f8703854cc7e1e5e3f833afecf9523c964579'
RESULT_CONTRACT_SHA256='7bd0d2d7b7838e046a619960c149622b95b5ab4856eb496ff398daaa48d70594'
MATERIAL_RESULT_SHA256='65ba4b885b2dc0d56f84b8115791f82457d3e0a7351d1fce58d742a687b0e2f0'

PREEXEC_FRAME_SHA256='f4acfeedd6bad2f700138bcf9179e9e39c309a0f69c5a22db5118c3f43bdf208'
FREEZE_FRAME_SHA256='baba641badb3725ccd9f0475f52d5afa83cf25fbe0bdde4b9218515891da85ee'
MATERIAL_PREEXEC_FRAME_SHA256='effc9fdc563a62368c097f1d8d5cbfad91961b0c8cb140a7956f13452222b84a'
MATERIAL_SEAL_FRAME_SHA256='4a8442213ec91238836d5e1217ddab04ea31026e98b1f3ce8d25d07534bfb2ce'
SOUNIO_SEAL_FRAME_SHA256='15ea47c2baae3d8586dcd35512802f387e6f568ece1d13b660ab66f0c4e763ac'
PYTHON_FRAME_SHA256='c2b346eb525a57fe2348278dd9bdc84c9ad6b44dd2325398043c2f83483ab9db'
RUST_FRAME_SHA256='567e419835d49d41745d828b57607622cb8aa26d9644a5583c6e92c0396bb5d0'
LLM_FRAME_SHA256='8a9d05fa1113d40e4c692c7be351b083431c3217d63c47bf1f740ebee9a7d335'
CPP_AUTHORITY_FRAME_SHA256='7fa403731abcc1625e1e628ec489ffb937ff9df7ea48338ca047b644d969b9f3'
POLICY_MISSING_FRAME_SHA256='7ee66e70e8b21a56f7c72de7a288af317c1cfa947e7fa39338831be111af4338'
POLICY_TIMEOUT_FRAME_SHA256='cdd8d4c8c426dccf8c93b5fa9a3676a4a0006071f66c25011e80bea4792a8221'
CLAIM_FRAME_SHA256='af229064c82565dba2773a99cdb3f1cd88c44cd6b80abd9eafb3af0d8b934ac2'

PYTHON_TOOLCHAIN_SHA256='5c8cfd947420cd48743adb75469089a210d7782421a4e9e46bfc4c40021fb7cf'
PYTHON_COMMAND_SHA256='adedad5e22a944b2a99549bc78377f849e8159d831aecfd7679a356746a06067'
RUST_TOOLCHAIN_SHA256='ae2ed5be700fb051fe99519bd8a9376ad8e4d5091c394ccbb9d8b57c36b7cd11'
RUST_COMMAND_SHA256='b47413e1f12243e68533941a2fdc86080877a56609a0388332cadedce0c2282d'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus U250 kernel launch receipt: FAIL: %s\n' "$*" >&2
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
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${WAITING_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  local policy="${1:-1}"
  printf '9020 2 3 1 1 %s 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${policy}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${WAITING_COMMAND_SHA256}")" \
    "$(sha_limbs "${WAITING_RESULT_SHA256}")" "${ZERO}"
}

policy_frame() {
  local policy="$1"
  printf '9020 2 3 1 1 %s 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${policy}" "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${PARITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${PARITY_RESULT_SHA256}")" "${ZERO}"
}

material_preexec_frame() {
  printf '9020 3 4 4 4 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${MATERIAL_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${MATERIAL_HARDWARE_SHA256}")" \
    "$(sha_limbs "${MATERIAL_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

material_seal_frame() {
  printf '9020 4 8 4 4 1 0 0 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${MATERIAL_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${MATERIAL_HARDWARE_SHA256}")" \
    "$(sha_limbs "${MATERIAL_COMMAND_SHA256}")" \
    "$(sha_limbs "${RAW_SHA256}")" "${ZERO}"
}

sounio_seal_frame() {
  printf '9020 4 8 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${PARITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${PARITY_RESULT_SHA256}")" "${ZERO}"
}

python_frame() {
  printf '9020 4 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PYTHON_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${PYTHON_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

rust_frame() {
  printf '9020 4 4 8 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${RUST_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${RUST_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

llm_authority_frame() {
  printf '9020 4 5 6 1 1 1 1 0 1 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${PARITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${PARITY_RESULT_SHA256}")" "${ZERO}"
}

cpp_authority_frame() {
  printf '9020 4 4 4 1 1 1 1 1 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${PARITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${PARITY_RESULT_SHA256}")" "${ZERO}"
}

claim_frame() {
  printf '9020 4 7 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${MODULE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${SOUNIO_TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${SOUNIO_HARDWARE_SHA256}")" \
    "$(sha_limbs "${PARITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${PARITY_RESULT_SHA256}")" "${ZERO}"
}

authorize() {
  local label="$1" frame="$2" expected_sha="$3" expected_rc="$4" expected="$5"
  local decision rc
  [[ "$(sha_text "${frame}")" == "${expected_sha}" ]] ||
    fail "Guardian frame drift: ${label}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" -eq "${expected_rc}" ]] ||
    fail "Guardian exit drift for ${label}: expected ${expected_rc}, got ${rc}"
  [[ "${decision}" == "${expected}" ]] ||
    fail "Guardian decision drift for ${label}: ${decision}"
  printf 'GUARDIAN_DECISION label=%s decision=%s\n' "${label}" "${decision}"
}

[[ -x "${GUARDIAN}" ]] || fail 'native Loom Guardian unavailable'
require_hash "${GARDEN}" "${GARDEN_SHA256}"
require_hash "${MODULE}" "${MODULE_SHA256}"
require_hash "${EXAMPLE}" "${EXAMPLE_SHA256}"
require_hash "${TEST}" "${TEST_SHA256}"
require_hash "${FREEZE}" "${FREEZE_SHA256}"
require_hash "${CPP}" "${CPP_SHA256}"
require_hash "${RAW}" "${RAW_SHA256}"
require_hash "${RECEIPT}" "${RECEIPT_SHA256}"
require_hash "${TAMPERED}" "${TAMPERED_SHA256}"
require_hash "${EVIDENCE}" "${EVIDENCE_SHA256}"
require_hash "${ROOT}/bin/souc" "${WRAPPER_SHA256}"
require_hash "${COMPILER}" "${COMPILER_SHA256}"

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_GATE_COMMIT}" \
  "${GARDEN_COMMIT}" || fail 'parent material gate does not precede Garden'
git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" \
  "${FREEZE_COMMIT}" || fail 'Sounio executable does not precede freeze'
git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_COMMIT}" \
  "${PROBE_SOURCE_COMMIT}" || fail 'C++ probe source preceded Sounio freeze'
git -C "${ROOT}" merge-base --is-ancestor "${PROBE_SOURCE_COMMIT}" \
  "${MATERIAL_COMMIT}" || fail 'material observation preceded probe source'
git -C "${ROOT}" merge-base --is-ancestor "${MATERIAL_COMMIT}" \
  "${RECEIPT_COMMIT}" || fail 'receipt preceded material observation'
git -C "${ROOT}" merge-base --is-ancestor "${RECEIPT_COMMIT}" HEAD ||
  fail 'receipt commit is not an ancestor of HEAD'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == "${MODULE_SHA256}" ]] ||
  fail 'executable source hash drift'
if git -C "${ROOT}" cat-file -e "${FREEZE_COMMIT}:${CPP_REL}" 2>/dev/null; then
  fail 'C++ launch probe existed at Sounio freeze commit'
fi
[[ "$(git -C "${ROOT}" show "${PROBE_SOURCE_COMMIT}:${CPP_REL}" | sha256sum | cut -d' ' -f1)" == "${CPP_SHA256}" ]] ||
  fail 'C++ launch probe source hash drift'

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
[[ "$(sha_text "${toolchain_record}")" == "${SOUNIO_TOOLCHAIN_SHA256}" ]] ||
  fail 'Sounio toolchain record drift'
hardware_record="$(printf '%s\n' \
  'hostname=sounio-workspace-control-0' 'os=Linux 7.0.2-5-pve' \
  'architecture=x86_64' 'cpu_model=INTEL(R) XEON(R) GOLD 6526Y' \
  'sockets=2' 'cores_per_socket=16' 'threads_per_core=2' \
  'logical_cpus=64')"
[[ "$(sha_text "${hardware_record}")" == "${SOUNIO_HARDWARE_SHA256}" ]] ||
  fail 'Sounio hardware record drift'
material_toolchain_record="$(printf '%s\n' \
  'compiler_path=/usr/bin/g++' \
  'compiler_version=g++ (Debian 14.2.0-19) 14.2.0' \
  'xrt_version=2.23.0' 'xrt_root=/opt/xilinx/xrt' \
  'xrt_coreutil=libxrt_coreutil.so.2.23.0')"
[[ "$(sha_text "${material_toolchain_record}")" == "${MATERIAL_TOOLCHAIN_SHA256}" ]] ||
  fail 'material toolchain record drift'
material_hardware_record="$(printf '%s\n' \
  'hostname=dl380-proxmox' 'os=Linux 7.0.14-8-pve' \
  'architecture=x86_64' \
  'cpu_model=Intel(R) Xeon(R) Gold 6262V CPU @ 1.90GHz' \
  'device_bdf=0000:d8:00.1' 'card_serial=22000321B01F' \
  'shell=xilinx_u250_gen3x16_xdma_shell_4_1')"
[[ "$(sha_text "${material_hardware_record}")" == "${MATERIAL_HARDWARE_SHA256}" ]] ||
  fail 'material hardware record drift'
material_command_record="$(printf '%s\n' \
  'probe=/tmp/pireus-u250-kernel-launch-probe' \
  'artifact=/tmp/krnl_san_scan.hw.xclbin' \
  'device_bdf=0000:d8:00.1' 'kernel=krnl_san_scan' \
  'n_samples=4' 'n_points=2' 'q_delta=16384')"
[[ "$(sha_text "${material_command_record}")" == "${MATERIAL_COMMAND_SHA256}" ]] ||
  fail 'material command record drift'
result_contract_record="$(printf '%s\n' \
  'device_programmed=true' 'kernel_opened=true' 'buffers_allocated=5' \
  'inputs_synced=true' 'run_submitted=true' 'run_completed=true' \
  'outputs_synced=3' 'output_values_recorded=true' \
  'semantic_verdict_emitted=false' 'kernel_correctness_present=false' \
  'performance_present=false' 'claim_ready=false')"
[[ "$(sha_text "${result_contract_record}")" == "${RESULT_CONTRACT_SHA256}" ]] ||
  fail 'result contract record drift'
material_result_record="$(printf '%s\n' \
  'probe_exit=0' "raw_probe_sha256=${RAW_SHA256}" 'raw_probe_lines=47' \
  'device_programmed=true' 'kernel_opened=true' 'buffers_allocated=5' \
  'run_completed=true' 'outputs_synced=3' 'output_values_recorded=true' \
  'probe_valid=true' 'semantic_verdict_emitted=false')"
[[ "$(sha_text "${material_result_record}")" == "${MATERIAL_RESULT_SHA256}" ]] ||
  fail 'material result record drift'

require_line "${FREEZE}" 'stage=SEMANTICS_FROZEN'
require_line "${FREEZE}" 'producing_language=Sounio'
require_line "${FREEZE}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${FREEZE}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${FREEZE}" 'cpp_launch_probe_source_present=false'
require_line "${FREEZE}" 'cpp_launch_probe_compiled=false'
require_line "${FREEZE}" 'cpp_launch_probe_executed=false'
require_line "${FREEZE}" 'kernel_launch=false'
require_line "${RAW}" 'producer_language=C++'
require_line "${RAW}" 'producer_role=MATERIAL_PARITY'
require_line "${RAW}" 'semantic_authority_language=Sounio'
require_line "${RAW}" 'device_programmed=true'
require_line "${RAW}" 'kernel_opened=true'
require_line "${RAW}" 'run_completed=true'
require_line "${RAW}" 'outputs_synced=3'
require_line "${RAW}" 'output_values_recorded=true'
require_line "${RAW}" 'semantic_verdict_emitted=false'
require_line "${RAW}" 'expected_output_present=false'
require_line "${RAW}" 'kernel_correctness_present=false'
require_line "${RAW}" 'performance_present=false'
require_line "${RAW}" 'claim_ready=false'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=C++'
require_line "${RECEIPT}" 'language_role=MATERIAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
require_line "${RECEIPT}" "sounio_freeze_sha256=${FREEZE_SHA256}"
require_line "${RECEIPT}" "cpp_source_sha256=${CPP_SHA256}"
require_line "${RECEIPT}" "cpp_binary_sha256=${CPP_BINARY_SHA256}"
require_line "${RECEIPT}" 'output_values_interpreted=false'
require_line "${RECEIPT}" 'execution_parity_open=true'
require_line "${RECEIPT}" 'kernel_execution_observed=true'
require_line "${RECEIPT}" 'kernel_correctness_present=false'
require_line "${RECEIPT}" 'lowering_authorized=false'
require_line "${RECEIPT}" 'isa_claim_present=false'
require_line "${RECEIPT}" 'performance_present=false'
require_line "${RECEIPT}" 'claim_ready=false'
require_line "${EVIDENCE}" "material_receipt_sha256=${RECEIPT_SHA256}"
require_line "${EVIDENCE}" 'tampered_probe=REFUSED'
require_line "${EVIDENCE}" 'python_process_launched=false'
require_line "${EVIDENCE}" 'rust_process_launched=false'
require_line "${EVIDENCE}" 'default_compiler_result=FAILED_NO_ELF'
require_line "${EVIDENCE}" 'sounio_executable_engine_role=EXPLICIT_BOOTSTRAP_FALLBACK'

tracked_bits="$(git -C "${ROOT}" ls-files '*.xclbin' '*.xo')"
[[ -z "${tracked_bits}" ]] || fail 'xclbin/xo must remain outside Git'

authorize PREEXEC "$(preexec_frame)" "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize FREEZE "$(freeze_frame 1)" "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize MATERIAL_PREEXEC "$(material_preexec_frame)" \
  "${MATERIAL_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize MATERIAL_SEAL "$(material_seal_frame)" \
  "${MATERIAL_SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize SOUNIO_SEAL "$(sounio_seal_frame)" "${SOUNIO_SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize PYTHON_ORACLE "$(python_frame)" "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN'
authorize RUST_ORACLE "$(rust_frame)" "${RUST_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN'
authorize LLM_AUTHORITY "$(llm_authority_frame)" "${LLM_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=PARITY_OPEN'
authorize CPP_AUTHORITY "$(cpp_authority_frame)" \
  "${CPP_AUTHORITY_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=PARITY_OPEN'
authorize POLICY_MISSING "$(policy_frame 0)" "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize POLICY_TIMEOUT "$(policy_frame 2)" "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
authorize CLAIM_PROMOTION "$(claim_frame)" "${CLAIM_FRAME_SHA256}" 122 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=122 reason=parity-receipt-missing next_stage=PARITY_OPEN'

"${ROOT}/scripts/ci/pireus_u250_xclbin_material_receipt.sh" >/dev/null

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-u250-kernel-launch-v0.XXXXXX")"
trap 'rm -rf "${work}"' EXIT
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" \
    "${CARD_RAW_REL}" "${CARD_RECEIPT_REL}" "${XCLBIN_RAW_REL}"
) >"${work}/waiting.txt"
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" \
    "${CARD_RAW_REL}" "${CARD_RECEIPT_REL}" "${XCLBIN_RAW_REL}" "${RAW_REL}"
) >"${work}/parity.txt"
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}" \
    "${CARD_RAW_REL}" "${CARD_RECEIPT_REL}" "${XCLBIN_RAW_REL}" \
    "${RAW_REL}" "${TAMPERED_REL}"
) >"${work}/test.txt"
require_hash "${work}/waiting.txt" "${WAITING_OUTPUT_SHA256}"
require_hash "${work}/parity.txt" "${PARITY_OUTPUT_SHA256}"
require_hash "${work}/test.txt" "${TEST_OUTPUT_SHA256}"
require_line "${work}/waiting.txt" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.u250-kernel-launch-receipt.v0 stage=SOUNIO_EXECUTABLE'
require_line "${work}/waiting.txt" ' status=715090'
require_line "${work}/waiting.txt" ' launch_observed=0'
require_line "${work}/waiting.txt" ' execution_parity_open=0'
require_line "${work}/parity.txt" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.u250-kernel-launch-receipt.v0 stage=PARITY_OPEN'
require_line "${work}/parity.txt" ' status=715091'
require_line "${work}/parity.txt" ' launch_observed=1'
require_line "${work}/parity.txt" ' lifecycle_observations=7'
require_line "${work}/parity.txt" ' execution_parity_open=1'
require_line "${work}/parity.txt" ' execution_observed=1'
require_line "${work}/parity.txt" ' correctness_present=0'
require_line "${work}/parity.txt" ' operation_capabilities=0'
require_line "${work}/parity.txt" ' lowering_authorized=0'
require_line "${work}/parity.txt" ' isa_claim=0'
require_line "${work}/parity.txt" ' performance_present=0'
require_line "${work}/parity.txt" ' claim_ready=0'
require_line "${work}/parity.txt" 'PIREUS_U250_KERNEL_LAUNCH_GRAPH triples=457'
require_line "${work}/parity.txt" ' receipt_edges=1'
require_line "${work}/parity.txt" ' lifecycle_edges=1'
require_line "${work}/parity.txt" ' output_edges=1'
require_line "${work}/test.txt" \
  'PIREUS_U250_KERNEL_LAUNCH_TEST_PASS waiting=1 parity_projection=1 parent=REFUSED premature=REFUSED incomplete=REFUSED missing_output=REFUSED device_drift=REFUSED digest_drift=REFUSED samples_drift=REFUSED lut_drift=REFUSED width_drift=REFUSED cpp_authority=REFUSED python_authority=REFUSED rust_authority=REFUSED llm_authority=REFUSED child_verdict=REFUSED expected_output=REFUSED correctness=REFUSED operation=REFUSED lowering=REFUSED isa=REFUSED performance=REFUSED claim_ready=REFUSED'

printf 'PIREUS_U250_KERNEL_LAUNCH_RECEIPT_GATE_PASS=true stage=PARITY_OPEN artifact_sha256=d30078c7b2e8690aef892b4b6cf96af0f490b70e2b367e5e3679be04fcd4bdbf device_bdf=0000:d8:00.1 serial=22000321B01F kernel=krnl_san_scan n_samples=4 lifecycle_observations=7 buffers=5 output_syncs=3 triples=457 receipt_edges=1 lifecycle_edges=1 output_edges=1 tampered=REFUSED python_oracle=E110 rust_oracle=E110 python_process_launched=false rust_process_launched=false llm_authority=E113 cpp_authority=E113 policy_missing=E101 policy_timeout=E102 claim_promotion=E122 xclbin_build=false fpga_programming_observed=true kernel_launch_observed=true output_values_interpreted=false correctness_present=false operation_capabilities=0 lowering_authorized=false isa_claim=false performance_present=false claim_ready=false\n'
