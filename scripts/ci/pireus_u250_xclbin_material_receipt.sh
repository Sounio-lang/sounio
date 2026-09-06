#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN_REL='tools/pireus/GARDEN_U250_XCLBIN_MATERIAL_RECEIPT_V0.md'
MODULE_REL='stdlib/hardware/pireus/u250_xclbin_material_receipt.sio'
EXAMPLE_REL='examples/pireus_u250_xclbin_material_receipt.sio'
TEST_REL='tests/stdlib/hardware/test_pireus_u250_xclbin_material_receipt.sio'
FREEZE_REL='tools/pireus/u250_xclbin_material_receipt.freeze.v0'
CPP_REL='tools/pireus/u250_xclbin_material_probe.cpp'
RAW_REL='docs/research/evidence/pireus_u250_xclbin_material_probe_20260828.txt'
RECEIPT_REL='docs/research/evidence/pireus_u250_xclbin_material_receipt_20260828.txt'
TAMPERED_REL='tests/fixtures/pireus_u250_xclbin_material_tampered_v0.txt'
EVIDENCE_REL='tools/pireus/evidence/u250_xclbin_material_receipt_v0.txt'
CARD_RAW_REL='docs/research/evidence/pireus_u250_dl380_material_probe_20260828.txt'
CARD_RECEIPT_REL='docs/research/evidence/pireus_u250_dl380_material_parity_20260828.txt'

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
CARD_RAW="${ROOT}/${CARD_RAW_REL}"
CARD_RECEIPT="${ROOT}/${CARD_RECEIPT_REL}"
COMPILER="${ROOT}/bin/souc-lean-single-x86_64"

PARENT_FREEZE_COMMIT='a4973209a83e9eba09314ce0e7bf1cd4fd63353c'
GARDEN_COMMIT='1f0a78872af7daa47b82bb0d00973b198796317e'
EXECUTABLE_COMMIT='28db0c1b90baf8910869589c509dda828b435658'
FREEZE_COMMIT='6bd1e9fd8318d20049f324b0d673e09c8b69797a'
MATERIAL_COMMIT='d775c249d151b6ca1a8d320b32e5238dfc027af0'

GARDEN_SHA256='cd2622a542c827722e4a834a4c9afeb938b0f17dfd8c1829e35b03d3baead189'
MODULE_SHA256='fa41c99dfb5ee3b8d27eac0c4dc4337d992092498d3f5bb628dea2108e95b7e3'
EXAMPLE_SHA256='15bfa7d7aa4abeb20c013f3999e516d105aaf69c91a3464147466a9f092a07b7'
TEST_SHA256='286ad462509bfa300773a931dfa5485f487a8235554d2f7f56692177dbf3bb5b'
FREEZE_SHA256='662fe6e6f2dc0c5b27f227c26e5d92d1f0da3c588e7ef94accbf4b49b8abd2e5'
CPP_SHA256='af61acf7c91873c6a8c149cfcf436939c86d8dc51c4c0185eda42858525431a1'
CPP_BINARY_SHA256='b6423f58d148fcdb7b31a70085c89c9f927d10176f1bb52381e1dc582c559e16'
RAW_SHA256='b03a091ca2dd725feb05811b094c7405b3ed977b966cac796fbf8e2be1787b67'
RECEIPT_SHA256='cdb14cb23063fee3afe3cbad675cf5aff8aefc413493636b39527ad864b27fa7'
TAMPERED_SHA256='a0237a820dda6137324ff78b950489b6d369d60cd9b89e5d8f21b5454ecc17a5'
EVIDENCE_SHA256='0984716a66c3bba3e5d91104273b831e5c2b851bfadcb3abce3bc56889873d6c'
SOURCE_MANIFEST_SHA256='9363c768fbf8cfc81cf2ce9e64ba35e3f9dc4e85903a924ac91344c9e6fa242b'
SEMANTICS_SHA256='8d89d94c4b808548a9b8827ac6f54c29d510a40674f2421e99b497f8a6f32f05'
PARENT_SEMANTICS_SHA256='e7d4a83e81c054a1d15808292d49fbcda6ea43a06dbf31469e7c4c81d51d3fe5'
PARENT_FREEZE_SHA256='89278d99fab89bc2b582958a27d2806775b0b13a1e8f258550924fc20e3dc05e'
WRAPPER_SHA256='ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008'
COMPILER_SHA256='6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2'

SOUNIO_TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
SOUNIO_HARDWARE_SHA256='46262c5d0fc8df5734998677c2ad063c686fa2d1120fb8dc18dc5b382c7c4805'
WAITING_COMMAND_SHA256='200684c9b9711c38ed3a22e4a9e20b5b033919cdab74da65c49580051554d91d'
WAITING_RESULT_SHA256='926c750db101e39429b524e950ef568225d30059bbbffcf78e86937fd6fea03c'
WAITING_EXAMPLE_OUTPUT_SHA256='663068305319c232725482277993e050d836c224205abf72010ad9758934cb8f'
PARITY_COMMAND_SHA256='9f2891d8a00565eb3d40464b0f4a322177145ac6c41f44de82eeefa392af72c4'
PARITY_RESULT_SHA256='e65dcdf317eeb4187a0443f993de73c500a5d3578ff227c19ac91444c2f2573a'
PARITY_EXAMPLE_OUTPUT_SHA256='8298d49a4eaee33941bd1934fc1ee31e5709de3cd90e9a867301b1a10906a382'
TEST_OUTPUT_SHA256='1c4113c9262f3cccf1c170ca7ebfe0e23e296287da517fab92cb79e7fa5d1df9'

MATERIAL_TOOLCHAIN_SHA256='78174c2f9b0cf932ee3dbd3fdb42eb82d42014dc4d9a0816847c74bf87e62cf5'
MATERIAL_HARDWARE_SHA256='4aaf0347589ac139df5a31d9da7fe1bcbeb29c79a04eef2439e75cd2fcd9d065'
MATERIAL_COMMAND_SHA256='58409a54541ec1739ef4728dd8fb18c220d80f6bd6420b8b2e08c00817b95a8d'
MATERIAL_RESULT_SHA256='bb726bf6690880ac753f9d203820a61ea7bf481999ef4888da778e6185fa5592'

PREEXEC_FRAME_SHA256='0e5e492975b6c07c8bd83b9d7cfa40b84a21f3b3b7a7d8be746525407ea2e0ec'
FREEZE_FRAME_SHA256='aabf4e41b8b9e70c636f26e4e3781571661dad78aab044fb372a212bc38f9b83'
MATERIAL_PREEXEC_FRAME_SHA256='6e4d640440b98f4970ee8e0469d5e8c54e8928f7c5a4f359f94c710f142ecc40'
MATERIAL_SEAL_FRAME_SHA256='167108c554dfeaf0763c5a4f5a5103f07c3fe3dafcc124d53856c8af781644ce'
SOUNIO_SEAL_FRAME_SHA256='eca5184d4bd7383a6731d794b43e348ed96a1e85c3fd8acfa36bac8012d7a5e2'
PYTHON_FRAME_SHA256='077592cd95992740357116b2ad0bad31d73b74ddf2bb042e823d0c400a7ac3ce'
RUST_FRAME_SHA256='e35771cb4d05156284ade88d9b98b9e656eb7ab80e42479b7b37f0c4e9deb432'
LLM_FRAME_SHA256='909cdaa9072f0b510f994ab43582ff8c679019ff046e5ee0f282e61c8ab668f5'
CPP_AUTHORITY_FRAME_SHA256='598ab0a7327b9f25e67394d3be29f5a3a48cd7dcc3dfa5eb0ecea74ca99d8ca7'
POLICY_MISSING_FRAME_SHA256='186fe03cdb6c9b3c4c85b07b43dc0107e984800649b65860f5ca16c9532ff2de'
POLICY_TIMEOUT_FRAME_SHA256='152045f72546deb733132eae04d617bfad84ca8474c0facd004a62f36931da11'
CLAIM_FRAME_SHA256='08274976b724717b678d4b8b8b3dfaed6115a35532b5513094eb0310c65ffd4a'

PYTHON_TOOLCHAIN_SHA256='5c8cfd947420cd48743adb75469089a210d7782421a4e9e46bfc4c40021fb7cf'
PYTHON_COMMAND_SHA256='adedad5e22a944b2a99549bc78377f849e8159d831aecfd7679a356746a06067'
RUST_TOOLCHAIN_SHA256='ae2ed5be700fb051fe99519bd8a9376ad8e4d5091c394ccbb9d8b57c36b7cd11'
RUST_COMMAND_SHA256='b47413e1f12243e68533941a2fdc86080877a56609a0388332cadedce0c2282d'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus U250 xclbin material receipt: FAIL: %s\n' "$*" >&2
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
    "$(sha_limbs "${PARENT_SEMANTICS_SHA256}")" \
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

git -C "${ROOT}" merge-base --is-ancestor "${PARENT_FREEZE_COMMIT}" \
  "${GARDEN_COMMIT}" || fail 'parent artifact was not frozen before Garden'
git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" \
  "${EXECUTABLE_COMMIT}" || fail 'Garden does not precede Sounio executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" \
  "${FREEZE_COMMIT}" || fail 'Sounio executable does not precede freeze'
git -C "${ROOT}" merge-base --is-ancestor "${FREEZE_COMMIT}" \
  "${MATERIAL_COMMIT}" || fail 'material execution preceded Sounio freeze'
git -C "${ROOT}" merge-base --is-ancestor "${MATERIAL_COMMIT}" HEAD ||
  fail 'material receipt commit is not an ancestor of HEAD'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:${MODULE_REL}" | sha256sum | cut -d' ' -f1)" == "${MODULE_SHA256}" ]] ||
  fail 'executable source hash drift'
if git -C "${ROOT}" cat-file -e "${FREEZE_COMMIT}:${CPP_REL}" 2>/dev/null; then
  fail 'C++ material probe existed at Sounio freeze commit'
fi
[[ "$(git -C "${ROOT}" show "${MATERIAL_COMMIT}:${CPP_REL}" | sha256sum | cut -d' ' -f1)" == "${CPP_SHA256}" ]] ||
  fail 'material probe source hash drift'

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
  'compiler_version=g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0' \
  'compiler_sha256=1353e9bdd29a7295c7226bf6c63abccce056d8cac31f112e5cdbecc3f28c2769' \
  "cpp_source_sha256=${CPP_SHA256}" \
  "cpp_binary_sha256=${CPP_BINARY_SHA256}" \
  'metadata_tool_path=/opt/xilinx/xrt/bin/xclbinutil' \
  'metadata_tool_version=XRT-2.23.0' \
  'metadata_tool_sha256=f0755b3a9e4c868f6f95a51b3177614f5f6fe1ef524233d19a67d2b47b778a65')"
[[ "$(sha_text "${material_toolchain_record}")" == "${MATERIAL_TOOLCHAIN_SHA256}" ]] ||
  fail 'material toolchain record drift'
material_hardware_record="$(printf '%s\n' \
  'compiler_host=sounio-workspace-control-0' \
  'compiler_cpu=INTEL(R) XEON(R) GOLD 6526Y' \
  'probe_host=dl380-proxmox' 'probe_os=Linux 7.0.14-8-pve' \
  'probe_cpu=Intel(R) Xeon(R) Gold 6262V CPU @ 1.90GHz' \
  'artifact_origin_host=vitis-u250-builder' \
  'artifact_origin_disk=/dev/pve/vm-100-disk-0' \
  'target=AMD_ALVEO_U250' 'target_serial=22000321B01F' \
  'target_user_bdf=0000:d8:00.1')"
[[ "$(sha_text "${material_hardware_record}")" == "${MATERIAL_HARDWARE_SHA256}" ]] ||
  fail 'material hardware record drift'
material_command_record="$(printf '%s\n' \
  'g++ -std=c++20 -O2 -Wall -Wextra -Werror -static-libstdc++ -static-libgcc -Itools/pireus tools/pireus/u250_xclbin_material_probe.cpp -o /tmp/pireus-u250-xclbin-material-probe' \
  '/tmp/pireus-u250-xclbin-material-probe /tmp/krnl_san_scan.hw.xclbin /opt/xilinx/xrt/bin/xclbinutil')"
[[ "$(sha_text "${material_command_record}")" == "${MATERIAL_COMMAND_SHA256}" ]] ||
  fail 'material command record drift'
material_result_record="$(printf '%s\n' \
  'probe_exit=0' 'artifact_size_bytes=41112056' \
  'artifact_sha256=d30078c7b2e8690aef892b4b6cf96af0f490b70e2b367e5e3679be04fcd4bdbf' \
  "raw_probe_sha256=${RAW_SHA256}" 'raw_probe_lines=40' \
  'probe_valid=true' 'semantic_verdict_emitted=false')"
[[ "$(sha_text "${material_result_record}")" == "${MATERIAL_RESULT_SHA256}" ]] ||
  fail 'material result record drift'

require_line "${FREEZE}" 'stage=SEMANTICS_FROZEN'
require_line "${FREEZE}" 'producing_language=Sounio'
require_line "${FREEZE}" 'language_role=SEMANTIC_AUTHORITY'
require_line "${FREEZE}" "semantics_sha256=${SEMANTICS_SHA256}"
require_line "${FREEZE}" 'cpp_material_probe_compiled=false'
require_line "${FREEZE}" 'cpp_material_probe_executed=false'
require_line "${FREEZE}" 'parity_open=false'
require_line "${RAW}" 'producer_language=C++'
require_line "${RAW}" 'producer_role=MATERIAL_PARITY'
require_line "${RAW}" 'semantic_authority_language=Sounio'
require_line "${RAW}" 'artifact_size_bytes=41112056'
require_line "${RAW}" 'artifact_sha256=d30078c7b2e8690aef892b4b6cf96af0f490b70e2b367e5e3679be04fcd4bdbf'
require_line "${RAW}" 'argument_0=samples,M_AXI_GMEM0,bank1'
require_line "${RAW}" 'argument_1=lut,M_AXI_GMEM1,bank1'
require_line "${RAW}" 'probe_valid=true'
require_line "${RAW}" 'semantic_verdict_emitted=false'
require_line "${RAW}" 'classification_requested=false'
require_line "${RAW}" 'kernel_execution_observed=false'
require_line "${RAW}" 'claim_ready=false'
require_line "${RECEIPT}" 'stage=PARITY_OPEN'
require_line "${RECEIPT}" 'producing_language=C++'
require_line "${RECEIPT}" 'language_role=MATERIAL_PARITY'
require_line "${RECEIPT}" 'semantic_authority_language=Sounio'
require_line "${RECEIPT}" "sounio_semantics_sha256=${SEMANTICS_SHA256}"
require_line "${RECEIPT}" "sounio_freeze_sha256=${FREEZE_SHA256}"
require_line "${RECEIPT}" 'freeze_metadata_correction_scope=commit_identifier_suffixes_only'
require_line "${RECEIPT}" 'semantic_source_changed_after_freeze=false'
require_line "${RECEIPT}" 'material_probe_reexecution_required=false'
require_line "${RECEIPT}" "raw_probe_sha256=${RAW_SHA256}"
require_line "${RECEIPT}" "material_toolchain_sha256=${MATERIAL_TOOLCHAIN_SHA256}"
require_line "${RECEIPT}" "material_hardware_sha256=${MATERIAL_HARDWARE_SHA256}"
require_line "${RECEIPT}" "material_command_sha256=${MATERIAL_COMMAND_SHA256}"
require_line "${RECEIPT}" "material_result_sha256=${MATERIAL_RESULT_SHA256}"
require_line "${RECEIPT}" 'artifact_parity_open=true'
require_line "${RECEIPT}" 'kernel_execution_observed=false'
require_line "${RECEIPT}" 'kernel_correctness_present=false'
require_line "${RECEIPT}" 'lowering_authorized=false'
require_line "${RECEIPT}" 'performance_present=false'
require_line "${RECEIPT}" 'claim_ready=false'
require_line "${EVIDENCE}" "material_receipt_sha256=${RECEIPT_SHA256}"
require_line "${EVIDENCE}" 'tampered_probe=REFUSED'
require_line "${EVIDENCE}" 'python_process_launched=false'
require_line "${EVIDENCE}" 'rust_process_launched=false'
require_line "${EVIDENCE}" 'artifact_imported_to_git=false'
require_line "${EVIDENCE}" 'xclbin_build=false'
require_line "${EVIDENCE}" 'fpga_programming=false'
require_line "${EVIDENCE}" 'kernel_launch=false'

tracked_bits="$(git -C "${ROOT}" ls-files '*.xclbin' '*.xo')"
[[ -z "${tracked_bits}" ]] || fail 'xclbin/xo must remain outside Git'

authorize "$(preexec_frame)" "${PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize "$(freeze_frame 1)" "${FREEZE_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
authorize "$(material_preexec_frame)" "${MATERIAL_PREEXEC_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize "$(material_seal_frame)" "${MATERIAL_SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize "$(sounio_seal_frame)" "${SOUNIO_SEAL_FRAME_SHA256}" 0 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=PARITY_OPEN'
authorize "$(python_frame)" "${PYTHON_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN'
authorize "$(rust_frame)" "${RUST_FRAME_SHA256}" 110 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN'
authorize "$(llm_authority_frame)" "${LLM_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=PARITY_OPEN'
authorize "$(cpp_authority_frame)" "${CPP_AUTHORITY_FRAME_SHA256}" 113 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=113 reason=semantic-authority-required next_stage=PARITY_OPEN'
authorize "$(policy_frame 0)" "${POLICY_MISSING_FRAME_SHA256}" 101 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=101 reason=policy-missing next_stage=SOUNIO_EXECUTABLE'
authorize "$(policy_frame 2)" "${POLICY_TIMEOUT_FRAME_SHA256}" 102 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=102 reason=policy-timeout next_stage=SOUNIO_EXECUTABLE'
authorize "$(claim_frame)" "${CLAIM_FRAME_SHA256}" 122 \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=122 reason=parity-receipt-missing next_stage=PARITY_OPEN'

work="$(mktemp -d "${TMPDIR:-/tmp}/pireus-u250-xclbin-receipt-v0.XXXXXX")"
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" \
    "${CARD_RAW_REL}" "${CARD_RECEIPT_REL}"
) >"${work}/waiting.txt"
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${EXAMPLE_REL}" \
    "${CARD_RAW_REL}" "${CARD_RECEIPT_REL}" "${RAW_REL}"
) >"${work}/parity.txt"
(
  cd "${ROOT}"
  SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run "${TEST_REL}" \
    "${CARD_RAW_REL}" "${CARD_RECEIPT_REL}" "${RAW_REL}" \
    "${TAMPERED_REL}"
) >"${work}/test.txt"
require_hash "${work}/waiting.txt" "${WAITING_EXAMPLE_OUTPUT_SHA256}"
require_hash "${work}/parity.txt" "${PARITY_EXAMPLE_OUTPUT_SHA256}"
require_hash "${work}/test.txt" "${TEST_OUTPUT_SHA256}"
require_line "${work}/waiting.txt" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.u250-xclbin-material-receipt.v0 stage=SOUNIO_EXECUTABLE'
require_line "${work}/waiting.txt" ' status=714090'
require_line "${work}/waiting.txt" ' bits=0'
require_line "${work}/waiting.txt" ' parity_open=0'
require_line "${work}/parity.txt" \
  'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.u250-xclbin-material-receipt.v0 stage=PARITY_OPEN'
require_line "${work}/parity.txt" ' status=714091'
require_line "${work}/parity.txt" ' bits=1'
require_line "${work}/parity.txt" ' digest=1'
require_line "${work}/parity.txt" ' metadata=1'
require_line "${work}/parity.txt" ' parity_open=1'
require_line "${work}/parity.txt" ' execution_observed=0'
require_line "${work}/parity.txt" ' correctness_present=0'
require_line "${work}/parity.txt" ' lowering_authorized=0'
require_line "${work}/parity.txt" ' performance_present=0'
require_line "${work}/parity.txt" ' claim_ready=0'
require_line "${work}/parity.txt" 'PIREUS_U250_XCLBIN_GRAPH triples=442'
require_line "${work}/parity.txt" ' bits_edges=1'
require_line "${work}/parity.txt" ' receipt_edges=1'
require_line "${work}/parity.txt" ' bank_edges=5'
require_line "${work}/test.txt" \
  'PIREUS_U250_XCLBIN_MATERIAL_TEST_PASS waiting=1 parity_projection=1 premature=REFUSED missing_bits=REFUSED missing_digest=REFUSED missing_metadata=REFUSED size_drift=REFUSED digest_drift=REFUSED uuid_drift=REFUSED kernel_drift=REFUSED platform_drift=REFUSED clock_drift=REFUSED order_drift=REFUSED interface_drift=REFUSED bank_drift=REFUSED cpp_authority=REFUSED python_authority=REFUSED rust_authority=REFUSED llm_authority=REFUSED child_verdict=REFUSED execution=REFUSED correctness=REFUSED operation=REFUSED lowering=REFUSED performance=REFUSED claim_ready=REFUSED'

printf 'PIREUS_U250_XCLBIN_MATERIAL_RECEIPT_GATE_PASS=true stage=PARITY_OPEN artifact=krnl_san_scan.hw.xclbin artifact_sha256=d30078c7b2e8690aef892b4b6cf96af0f490b70e2b367e5e3679be04fcd4bdbf size_bytes=41112056 abi_arguments=8 interfaces=8 bank_bindings=5 triples=442 bits_edges=1 receipt_edges=1 bank_edges=5 tampered=REFUSED python_oracle=E110 rust_oracle=E110 python_process_launched=false rust_process_launched=false llm_authority=E113 cpp_authority=E113 policy_missing=E101 policy_timeout=E102 claim_promotion=E122 xclbin_build=false fpga_programming=false kernel_launch=false execution_observed=false correctness_present=false lowering_authorized=false performance_present=false claim_ready=false\n'
