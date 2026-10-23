#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
NAMESPACE="${PIREUS_T560_NAMESPACE:-beagle}"
POD="${PIREUS_T560_POD:-node-ephemeral-governance-kp96t}"
KEY="${PIREUS_T560_KEY:-/host/home/devsounio/.ssh/id_ed25519}"
APPLE_ADDRESS="${PIREUS_APPLE_ADDRESS:-100.91.184.41}"
APPLE_USER="${PIREUS_APPLE_USER:-demetriosagourakis}"
DGX_ADDRESS="${PIREUS_DGX_ADDRESS:-192.168.3.24}"
DGX_USER="${PIREUS_DGX_USER:-demetrios}"
ZERO='0 0 0 0 0 0 0 0'

HEADER="${ROOT}/tools/pireus/material_sha256.hpp"
APPLE_SOURCE="${ROOT}/tools/pireus/apple_a64_tbl_material_parity.cpp"
DGX_SOURCE="${ROOT}/tools/pireus/dgx_ptx_shfl_material_parity.cu"

HEADER_SHA256='ae54c8f455d5ef057f182212aacd466bdf5e014898872706e80e51f6b16e7782'
APPLE_CPP_SHA256='7bb640be1093b3add99961c43d7c53da276cceacef70b7ee9e2002c961c5d66e'
DGX_CPP_SHA256='6820fa05ff91cb89012bb0a7651896e196d5ff379be8f90afdc0b8ae08a8688a'

APPLE_SOUNIO_SOURCE_SHA256='79c2e859ffe81f3add1ebb36608a5995672c10a5c1645ec4500a03fcd9bcd031'
APPLE_SEMANTICS_SHA256='377aed20ffd302aeb3ff71f6609643f17d2a9983129e319d5545b81c589dc3e6'
APPLE_TOOLCHAIN_SHA256='2e20f3f44c17d6fc4c1e58b26c38cf3af1ea2df887778d0aa723e6ddfe4b72e1'
APPLE_HARDWARE_SHA256='49702cf6d0b079bf52bf26f98f377266e41d4ce232fea99eb80c30d6554dbc28'
APPLE_COMMAND_SHA256='e290c25bf7cf3d5c47d3c255d11ab89d8a5eba775a63685fbe0b981cfd76bff5'
APPLE_RESULT_SHA256='bd64bc56037a64a93c0136fa29a6ff1a294e8b84be549a5a4abfeaaf81a2e700'

DGX_SOUNIO_SOURCE_SHA256='4be23864a14274d7996dd890473a5b3356a88441a589e509080c9978ba1cf404'
DGX_SEMANTICS_SHA256='a163f5924428de0f8f2a33a54ea864d82bfab753cf80dc04b8c9698c4a225336'
DGX_TOOLCHAIN_SHA256='10ab88d927b1285a5ccc6e717c5beb721d76f6e074c4d0c1e9d2f36072c57cf5'
DGX_HARDWARE_SHA256='8b048f0a20ac0967af5622606935aa4ea4e6caf0baef6a3dcd9b7ff58f2a66d4'
DGX_COMMAND_SHA256='cf1796b78caaedc7866e26ba2885cd2fdc3224bdd948e40f8b72fea092223a72'
DGX_RESULT_SHA256='1e776e655761bd9e59322ac64e736629bd35586a958d503ea814ddddaa865f3c'

fail() {
  printf 'pireus canonical target material parity: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() {
  sha256sum "$1" | cut -d' ' -f1
}

sha_text() {
  printf '%s' "$1" | sha256sum | cut -d' ' -f1
}

sha_limbs() {
  local hex="$1"
  local out='' i part
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

authority_frame() {
  local stage="$1" action="$2" source="$3" semantics="$4"
  local toolchain="$5" hardware="$6" command="$7" result="$8"
  local receipt_valid="$9"
  printf '9020 %s %s 4 4 1 0 0 %s 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${stage}" "${action}" "${receipt_valid}" \
    "$(sha_limbs "${source}")" "$(sha_limbs "${semantics}")" \
    "$(sha_limbs "${semantics}")" "$(sha_limbs "${toolchain}")" \
    "$(sha_limbs "${hardware}")" "$(sha_limbs "${command}")" \
    "${result}" "${ZERO}"
}

guardian_allow() {
  local frame="$1" expected_stage="$2"
  local decision
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == "SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=${expected_stage}" ]] || \
    fail "Loom refused action: ${decision}"
  printf '%s' "${decision}"
}

ssh_target() {
  local user="$1" address="$2" command="$3"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- \
    nsenter -t 1 -n /usr/bin/ssh -i "${KEY}" \
    -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${user}@${address}" "${command}"
}

copy_to_target() {
  local user="$1" address="$2" remote="$3" stage="$4"
  shift 4
  kubectl -n "${NAMESPACE}" exec "${POD}" -- rm -rf "${stage}"
  kubectl -n "${NAMESPACE}" exec "${POD}" -- mkdir -p "${stage}"
  local source
  for source in "$@"; do
    kubectl -n "${NAMESPACE}" cp "${source}" \
      "${POD}:${stage}/$(basename "${source}")"
  done
  ssh_target "${user}" "${address}" "rm -rf '${remote}' && mkdir -p '${remote}'"
  local staged=()
  for source in "$@"; do
    staged+=("${stage}/$(basename "${source}")")
  done
  kubectl -n "${NAMESPACE}" exec "${POD}" -- \
    nsenter -t 1 -n /usr/bin/scp -i "${KEY}" \
    -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "${staged[@]}" "${user}@${address}:${remote}/"
}

apple_command_record() {
  printf '%s\n' \
    'schema=pireus-apple-a64-tbl-parity-command.v1' \
    'action=PARITY_EXECUTE' \
    'transport=kubectl/nsenter/ssh' \
    'target=sounio-language-macbook@100.91.184.41' \
    'remote_dir=/tmp/pireus-apple-a64-tbl-material-parity-20260827' \
    "cpp_sha256=${APPLE_CPP_SHA256}" \
    "header_sha256=${HEADER_SHA256}" \
    'compile=xcrun clang++ -std=c++20 -O3 -fno-fast-math -fno-associative-math -ffp-contract=off -Wall -Wextra -Werror -arch arm64 -I. apple_a64_tbl_material_parity.cpp -o apple_a64_tbl_material_parity' \
    'assembly=xcrun clang++ -std=c++20 -O3 -fno-fast-math -fno-associative-math -ffp-contract=off -Wall -Wextra -Werror -arch arm64 -I. -S apple_a64_tbl_material_parity.cpp -o apple_a64_tbl_material_parity.s' \
    'disassemble=xcrun llvm-objdump -d apple_a64_tbl_material_parity' \
    'execute=./apple_a64_tbl_material_parity'
}

dgx_command_record() {
  printf '%s\n' \
    'schema=pireus-dgx-ptx-shfl-parity-command.v1' \
    'action=PARITY_EXECUTE' \
    'transport=kubectl/nsenter/ssh' \
    'target=demetrios@192.168.3.24' \
    'remote_dir=/tmp/pireus-dgx-ptx-shfl-material-parity-20260827' \
    "cu_sha256=${DGX_CPP_SHA256}" \
    "header_sha256=${HEADER_SHA256}" \
    'compile=/usr/local/cuda-13.0/bin/nvcc -std=c++20 -O3 -arch=sm_121 -lineinfo -I. dgx_ptx_shfl_material_parity.cu -o dgx_ptx_shfl_material_parity' \
    'ptx=/usr/local/cuda-13.0/bin/nvcc -std=c++20 -O3 -arch=sm_121 -I. --ptx dgx_ptx_shfl_material_parity.cu -o dgx_ptx_shfl_material_parity.ptx' \
    'sass=/usr/local/cuda-13.0/bin/cuobjdump --dump-sass dgx_ptx_shfl_material_parity' \
    'execute=./dgx_ptx_shfl_material_parity'
}

run_apple() {
  local remote='/tmp/pireus-apple-a64-tbl-material-parity-20260827'
  local stage='/tmp/pireus-apple-stage-20260827'
  local build result result_hash artifacts tbl_sites frame decision seal_frame seal_decision

  [[ "$(sha_file "${HEADER}")" == "${HEADER_SHA256}" ]] || fail 'shared header hash drift'
  [[ "$(sha_file "${APPLE_SOURCE}")" == "${APPLE_CPP_SHA256}" ]] || fail 'Apple C++ source hash drift'
  [[ "$(sha_text "$(apple_command_record)")" == "${APPLE_COMMAND_SHA256}" ]] || fail 'Apple command record drift'
  [[ -x "${GUARDIAN}" ]] || fail "Loom Guardian unavailable: ${GUARDIAN}"

  local identity
  identity="$(ssh_target "${APPLE_USER}" "${APPLE_ADDRESS}" \
    'hostname; uname -sr; uname -m; sysctl -n hw.model; sysctl -n machdep.cpu.brand_string; sysctl -n hw.targettype; clang --version | head -n 1; xcodebuild -version')"
  [[ "${identity}" == $'Sounio-Language-MacBook\nDarwin 27.0.0\narm64\nMac17,7\nApple M5 Max\nJ714c\nApple clang version 21.0.0 (clang-2100.3.27.1)\nXcode 27.0\nBuild version 27A5228h' ]] || \
    fail "Apple target identity drift: ${identity}"

  frame="$(authority_frame 3 4 "${APPLE_SOUNIO_SOURCE_SHA256}" \
    "${APPLE_SEMANTICS_SHA256}" "${APPLE_TOOLCHAIN_SHA256}" \
    "${APPLE_HARDWARE_SHA256}" "${APPLE_COMMAND_SHA256}" "${ZERO}" 0)"
  decision="$(guardian_allow "${frame}" PARITY_OPEN)"

  copy_to_target "${APPLE_USER}" "${APPLE_ADDRESS}" "${remote}" "${stage}" \
    "${HEADER}" "${APPLE_SOURCE}"
  build="cd '${remote}' && xcrun clang++ -std=c++20 -O3 -fno-fast-math -fno-associative-math -ffp-contract=off -Wall -Wextra -Werror -arch arm64 -I. apple_a64_tbl_material_parity.cpp -o apple_a64_tbl_material_parity && xcrun clang++ -std=c++20 -O3 -fno-fast-math -fno-associative-math -ffp-contract=off -Wall -Wextra -Werror -arch arm64 -I. -S apple_a64_tbl_material_parity.cpp -o apple_a64_tbl_material_parity.s && xcrun llvm-objdump -d apple_a64_tbl_material_parity > apple_a64_tbl_material_parity.objdump && ./apple_a64_tbl_material_parity"
  result="$(ssh_target "${APPLE_USER}" "${APPLE_ADDRESS}" "${build}")"
  result_hash="$(sha_text "${result}"$'\n')"
  [[ "${result_hash}" == "${APPLE_RESULT_SHA256}" ]] || fail "Apple result hash drift: ${result_hash}"
  [[ "${result}" == *$'\nresult=PASS' ]] || fail 'Apple material comparator failed'
  artifacts="$(ssh_target "${APPLE_USER}" "${APPLE_ADDRESS}" \
    "cd '${remote}' && shasum -a 256 apple_a64_tbl_material_parity apple_a64_tbl_material_parity.s apple_a64_tbl_material_parity.objdump")"
  [[ "${artifacts}" == $'299a41090348d47b518929af6dab8137ada9032a046def3e8052a5d721c6fd71  apple_a64_tbl_material_parity\nf3c3216faab1809b20e419c4ed345eaf658e892a5f1e6d4545fc6d095d699f76  apple_a64_tbl_material_parity.s\n0a05d0ba5562260c062c6338c4b9818df9cc711d0bcbc4f93ef536c77617fab8  apple_a64_tbl_material_parity.objdump' ]] || \
    fail "Apple artifact hash drift: ${artifacts}"
  tbl_sites="$(ssh_target "${APPLE_USER}" "${APPLE_ADDRESS}" \
    "cd '${remote}' && grep -c 'tbl.16b' apple_a64_tbl_material_parity.s")"
  [[ "${tbl_sites}" == 1 ]] || fail "Apple TBL static site count drift: ${tbl_sites}"

  seal_frame="$(authority_frame 4 8 "${APPLE_SOUNIO_SOURCE_SHA256}" \
    "${APPLE_SEMANTICS_SHA256}" "${APPLE_TOOLCHAIN_SHA256}" \
    "${APPLE_HARDWARE_SHA256}" "${APPLE_COMMAND_SHA256}" \
    "$(sha_limbs "${result_hash}")" 1)"
  seal_decision="$(guardian_allow "${seal_frame}" PARITY_OPEN)"
  printf 'target=apple_m5_max\nparity_open_frame_sha256=%s\nparity_open_decision=%s\nresult_sha256=%s\nreceipt_seal_frame_sha256=%s\nreceipt_seal_decision=%s\ntbl_static_sites=%s\nresult=PASS\n' \
    "$(sha_text "${frame}"$'\n')" "${decision}" "${result_hash}" \
    "$(sha_text "${seal_frame}"$'\n')" "${seal_decision}" "${tbl_sites}"
}

run_dgx() {
  local remote='/tmp/pireus-dgx-ptx-shfl-material-parity-20260827'
  local stage='/tmp/pireus-dgx-stage-20260827'
  local build result result_hash artifacts binary_hash ptx_sites ptx_c_sites sass_sites frame decision seal_frame seal_decision

  [[ "$(sha_file "${HEADER}")" == "${HEADER_SHA256}" ]] || fail 'shared header hash drift'
  [[ "$(sha_file "${DGX_SOURCE}")" == "${DGX_CPP_SHA256}" ]] || fail 'DGX CUDA C++ source hash drift'
  [[ "$(sha_text "$(dgx_command_record)")" == "${DGX_COMMAND_SHA256}" ]] || fail 'DGX command record drift'
  [[ -x "${GUARDIAN}" ]] || fail "Loom Guardian unavailable: ${GUARDIAN}"

  local identity
  identity="$(ssh_target "${DGX_USER}" "${DGX_ADDRESS}" \
    'hostname; uname -sr; uname -m; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader; /usr/local/cuda-13.0/bin/nvcc --version | tail -n 1')"
  [[ "${identity}" == $'spark-3c59\nLinux 6.17.0-1021-nvidia\naarch64\nNVIDIA GB10, 580.159.03, 12.1\nBuild cuda_13.0.r13.0/compiler.36424714_0' ]] || \
    fail "DGX target identity drift: ${identity}"

  frame="$(authority_frame 3 4 "${DGX_SOUNIO_SOURCE_SHA256}" \
    "${DGX_SEMANTICS_SHA256}" "${DGX_TOOLCHAIN_SHA256}" \
    "${DGX_HARDWARE_SHA256}" "${DGX_COMMAND_SHA256}" "${ZERO}" 0)"
  decision="$(guardian_allow "${frame}" PARITY_OPEN)"

  copy_to_target "${DGX_USER}" "${DGX_ADDRESS}" "${remote}" "${stage}" \
    "${HEADER}" "${DGX_SOURCE}"
  build="cd '${remote}' && /usr/local/cuda-13.0/bin/nvcc -std=c++20 -O3 -arch=sm_121 -lineinfo -I. dgx_ptx_shfl_material_parity.cu -o dgx_ptx_shfl_material_parity && /usr/local/cuda-13.0/bin/nvcc -std=c++20 -O3 -arch=sm_121 -I. --ptx dgx_ptx_shfl_material_parity.cu -o dgx_ptx_shfl_material_parity.ptx && /usr/local/cuda-13.0/bin/cuobjdump --dump-sass dgx_ptx_shfl_material_parity > dgx_ptx_shfl_material_parity.sass && ./dgx_ptx_shfl_material_parity"
  result="$(ssh_target "${DGX_USER}" "${DGX_ADDRESS}" "${build}")"
  result_hash="$(sha_text "${result}"$'\n')"
  [[ "${result_hash}" == "${DGX_RESULT_SHA256}" ]] || fail "DGX result hash drift: ${result_hash}"
  [[ "${result}" == *$'\nresult=PASS' ]] || fail 'DGX material comparator failed'
  artifacts="$(ssh_target "${DGX_USER}" "${DGX_ADDRESS}" \
    "cd '${remote}' && sha256sum dgx_ptx_shfl_material_parity dgx_ptx_shfl_material_parity.ptx dgx_ptx_shfl_material_parity.sass")"
  [[ "${artifacts}" == *$'  dgx_ptx_shfl_material_parity\n480c3de12dd2e77b5c29e4f0b889e282fa8c4e0dd1147a151a039b2749db2d2f  dgx_ptx_shfl_material_parity.ptx\n5f34ba10b94797219b128522f1edd34bdbe2f915b67b2c75412c293d81a299f4  dgx_ptx_shfl_material_parity.sass' ]] || \
    fail "DGX PTX/SASS artifact hash drift: ${artifacts}"
  binary_hash="${artifacts%% *}"
  ptx_sites="$(ssh_target "${DGX_USER}" "${DGX_ADDRESS}" \
    "cd '${remote}' && grep -c 'shfl.sync.bfly.b32' dgx_ptx_shfl_material_parity.ptx")"
  ptx_c_sites="$(ssh_target "${DGX_USER}" "${DGX_ADDRESS}" \
    "cd '${remote}' && grep -c ', 4127, 65535;' dgx_ptx_shfl_material_parity.ptx")"
  sass_sites="$(ssh_target "${DGX_USER}" "${DGX_ADDRESS}" \
    "cd '${remote}' && grep -c 'SHFL.BFLY' dgx_ptx_shfl_material_parity.sass")"
  [[ "${ptx_sites}" == 32 && "${ptx_c_sites}" == 32 && "${sass_sites}" == 32 ]] || \
    fail "DGX SHFL site count drift: PTX=${ptx_sites} PTX_C_4127=${ptx_c_sites} SASS=${sass_sites}"

  seal_frame="$(authority_frame 4 8 "${DGX_SOUNIO_SOURCE_SHA256}" \
    "${DGX_SEMANTICS_SHA256}" "${DGX_TOOLCHAIN_SHA256}" \
    "${DGX_HARDWARE_SHA256}" "${DGX_COMMAND_SHA256}" \
    "$(sha_limbs "${result_hash}")" 1)"
  seal_decision="$(guardian_allow "${seal_frame}" PARITY_OPEN)"
  printf 'target=dgx_gb10\nparity_open_frame_sha256=%s\nparity_open_decision=%s\nresult_sha256=%s\nreceipt_seal_frame_sha256=%s\nreceipt_seal_decision=%s\nbinary_sha256=%s\nbinary_reproducible=false\nptx_shfl_static_sites=%s\nemitted_ptx_c=4127\nemitted_ptx_c_hex=0x101f\nptx_c_4127_static_sites=%s\nsass_shfl_static_sites=%s\nresult=PASS\n' \
    "$(sha_text "${frame}"$'\n')" "${decision}" "${result_hash}" \
    "$(sha_text "${seal_frame}"$'\n')" "${seal_decision}" "${binary_hash}" \
    "${ptx_sites}" "${ptx_c_sites}" "${sass_sites}"
}

negative_python() {
  local frame decision rc
  frame="9020 4 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 $(sha_limbs "${DGX_SOUNIO_SOURCE_SHA256}") $(sha_limbs "${DGX_SEMANTICS_SHA256}") $(sha_limbs "${DGX_SEMANTICS_SHA256}") $(sha_limbs "${DGX_TOOLCHAIN_SHA256}") $(sha_limbs "${DGX_HARDWARE_SHA256}") $(sha_limbs "${DGX_COMMAND_SHA256}") ${ZERO} ${ZERO}"
  set +e
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  rc=$?
  set -e
  [[ "${rc}" == 110 ]] || fail "Python request exit drift: ${rc}"
  [[ "${decision}" == 'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=PARITY_OPEN' ]] || \
    fail "Python request decision drift: ${decision}"
  printf 'python_frame_sha256=%s\npython_decision=%s\ninterpreter_launch_count=0\n' \
    "$(sha_text "${frame}"$'\n')" "${decision}"
}

case "${1:-all}" in
  apple)
    run_apple
    ;;
  dgx)
    run_dgx
    ;;
  all)
    run_apple
    run_dgx
    negative_python
    printf 'PARITY_OPEN=true\nCLAIM_READY=false\nPIREUS_CANONICAL_TARGET_MATERIAL_PARITY_PASS=true\n'
    ;;
  *)
    fail "unknown target: ${1}"
    ;;
esac
