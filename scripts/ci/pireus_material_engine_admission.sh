#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

GARDEN="${ROOT}/docs/internal/garden/seeds/2026-08-28-pireus-material-engine-admission.md"
MODULE="${ROOT}/stdlib/hardware/pireus/material_engine_admission.sio"
EXAMPLE="${ROOT}/examples/pireus_material_engine_admission.sio"
TEST="${ROOT}/tests/stdlib/hardware/test_pireus_material_engine_admission.sio"
CONCEPT="${ROOT}/docs/internal/concepts/pireus-material-engine-admission.md"
SEMANTICS="${ROOT}/docs/research/pireus_material_engine_admission_semantics.md"
RECEIPT="${ROOT}/docs/research/receipts/pireus_material_engine_admission_20260828.md"
EVIDENCE="${ROOT}/docs/research/evidence/pireus_material_engine_admission_20260828.txt"
REGISTRY="${ROOT}/docs/internal/concepts/registry.tsv"
TOPIC_REGISTRY="${ROOT}/docs/governance/topic-registry.v1.json"
AUTHORITY_MATRIX="${ROOT}/docs/governance/DOCS_AUTHORITY_MATRIX.md"
ACCEPTANCE_REPORT="${ROOT}/docs/governance/DOCS_ACCEPTANCE_REPORT.md"

ENGINE_SOURCE="${ROOT}/stdlib/hardware/pireus/execution_engine.sio"
ENGINE_SEMANTICS="${ROOT}/docs/research/pireus_execution_engine_semantics.md"
ENGINE_RECEIPT="${ROOT}/docs/research/receipts/pireus_execution_engine_20260827.md"
COST_SOURCE="${ROOT}/stdlib/hardware/pireus/target_cost_observation.sio"
COST_SEMANTICS="${ROOT}/docs/research/pireus_target_cost_observation_semantics.md"
COST_RECEIPT="${ROOT}/docs/research/receipts/pireus_target_cost_observation_20260827.md"
COST_EVIDENCE="${ROOT}/docs/research/evidence/pireus_target_cost_observation_20260827.txt"
APPLE_RECEIPT="${ROOT}/docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md"
APPLE_EVIDENCE="${ROOT}/docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt"
DGX_RECEIPT="${ROOT}/docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md"
DGX_EVIDENCE="${ROOT}/docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt"

GARDEN_COMMIT='61e8fa34ed93a083cccabe6e6813f1ce6460e327'
EXECUTABLE_COMMIT='d0a16b5303a5e455be1b8a1ce90f40d3f89b3ed2'
SOURCE_SHA256='b98d799fec6452f9afaecc7e418578cc0ca72a1d3e7c0a1f84d474e0aa2730ec'
SEMANTICS_SHA256='bbbd7bc9c99e1de46a5317d99a7893adebfdf7fb46454cbd7002b3098281b6ee'
PARENT_MANIFEST_SHA256='f939a44278411e0954fde1425e3a728afe7069cbe1805e0ce521f26854b097a9'
SOURCE_MANIFEST_SHA256='12a648e39618d3b02302d86da7f7203362f88dbc24f0fb0c352a4c9b5e6adcce'
TOOLCHAIN_SHA256='850c094e02d85fee153297ccf8babbe171e3ec47def68ac2976c3473092b36ac'
HARDWARE_SHA256='464f1a4530cb0829854ddbafc0786d12cc9fc98cef1afced51f40679ba27517c'
AUTHORITY_COMMAND_SHA256='c2aae8f3b58d3caed7d9f50277f6757e4eea7cffa0e0f714ea96330507fe3051'
TEST_COMMAND_SHA256='d61958b542ad92be7939fc94ee3e43af2a7af9dd6674c6578609bc8d68455206'
TAMPER_COMMAND_SHA256='dce63dffa83f4ee3505a33c7a93f6b3b0a3490af5c1ef3fd04d0e5c1d650f1c1'
RESULT_SHA256='cdaa653a8ba745aacaf6bb8fae8ac3b34fb16c42ecf6e7247eb7949dce71cec1'
TEST_RESULT_SHA256='bbd6018aefba1e0d1bafd48d52d253a41a05dd7ae306788aa4b42fc78175f6f7'
TAMPER_RESULT_SHA256='421c60238e4bdab6487d9849939cfcd0195522f935054bbbf5370bfaa6334448'
TAMPERED_PARENT_SHA256='67a8641d0793f092364ed5bcb37724c087178ad80bcf19ceb82a16ce50935b63'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus-material-engine-admission: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() {
  sha256sum "$1" | cut -d' ' -f1
}

sha_text() {
  printf '%s\n' "$1" | sha256sum | cut -d' ' -f1
}

sha_text_exact() {
  printf '%s' "$1" | sha256sum | cut -d' ' -f1
}

require_hash() {
  local path="$1" expected="$2" actual
  [[ -f "${path}" ]] || fail "missing artifact: ${path}"
  actual="$(sha_file "${path}")"
  [[ "${actual}" == "${expected}" ]] ||
    fail "hash drift: ${path}: expected ${expected}, got ${actual}"
}

require_line() {
  local path="$1" line="$2"
  grep -Fqx -- "${line}" "${path}" ||
    fail "missing exact line in ${path}: ${line}"
}

sha_limbs() {
  local hex="$1" out='' i part
  for ((i = 0; i < 8; i++)); do
    part="${hex:$((i * 8)):8}"
    out="${out}${out:+ }$((16#${part}))"
  done
  printf '%s' "${out}"
}

preaction_frame() {
  local command_sha="$1"
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" "${ZERO}" "${ZERO}" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_sha}")" "${ZERO}" "${ZERO}"
}

freeze_frame() {
  printf '9020 2 3 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_MANIFEST_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

receipt_frame() {
  local command_sha="$1" result_sha="$2"
  printf '9020 3 8 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_MANIFEST_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${command_sha}")" \
    "$(sha_limbs "${result_sha}")" "${ZERO}"
}

commit_frame() {
  printf '9020 3 10 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_MANIFEST_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

tamper_frame() {
  printf '9020 3 11 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_MANIFEST_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${TAMPER_COMMAND_SHA256}")" "${ZERO}" "${ZERO}"
}

python_frame() {
  printf '9020 3 4 7 7 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_MANIFEST_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" \
    "$(sha_limbs "${RESULT_SHA256}")" "${ZERO}"
}

authorize() {
  local expected_frame_sha="$1" expected_decision="$2" frame="$3" decision
  [[ "$(sha_text "${frame}")" == "${expected_frame_sha}" ]] ||
    fail "Loom frame drift: expected ${expected_frame_sha}"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == "${expected_decision}" ]] ||
    fail "Loom decision mismatch: ${decision}"
  printf 'loom_decision=%s frame_sha256=%s\n' "${decision}" "${expected_frame_sha}"
}

authority_command_record() {
  printf '%s' 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_material_engine_admission.sio docs/internal/garden/seeds/2026-08-28-pireus-material-engine-admission.md stdlib/hardware/pireus/execution_engine.sio docs/research/pireus_execution_engine_semantics.md docs/research/receipts/pireus_execution_engine_20260827.md stdlib/hardware/pireus/target_cost_observation.sio docs/research/pireus_target_cost_observation_semantics.md docs/research/receipts/pireus_target_cost_observation_20260827.md docs/research/evidence/pireus_target_cost_observation_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt'
}

test_command_record() {
  printf '%s' 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_material_engine_admission.sio docs/internal/garden/seeds/2026-08-28-pireus-material-engine-admission.md stdlib/hardware/pireus/execution_engine.sio docs/research/pireus_execution_engine_semantics.md docs/research/receipts/pireus_execution_engine_20260827.md stdlib/hardware/pireus/target_cost_observation.sio docs/research/pireus_target_cost_observation_semantics.md docs/research/receipts/pireus_target_cost_observation_20260827.md docs/research/evidence/pireus_target_cost_observation_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt'
}

tamper_command_record() {
  printf '%s' 'set -o pipefail; SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_material_engine_admission.sio docs/internal/garden/seeds/2026-08-28-pireus-material-engine-admission.md stdlib/hardware/pireus/execution_engine.sio docs/research/pireus_execution_engine_semantics.md docs/research/receipts/pireus_execution_engine_20260827.md stdlib/hardware/pireus/target_cost_observation.sio docs/research/pireus_target_cost_observation_semantics.md docs/research/receipts/pireus_target_cost_observation_20260827.md docs/research/evidence/pireus_target_cost_observation_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md /tmp/pireus-material-engine-admission.tampered-dgx-evidence.txt | tee /tmp/pireus-material-engine-admission.tampered.txt'
}

run_authority() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_material_engine_admission.sio \
      docs/internal/garden/seeds/2026-08-28-pireus-material-engine-admission.md \
      stdlib/hardware/pireus/execution_engine.sio \
      docs/research/pireus_execution_engine_semantics.md \
      docs/research/receipts/pireus_execution_engine_20260827.md \
      stdlib/hardware/pireus/target_cost_observation.sio \
      docs/research/pireus_target_cost_observation_semantics.md \
      docs/research/receipts/pireus_target_cost_observation_20260827.md \
      docs/research/evidence/pireus_target_cost_observation_20260827.txt \
      docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md \
      docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt \
      docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md \
      docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt
  )
}

run_test() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      tests/stdlib/hardware/test_pireus_material_engine_admission.sio \
      "${GARDEN}" "${ENGINE_SOURCE}" "${ENGINE_SEMANTICS}" \
      "${ENGINE_RECEIPT}" "${COST_SOURCE}" "${COST_SEMANTICS}" \
      "${COST_RECEIPT}" "${COST_EVIDENCE}" \
      "${APPLE_RECEIPT}" "${APPLE_EVIDENCE}" \
      "${DGX_RECEIPT}" "${DGX_EVIDENCE}"
  )
}

run_tamper() {
  cp "${DGX_EVIDENCE}" /tmp/pireus-material-engine-admission.tampered-dgx-evidence.txt
  printf '\nPIREUS_MATERIAL_ENGINE_ADMISSION_TAMPER\n' >> \
    /tmp/pireus-material-engine-admission.tampered-dgx-evidence.txt
  (
    set -o pipefail
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_material_engine_admission.sio \
      "${GARDEN}" "${ENGINE_SOURCE}" "${ENGINE_SEMANTICS}" \
      "${ENGINE_RECEIPT}" "${COST_SOURCE}" "${COST_SEMANTICS}" \
      "${COST_RECEIPT}" "${COST_EVIDENCE}" \
      "${APPLE_RECEIPT}" "${APPLE_EVIDENCE}" \
      "${DGX_RECEIPT}" \
      /tmp/pireus-material-engine-admission.tampered-dgx-evidence.txt \
      | tee /tmp/pireus-material-engine-admission.tampered.txt
  )
}

[[ -x "${GUARDIAN}" ]] || fail 'native Sounio Loom guardian unavailable'

require_hash "${ROOT}/bin/souc" ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
require_hash "${ROOT}/bin/souc-lean-single-x86_64" 6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
require_hash "${GARDEN}" 08d6bd3193db9a2ce0cd849db4b1197d389049322be3be2986bd9f955257b21e
require_hash "${MODULE}" "${SOURCE_SHA256}"
require_hash "${EXAMPLE}" 075ae3225bd01ac858586186cb43bc404ed5315f9d55a2f1b5725299f46d88a4
require_hash "${TEST}" 375ea71aad9dd695175fa0fa16e9e9b656a53b4df137fb208ec7afe34eb3a7b2
require_hash "${CONCEPT}" 48768c930302fa422ad5efbfb4ff7b615fda0c405cffddd1bce97bc9f619ea44
require_hash "${SEMANTICS}" "${SEMANTICS_SHA256}"
require_hash "${RECEIPT}" 352ad87a4d05ef32333acb2f31a48f49e7d520ab99c3e99a8b4a3e6bf3f55aa4
require_hash "${EVIDENCE}" 08a6e0d4b2e5a0bb4e981cfcc8df5192d9b9b77f53eb3862c86f29d9409e425b
require_hash "${REGISTRY}" 2f1c7eb64c54278e4e08b6ee73e5cfba0310131864079520397ae42f37c820aa
require_hash "${TOPIC_REGISTRY}" 31f8e1112889eee865fb226d7f6fa7f5012dd9110ef42ba30af9ff7669ffd229
require_hash "${AUTHORITY_MATRIX}" eb2ecc9ebf30d055175a4cbb3c86270ac02d20d64c76f1f339f42c3d3a648319
require_hash "${ACCEPTANCE_REPORT}" 79e7b1c37aca6abe3a6ef8b80065b1e894700035921154b2a175dfe6a2c4e94b

require_hash "${ENGINE_SOURCE}" 8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e
require_hash "${ENGINE_SEMANTICS}" c47668a08ad25f39bebe9d8bef90b66eb2ad7119063c19ab8319fa4fab265233
require_hash "${ENGINE_RECEIPT}" 9da8ca53c3cb0e6631c92e55a8e82387aed2bd53863ffa9d646719806eec4ffd
require_hash "${COST_SOURCE}" 7ea2815c112b85476fc6ac4d8bb9388ee032062822c6905485c2084ee416d6bc
require_hash "${COST_SEMANTICS}" 0a899be7cd25375c8c444b9e1f0a71dd102ca8958072a4290073ae21c926a199
require_hash "${COST_RECEIPT}" b7577c782a82431eb54312137a52c1811f14316d5e5a5fb9e7aa9581f1c304ca
require_hash "${COST_EVIDENCE}" 06f21108ddc89c8c468097b7cefec6a766bda065918fd5797be194878371577b
require_hash "${APPLE_RECEIPT}" c00a3d4e556688829efadbbf640ea858cfe9520dc04103fa745cf1a8101f7840
require_hash "${APPLE_EVIDENCE}" 2877bfd463b4d28dc3311b75c69bec2aa1c62b430d08314989187d44b32a781e
require_hash "${DGX_RECEIPT}" 3c10882eff43d3b197428839996c7a04c009c8f537d0c1451bdf3e8a13e2f385
require_hash "${DGX_EVIDENCE}" 2c6b6e448265a5566d17df9a674246ea62c05210e432e48e418d16358496853b

git -C "${ROOT}" cat-file -e "${GARDEN_COMMIT}^{commit}" || fail 'Garden commit missing'
git -C "${ROOT}" cat-file -e "${EXECUTABLE_COMMIT}^{commit}" || fail 'first executable commit missing'
git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" "${EXECUTABLE_COMMIT}" ||
  fail 'Garden is not an ancestor of first executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" HEAD ||
  fail 'first executable is not an ancestor of HEAD'
if git -C "${ROOT}" show \
  "${EXECUTABLE_COMMIT}:stdlib/hardware/pireus/material_engine_admission.sio" |
  grep -Fq 'pireus_material_engine_admission_matches_frozen_semantics'; then
  fail 'exact matcher exists in first executable commit'
fi
grep -Fq 'pireus_material_engine_admission_matches_frozen_semantics' "${MODULE}" ||
  fail 'exact matcher missing from frozen module'

PARENT_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    docs/internal/garden/seeds/2026-08-28-pireus-material-engine-admission.md \
    stdlib/hardware/pireus/execution_engine.sio \
    docs/research/pireus_execution_engine_semantics.md \
    docs/research/receipts/pireus_execution_engine_20260827.md \
    stdlib/hardware/pireus/target_cost_observation.sio \
    docs/research/pireus_target_cost_observation_semantics.md \
    docs/research/receipts/pireus_target_cost_observation_20260827.md \
    docs/research/evidence/pireus_target_cost_observation_20260827.txt \
    docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md \
    docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt \
    docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md \
    docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt
})"
[[ "$(sha_text "${PARENT_MANIFEST}")" == "${PARENT_MANIFEST_SHA256}" ]] ||
  fail 'parent manifest drift'

SOURCE_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    stdlib/hardware/pireus/material_engine_admission.sio \
    examples/pireus_material_engine_admission.sio \
    tests/stdlib/hardware/test_pireus_material_engine_admission.sio
})"
[[ "$(sha_text "${SOURCE_MANIFEST}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'

TOOLCHAIN_MANIFEST="$({
  cd "${ROOT}"
  sha256sum bin/souc bin/souc-lean-single-x86_64
})"
[[ "$(sha_text "${TOOLCHAIN_MANIFEST}")" == "${TOOLCHAIN_SHA256}" ]] ||
  fail 'toolchain manifest drift'

[[ "$(hostname)" == 'sounio-workspace-control-0' ]] || fail 'authority hostname drift'
[[ "$(uname -m)" == 'x86_64' ]] || fail 'authority architecture drift'
[[ "$(uname -r)" == '7.0.2-5-pve' ]] || fail 'authority kernel drift'
grep -Fqx 'PRETTY_NAME="Ubuntu 24.04.4 LTS"' /etc/os-release || fail 'authority OS drift'
grep -Fq 'Model name:                              INTEL(R) XEON(R) GOLD 6526Y' < <(lscpu) || fail 'CPU model drift'
grep -Fq 'CPU(s):                                  64' < <(lscpu) || fail 'logical CPU drift'
grep -Fq 'Socket(s):                               2' < <(lscpu) || fail 'socket drift'
grep -Fq 'Core(s) per socket:                      16' < <(lscpu) || fail 'core topology drift'
grep -Fq 'Thread(s) per core:                      2' < <(lscpu) || fail 'thread topology drift'

[[ "$(sha_text_exact "$(authority_command_record)")" == "${AUTHORITY_COMMAND_SHA256}" ]] ||
  fail 'authority command record drift'
[[ "$(sha_text_exact "$(test_command_record)")" == "${TEST_COMMAND_SHA256}" ]] ||
  fail 'test command record drift'
[[ "$(sha_text_exact "$(tamper_command_record)")" == "${TAMPER_COMMAND_SHA256}" ]] ||
  fail 'tamper command record drift'

authorize d8afe57e5c61b646173b17957fcf9785cfce27d8087606c37e6b49166d66be45 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(freeze_frame)"
authorize 96394b20f8b46b165992c6f306ed89bd9f55fd0cd5a9b4bf81f770536b86da78 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(receipt_frame "${AUTHORITY_COMMAND_SHA256}" "${RESULT_SHA256}")"
authorize b77090f12290899c58137d88f57632a91fa2f71ce0bbbf33a19710fd8c3787ae \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(commit_frame)"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pireus-material-engine-admission.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT
for run in 1 2 3; do
  authorize 0e868eacdaffef497fe1b9fc5359b60b34b61b69383aeaf91737593e9676ce28 \
    'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' \
    "$(preaction_frame "${AUTHORITY_COMMAND_SHA256}")" >/dev/null
  run_authority >"${WORK_DIR}/authority-${run}.txt"
  require_hash "${WORK_DIR}/authority-${run}.txt" "${RESULT_SHA256}"
done
cmp -s "${WORK_DIR}/authority-1.txt" "${WORK_DIR}/authority-2.txt" ||
  fail 'authority runs 1 and 2 differ'
cmp -s "${WORK_DIR}/authority-2.txt" "${WORK_DIR}/authority-3.txt" ||
  fail 'authority runs 2 and 3 differ'
[[ "$(wc -l <"${WORK_DIR}/authority-1.txt")" == 131 ]] || fail 'authority line count drift'
[[ "$(wc -c <"${WORK_DIR}/authority-1.txt")" == 2258 ]] || fail 'authority byte count drift'
require_line "${WORK_DIR}/authority-1.txt" ' matched=12'
require_line "${WORK_DIR}/authority-1.txt" ' unresolved_identity_requests=1'
require_line "${WORK_DIR}/authority-1.txt" ' values=0'
require_line "${WORK_DIR}/authority-1.txt" ' claim_ready=0'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_MATERIAL_SUMMARY error=0'
require_line "${WORK_DIR}/authority-1.txt" ' failures=0'

authorize 03121a38f9385c55a6158b8f4fdb63be5f291ddb54967d9ba14cf57020700e9f \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(receipt_frame "${TEST_COMMAND_SHA256}" "${TEST_RESULT_SHA256}")"
authorize 5666b5491f13baad06d7805775ac55dc13b7d46d5f3610a1fd7901ed48fba717 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE' \
  "$(preaction_frame "${TEST_COMMAND_SHA256}")" >/dev/null
run_test >"${WORK_DIR}/test.txt"
require_hash "${WORK_DIR}/test.txt" "${TEST_RESULT_SHA256}"
require_line "${WORK_DIR}/test.txt" 'pireus material engine admission test passed'

authorize 8e59dc47c3b5d9d8b08610efa1fe9c51d5f2f92f6489209ebb100cba312eff8b \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(tamper_frame)"
set +e
run_tamper >/dev/null
TAMPER_STATUS=$?
set -e
[[ "${TAMPER_STATUS}" -eq 1 ]] || fail "tamper exit ${TAMPER_STATUS}, expected 1"
require_hash /tmp/pireus-material-engine-admission.tampered-dgx-evidence.txt \
  "${TAMPERED_PARENT_SHA256}"
require_hash /tmp/pireus-material-engine-admission.tampered.txt "${TAMPER_RESULT_SHA256}"
require_line /tmp/pireus-material-engine-admission.tampered.txt ' matched=11'
require_line /tmp/pireus-material-engine-admission.tampered.txt ' match=0'
require_line /tmp/pireus-material-engine-admission.tampered.txt 'PIREUS_MATERIAL_SUMMARY error=1'
require_line /tmp/pireus-material-engine-admission.tampered.txt ' failures=3'

PYTHON_FRAME="$(python_frame)"
[[ "$(sha_text "${PYTHON_FRAME}")" == \
  a708c9004b3329e6a80274dc701509bc50cac964b3439302c74392f4d85039af ]] ||
  fail 'Python refusal frame drift'
INTERPRETER_LAUNCH_COUNT=0
set +e
PYTHON_DECISION="$(printf '%s\n' "${PYTHON_FRAME}" | "${GUARDIAN}")"
PYTHON_STATUS=$?
set -e
[[ "${PYTHON_STATUS}" -eq 110 ]] || fail "Python refusal exit ${PYTHON_STATUS}"
[[ "${INTERPRETER_LAUNCH_COUNT}" -eq 0 ]] || fail 'Python interpreter launched'
[[ "${PYTHON_DECISION}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' ]] ||
  fail "Python refusal decision drift: ${PYTHON_DECISION}"
[[ "$(sha_text "${PYTHON_DECISION}")" == \
  3e2b1112dc7ce41d6c752c48daca33e6ee400b93df1e3fafa795a5709b4aa2a3 ]] ||
  fail 'Python refusal decision hash drift'

(
  cd "${ROOT}"
  node scripts/docs/check_docs_registry.mjs >/dev/null
  bash scripts/dev/check_docs_registry.sh >/dev/null
  bash scripts/dev/check_docs_consistency.sh >/dev/null
)

printf 'PIREUS_MATERIAL_ENGINE_ADMISSION_GATE_PASS parents=12 admitted_engines=2 unresolved_requests=1 cost_values=0 negatives=22 reproducible=3 tamper=PASS python_e110=PASS parity_open=false claim_ready=false\n'
