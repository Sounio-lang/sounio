#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"

PDF="/tmp/intel-sdm-vol-2c-326018-092.pdf"
XED="/tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt"
GARDEN="${ROOT}/docs/internal/garden/seeds/2026-08-27-pireus-target-cost-observation.md"
MODULE="${ROOT}/stdlib/hardware/pireus/target_cost_observation.sio"
EXAMPLE="${ROOT}/examples/pireus_target_cost_observation.sio"
TEST="${ROOT}/tests/stdlib/hardware/test_pireus_target_cost_observation.sio"
CONCEPT="${ROOT}/docs/internal/concepts/pireus-target-cost-observation.md"
SEMANTICS="${ROOT}/docs/research/pireus_target_cost_observation_semantics.md"
RECEIPT="${ROOT}/docs/research/receipts/pireus_target_cost_observation_20260827.md"
EVIDENCE="${ROOT}/docs/research/evidence/pireus_target_cost_observation_20260827.txt"
REGISTRY="${ROOT}/docs/internal/concepts/registry.tsv"

ADMISSION_SOURCE="${ROOT}/stdlib/hardware/pireus/xor_selector_material_admission.sio"
ADMISSION_SEMANTICS="${ROOT}/docs/research/pireus_xor_selector_material_admission_semantics.md"
ADMISSION_RECEIPT="${ROOT}/docs/research/receipts/pireus_xor_selector_material_admission_20260827.md"
ADMISSION_EVIDENCE="${ROOT}/docs/research/evidence/pireus_xor_selector_material_admission_20260827.txt"
ENGINE_SOURCE="${ROOT}/stdlib/hardware/pireus/execution_engine.sio"
ENGINE_SEMANTICS="${ROOT}/docs/research/pireus_execution_engine_semantics.md"
ENGINE_RECEIPT="${ROOT}/docs/research/receipts/pireus_execution_engine_20260827.md"
OPERATION_SOURCE="${ROOT}/stdlib/hardware/pireus/xor_convolution_operation.sio"
OPERATION_SEMANTICS="${ROOT}/docs/research/pireus_xor_convolution_operation_semantics.md"
OPERATION_RECEIPT="${ROOT}/docs/research/receipts/pireus_xor_convolution_operation_20260827.md"

DARWIN_RECEIPT="${ROOT}/docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md"
DARWIN_EVIDENCE="${ROOT}/docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt"
APPLE_RECEIPT="${ROOT}/docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md"
APPLE_EVIDENCE="${ROOT}/docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt"
DGX_RECEIPT="${ROOT}/docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md"
DGX_EVIDENCE="${ROOT}/docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt"

GARDEN_COMMIT='de9e1b4d1cb818a0cc1af7cf710e7f31a347211d'
EXECUTABLE_COMMIT='ad38229c7c7d21318e6a445fe5c078b5c72f49be'
SOURCE_SHA256='7ea2815c112b85476fc6ac4d8bb9388ee032062822c6905485c2084ee416d6bc'
SEMANTICS_SHA256='0a899be7cd25375c8c444b9e1f0a71dd102ca8958072a4290073ae21c926a199'
PARENT_MANIFEST_SHA256='9a4f1f28651b8984a0d719ecc4415572b6b301c0ccbee7520d960afeea6bf605'
SOURCE_MANIFEST_SHA256='cdfa7c1438aa524e884bf3f4a69e19e8ed99582e0f2f2effa16451c582fcb596'
TOOLCHAIN_SHA256='2ce5194cdc517de8d7a0063e09d4f4e7b6b5701a23fa1031a3ec9e8f56486b6e'
HARDWARE_SHA256='b6326139297e9ba59d82e208a2404b01e3d57445357a5e803c51c845dd388db0'
AUTHORITY_COMMAND_SHA256='9e5c49dfefa3278f32d5c2381b49a329f90e59181c6c178a8fa47a38273af885'
TEST_COMMAND_SHA256='497e6a868ee9437034bf4ff72916f4831bce0a3a08108a67267b11fb164860e2'
TAMPER_COMMAND_SHA256='8186202c90e3fb344ddd4b911b1ead49ecd9fb054f1d783885d93acdea7f1809'
RESULT_SHA256='99f2e7f0dff71d76c55b5f39f5f514e82128c97a81e31d3a1263a29d2d816d9b'
TEST_RESULT_SHA256='04cf5a9aa26fe3405e8c10249f930ef8ad33b040ba49650e3bef4fdbcd6382ab'
TAMPER_RESULT_SHA256='924c5b542f55a8e5bf29e01aa6abe9029a971a9bf07728d67aff47fb8c2a9345'
ZERO='0 0 0 0 0 0 0 0'

fail() {
  printf 'pireus-target-cost-observation: FAIL: %s\n' "$*" >&2
  exit 1
}

sha_file() {
  sha256sum "$1" | cut -d' ' -f1
}

sha_text() {
  printf '%s\n' "$1" | sha256sum | cut -d' ' -f1
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

freeze_frame() {
  printf '9020 2 3 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
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
  printf '%s' 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_target_cost_observation.sio /tmp/intel-sdm-vol-2c-326018-092.pdf /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt docs/internal/garden/seeds/2026-08-27-pireus-target-cost-observation.md stdlib/hardware/pireus/xor_selector_material_admission.sio docs/research/pireus_xor_selector_material_admission_semantics.md docs/research/receipts/pireus_xor_selector_material_admission_20260827.md docs/research/evidence/pireus_xor_selector_material_admission_20260827.txt stdlib/hardware/pireus/execution_engine.sio docs/research/pireus_execution_engine_semantics.md docs/research/receipts/pireus_execution_engine_20260827.md stdlib/hardware/pireus/xor_convolution_operation.sio docs/research/pireus_xor_convolution_operation_semantics.md docs/research/receipts/pireus_xor_convolution_operation_20260827.md | tee /tmp/pireus-target-cost-observation.authority.txt'
}

test_command_record() {
  printf '%s' 'SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_target_cost_observation.sio /tmp/intel-sdm-vol-2c-326018-092.pdf /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt docs/internal/garden/seeds/2026-08-27-pireus-target-cost-observation.md stdlib/hardware/pireus/xor_selector_material_admission.sio docs/research/pireus_xor_selector_material_admission_semantics.md docs/research/receipts/pireus_xor_selector_material_admission_20260827.md docs/research/evidence/pireus_xor_selector_material_admission_20260827.txt stdlib/hardware/pireus/execution_engine.sio docs/research/pireus_execution_engine_semantics.md docs/research/receipts/pireus_execution_engine_20260827.md stdlib/hardware/pireus/xor_convolution_operation.sio docs/research/pireus_xor_convolution_operation_semantics.md docs/research/receipts/pireus_xor_convolution_operation_20260827.md'
}

tamper_command_record() {
  printf '%s' 'set -o pipefail; SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_target_cost_observation.sio /tmp/intel-sdm-vol-2c-326018-092.pdf /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt docs/internal/garden/seeds/2026-08-27-pireus-target-cost-observation.md stdlib/hardware/pireus/xor_selector_material_admission.sio docs/research/pireus_xor_selector_material_admission_semantics.md docs/research/receipts/pireus_xor_selector_material_admission_20260827.md docs/research/evidence/pireus_xor_selector_material_admission_20260827.txt stdlib/hardware/pireus/execution_engine.sio docs/research/pireus_execution_engine_semantics.md docs/research/receipts/pireus_execution_engine_20260827.md stdlib/hardware/pireus/xor_convolution_operation.sio docs/research/pireus_xor_convolution_operation_semantics.md /tmp/pireus-cost-tampered-operation-receipt.md | tee /tmp/pireus-target-cost-observation.tampered.txt'
}

run_authority() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_target_cost_observation.sio \
      "${PDF}" "${XED}" \
      docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md \
      docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt \
      docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md \
      docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt \
      docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md \
      docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt \
      docs/internal/garden/seeds/2026-08-27-pireus-target-cost-observation.md \
      stdlib/hardware/pireus/xor_selector_material_admission.sio \
      docs/research/pireus_xor_selector_material_admission_semantics.md \
      docs/research/receipts/pireus_xor_selector_material_admission_20260827.md \
      docs/research/evidence/pireus_xor_selector_material_admission_20260827.txt \
      stdlib/hardware/pireus/execution_engine.sio \
      docs/research/pireus_execution_engine_semantics.md \
      docs/research/receipts/pireus_execution_engine_20260827.md \
      stdlib/hardware/pireus/xor_convolution_operation.sio \
      docs/research/pireus_xor_convolution_operation_semantics.md \
      docs/research/receipts/pireus_xor_convolution_operation_20260827.md \
      | tee /tmp/pireus-target-cost-observation.authority.txt
  )
}

run_test() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      tests/stdlib/hardware/test_pireus_target_cost_observation.sio \
      "${PDF}" "${XED}" \
      "${DARWIN_RECEIPT}" "${DARWIN_EVIDENCE}" \
      "${APPLE_RECEIPT}" "${APPLE_EVIDENCE}" \
      "${DGX_RECEIPT}" "${DGX_EVIDENCE}" \
      "${GARDEN}" "${ADMISSION_SOURCE}" "${ADMISSION_SEMANTICS}" \
      "${ADMISSION_RECEIPT}" "${ADMISSION_EVIDENCE}" \
      "${ENGINE_SOURCE}" "${ENGINE_SEMANTICS}" "${ENGINE_RECEIPT}" \
      "${OPERATION_SOURCE}" "${OPERATION_SEMANTICS}" "${OPERATION_RECEIPT}"
  )
}

run_tamper() {
  cp "${OPERATION_RECEIPT}" /tmp/pireus-cost-tampered-operation-receipt.md
  printf '\nPIREUS_COST_TAMPER\n' >> /tmp/pireus-cost-tampered-operation-receipt.md
  (
    set -o pipefail
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_target_cost_observation.sio \
      "${PDF}" "${XED}" \
      "${DARWIN_RECEIPT}" "${DARWIN_EVIDENCE}" \
      "${APPLE_RECEIPT}" "${APPLE_EVIDENCE}" \
      "${DGX_RECEIPT}" "${DGX_EVIDENCE}" \
      "${GARDEN}" "${ADMISSION_SOURCE}" "${ADMISSION_SEMANTICS}" \
      "${ADMISSION_RECEIPT}" "${ADMISSION_EVIDENCE}" \
      "${ENGINE_SOURCE}" "${ENGINE_SEMANTICS}" "${ENGINE_RECEIPT}" \
      "${OPERATION_SOURCE}" "${OPERATION_SEMANTICS}" \
      /tmp/pireus-cost-tampered-operation-receipt.md \
      | tee /tmp/pireus-target-cost-observation.tampered.txt
  )
}

[[ -x "${GUARDIAN}" ]] || fail "native Sounio Loom guardian unavailable"

require_hash "${ROOT}/bin/souc" ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
require_hash "${ROOT}/bin/souc-lean-single-x86_64" 6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
require_hash "${PDF}" 939c9543ff98eefb80f5c5a517bf6f08e864497ea8e032334849f3e39a7b3b07
require_hash "${XED}" e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038

require_hash "${GARDEN}" ecef7f1ff683f1157e89bb5e552e121f51c4191e9662b17a0a5f4a71909223a8
require_hash "${MODULE}" "${SOURCE_SHA256}"
require_hash "${EXAMPLE}" ee237c9419c494879a97ecaaf285ac3d70b0f6ce6055101152a13a39f907c51d
require_hash "${TEST}" 06518d139e8a70c7734a731019d78f71c6d37dfca89001490c1834ddf62eb73e
require_hash "${CONCEPT}" 34914714a36d81adcacfe5b77c3c5773c0714c2358ef20f1592ff7d96a24585e
require_hash "${SEMANTICS}" "${SEMANTICS_SHA256}"
require_hash "${RECEIPT}" b7577c782a82431eb54312137a52c1811f14316d5e5a5fb9e7aa9581f1c304ca
require_hash "${EVIDENCE}" 06f21108ddc89c8c468097b7cefec6a766bda065918fd5797be194878371577b
require_hash "${REGISTRY}" 2d04f31080d3930882946d0a95d964ef126acefbec6ea3a5205bd12a86078db7

require_hash "${ADMISSION_SOURCE}" b9249fe24f5d08fb012631346164d826b8ee975130b0f298a809ad48f4843a66
require_hash "${ADMISSION_SEMANTICS}" 17196cbc2c3fa286c9c2c6e48f042cd3b180d731ee41e0e492077b355ca34ea9
require_hash "${ADMISSION_RECEIPT}" 2615448449a16faf1d826a6d42e0b0212036f485a3a3e815fc064c298070f979
require_hash "${ADMISSION_EVIDENCE}" a59d975337fb4e0d825038e25ba4bf4b11105e28863fdf837d1cba60919ffc7e
require_hash "${ENGINE_SOURCE}" 8b5063f0e9a39650fb0b60e8b70b315f339723690e06050c2bebacece888e37e
require_hash "${ENGINE_SEMANTICS}" c47668a08ad25f39bebe9d8bef90b66eb2ad7119063c19ab8319fa4fab265233
require_hash "${ENGINE_RECEIPT}" 9da8ca53c3cb0e6631c92e55a8e82387aed2bd53863ffa9d646719806eec4ffd
require_hash "${OPERATION_SOURCE}" bc039d5db9f195b94fbeb08f22f9c96164a174c2cea675739e901a07fdf54db8
require_hash "${OPERATION_SEMANTICS}" 40fe69829b1feb5843ea8b4720b70516303e8eda37c144989909b52d1b466fb1
require_hash "${OPERATION_RECEIPT}" 9e1e416defa4a1cfc477f0630623321e12209a40c9f5f878b85cce40be83d330

git -C "${ROOT}" cat-file -e "${GARDEN_COMMIT}^{commit}" || fail 'Garden commit missing'
git -C "${ROOT}" cat-file -e "${EXECUTABLE_COMMIT}^{commit}" || fail 'Sounio executable commit missing'
git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" "${EXECUTABLE_COMMIT}" ||
  fail 'Garden is not an ancestor of first executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" HEAD ||
  fail 'first executable is not an ancestor of HEAD'

PARENT_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    docs/internal/garden/seeds/2026-08-27-pireus-target-cost-observation.md \
    stdlib/hardware/pireus/xor_selector_material_admission.sio \
    docs/research/pireus_xor_selector_material_admission_semantics.md \
    docs/research/receipts/pireus_xor_selector_material_admission_20260827.md \
    docs/research/evidence/pireus_xor_selector_material_admission_20260827.txt \
    stdlib/hardware/pireus/execution_engine.sio \
    docs/research/pireus_execution_engine_semantics.md \
    docs/research/receipts/pireus_execution_engine_20260827.md \
    stdlib/hardware/pireus/xor_convolution_operation.sio \
    docs/research/pireus_xor_convolution_operation_semantics.md \
    docs/research/receipts/pireus_xor_convolution_operation_20260827.md
})"
[[ "$(sha_text "${PARENT_MANIFEST}")" == "${PARENT_MANIFEST_SHA256}" ]] ||
  fail 'parent manifest drift'

SOURCE_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    stdlib/hardware/pireus/target_cost_observation.sio \
    examples/pireus_target_cost_observation.sio \
    tests/stdlib/hardware/test_pireus_target_cost_observation.sio
})"
[[ "$(sha_text "${SOURCE_MANIFEST}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'

[[ "$(sha_text "$(authority_command_record)")" == "${AUTHORITY_COMMAND_SHA256}" ]] ||
  fail 'authority command record drift'
[[ "$(sha_text "$(test_command_record)")" == "${TEST_COMMAND_SHA256}" ]] ||
  fail 'test command record drift'
[[ "$(sha_text "$(tamper_command_record)")" == "${TAMPER_COMMAND_SHA256}" ]] ||
  fail 'tamper command record drift'

authorize e22440429d0beec5290d6bca95b4e70d1b6c07c571e87cf9247e501833e88913 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(freeze_frame)"
authorize 93e0867f20a3d6e0a1510f26b6b076a14726c1892c9ba8320c6c600126655306 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(receipt_frame "${AUTHORITY_COMMAND_SHA256}" "${RESULT_SHA256}")"

for run in 1 2 3; do
  run_authority >/dev/null
  cp /tmp/pireus-target-cost-observation.authority.txt \
    "/tmp/pireus-target-cost-observation.run${run}.txt"
  require_hash "/tmp/pireus-target-cost-observation.run${run}.txt" "${RESULT_SHA256}"
done
cmp -s /tmp/pireus-target-cost-observation.run1.txt \
  /tmp/pireus-target-cost-observation.run2.txt || fail 'authority run 1 != run 2'
cmp -s /tmp/pireus-target-cost-observation.run2.txt \
  /tmp/pireus-target-cost-observation.run3.txt || fail 'authority run 2 != run 3'
require_line /tmp/pireus-target-cost-observation.run3.txt \
  'PIREUS_COST_SUMMARY error=0'
grep -Fqx ' failures=0' /tmp/pireus-target-cost-observation.run3.txt ||
  fail 'authority failures are not zero'

authorize 06a0223480c5ae97d9c7a514ee36a74ef0b0556bb13975770d7fdd57f8568c4c \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(receipt_frame "${TEST_COMMAND_SHA256}" "${TEST_RESULT_SHA256}")"
TEST_OUTPUT="$(run_test)"
[[ "$(sha_text "${TEST_OUTPUT}")" == "${TEST_RESULT_SHA256}" ]] ||
  fail 'dedicated Sounio test result drift'

authorize bfb0e78a56c730aa28eb9eaf930ebc72b5d962aec4fbcdb7ddd7abe90ab1ad53 \
  'SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN' \
  "$(tamper_frame)"
set +e
run_tamper >/dev/null
TAMPER_STATUS=$?
set -e
[[ "${TAMPER_STATUS}" -eq 1 ]] || fail "tamper exit ${TAMPER_STATUS}, expected 1"
require_hash /tmp/pireus-target-cost-observation.tampered.txt "${TAMPER_RESULT_SHA256}"
grep -Fqx ' matched=10' /tmp/pireus-target-cost-observation.tampered.txt ||
  fail 'tamper did not reduce parent matches to 10'
require_line /tmp/pireus-target-cost-observation.tampered.txt \
  'PIREUS_COST_SUMMARY error=1'
grep -Fqx ' failures=3' /tmp/pireus-target-cost-observation.tampered.txt ||
  fail 'tamper failures != 3'

PYTHON_FRAME="$(python_frame)"
[[ "$(sha_text "${PYTHON_FRAME}")" == \
  e4261c050af8f486685bf6dd6da869a9d4c9fefedf67b4a0e5ad090c4a4a4eef ]] ||
  fail 'Python refusal frame drift'
set +e
PYTHON_DECISION="$(printf '%s\n' "${PYTHON_FRAME}" | "${GUARDIAN}")"
PYTHON_STATUS=$?
set -e
[[ "${PYTHON_STATUS}" -eq 110 ]] || fail "Python refusal exit ${PYTHON_STATUS}"
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
  bash scripts/dev/check_offload_policy.sh >/dev/null
)

printf 'PIREUS_TARGET_COST_OBSERVATION_GATE_PASS requests=7 parents=11 negatives=26 reproducible=3 tamper=PASS python_e110=PASS parity_open=false claim_ready=false\n'
