#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
NODE="$(command -v node)"
TAMPER_DIR='/tmp/pireus-apple-cpu-latency-3a92553aae0c4a96'
TAMPER_PARENT="${TAMPER_DIR}/material-evidence.txt"
TAMPER_OUTPUT="${TAMPER_DIR}/result.txt"

PDF="/tmp/intel-sdm-vol-2c-326018-092.pdf"
XED="/tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt"
GARDEN="${ROOT}/docs/internal/garden/seeds/2026-08-28-pireus-apple-cpu-dependency-latency-request.md"
MODULE="${ROOT}/stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio"
EXAMPLE="${ROOT}/examples/pireus_apple_cpu_dependency_latency_request.sio"
TEST="${ROOT}/tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_request.sio"
CONCEPT="${ROOT}/docs/internal/concepts/pireus-apple-cpu-dependency-latency-request.md"
SEMANTICS="${ROOT}/docs/research/pireus_apple_cpu_dependency_latency_request_semantics.md"
RECEIPT="${ROOT}/docs/research/receipts/pireus_apple_cpu_dependency_latency_request_20260828.md"
EVIDENCE="${ROOT}/docs/research/evidence/pireus_apple_cpu_dependency_latency_request_20260828.txt"
REGISTRY="${ROOT}/docs/internal/concepts/registry.tsv"
TOPIC_REGISTRY="${ROOT}/docs/governance/topic-registry.v1.json"
AUTHORITY_MATRIX="${ROOT}/docs/governance/DOCS_AUTHORITY_MATRIX.md"
ACCEPTANCE_REPORT="${ROOT}/docs/governance/DOCS_ACCEPTANCE_REPORT.md"

TARGET_SOURCE="${ROOT}/stdlib/hardware/pireus/target_cost_observation.sio"
TARGET_SEMANTICS="${ROOT}/docs/research/pireus_target_cost_observation_semantics.md"
TARGET_RECEIPT="${ROOT}/docs/research/receipts/pireus_target_cost_observation_20260827.md"
TARGET_EVIDENCE="${ROOT}/docs/research/evidence/pireus_target_cost_observation_20260827.txt"
MATERIAL_SOURCE="${ROOT}/stdlib/hardware/pireus/material_engine_admission.sio"
MATERIAL_SEMANTICS="${ROOT}/docs/research/pireus_material_engine_admission_semantics.md"
MATERIAL_RECEIPT="${ROOT}/docs/research/receipts/pireus_material_engine_admission_20260828.md"
MATERIAL_EVIDENCE="${ROOT}/docs/research/evidence/pireus_material_engine_admission_20260828.txt"

GARDEN_COMMIT='b1d80d17f0d2ab915557b34732e9580df269e19a'
EXECUTABLE_COMMIT='763322b28df3709eb7544d38dbc00ee779071631'
GARDEN_SHA256='b574d7352019576dceceab32675834aedb24961dcd76f97646a5cbed4277aa7f'
SOURCE_SHA256='3a92553aae0c4a9606f4964e1613a31452b7fb4d197b128dbb73cc24b87b550e'
EXAMPLE_SHA256='390c0db4231107fa8c7014cd7749acd26624658c330a82ebc3afbd59b0b6e259'
TEST_SHA256='d09c062f0d39cf256106aebcc3381cbd994dfdcefdff6f212657ba07dae86ea6'
CONCEPT_SHA256='0a68de28301919ed5d91d1a5a8bf9815a37cd733f436d4d00457b93de3749536'
SEMANTICS_SHA256='9bd767db814e47bfc087e07c0f9ff33b65faea5b885ae0f8ed3a6e646c015e6d'
RECEIPT_SHA256='0ee12f3502efb26056bdbcf850360c0a5df727627a3c67499d363744f7c73272'
EVIDENCE_SHA256='cf4455690426038cc7477b673bcf763e9755e8147f1ff55e882086826626482b'
REGISTRY_SHA256='b216ab8764f872547f1b5e6630084b86c41616a1f00f4ddaaefa54d12bb7408d'
TOPIC_REGISTRY_SHA256='755c2271a4732bed26d223fcf9d6790b5579a70ce2894911df09ca306764362c'
AUTHORITY_MATRIX_SHA256='1c7afcc62479d214339063ade3cf8b7a1ceab855e28c86d33d4259ccd0784621'
ACCEPTANCE_REPORT_SHA256='58fa7243e01771fcf9b8f316a60e657fe3264659a7811e54b1b6ef077211ff21'
SOURCE_MANIFEST_SHA256='14ee66df590cccbaea0f0289ab9d9abcc693f17989059443b1841e7d0970ef3f'
PARENT_MANIFEST_SHA256='aca23c0c43db3fee6d4fd8c7f4ca58ed9bde460c6b0e5978442170d3ab7320af'
EXECUTION_INPUT_MANIFEST_SHA256='5abb425da9e16046615acd1abba45997057399f9b294f3b746076729c85c8596'
TOOLCHAIN_SHA256='850c094e02d85fee153297ccf8babbe171e3ec47def68ac2976c3473092b36ac'
HARDWARE_SHA256='464f1a4530cb0829854ddbafc0786d12cc9fc98cef1afced51f40679ba27517c'
AUTHORITY_COMMAND_SHA256='70f0201b957cba573bd675f12880cd6a5c12fefa9ef15390b5f8a5fa9eb9b5fc'
TEST_COMMAND_SHA256='00630e8abba5421ef537e03ebb5166dc6dd3a75d968f98a57cdd575b5916404f'
TAMPER_COMMAND_SHA256='65bae17a1fea81a4e1d5e7b8486ccacc89e08203e7c2499c3d7d9059621a496d'
DEFAULT_CHECK_COMMAND_SHA256='bfa3021d0ee37678d348eaed14a781a8b14dfe4077ebdf6753322d7b906ce020'
RESULT_SHA256='443e49c11a2ad04ac7e3f9b061bebe50fca7959f812aa93039c449c2133b5349'
TEST_RESULT_SHA256='88dd937d09afb205e80e77e646300928d3d7dad1f53d98a62a9232b1f6579774'
TAMPERED_PARENT_SHA256='b81276e7596dbfb158107a32bd8de3a6cd134b83c7ee0d8a7491915b0e734310'
TAMPER_RESULT_SHA256='ee90eda9b0cf7659e55aa2212978007c4cbe660f29c23d9aacc4c08c1e6f231d'
DEFAULT_CHECK_RESULT_SHA256='de792d0508cf702df0b632503ee506c971e637d96ff3be250cf4c86c69b06b98'
ZERO='0 0 0 0 0 0 0 0'

AUTHORITY_ARGS=(
  /tmp/intel-sdm-vol-2c-326018-092.pdf
  /tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt
  docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md
  docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt
  docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md
  docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt
  docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md
  docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt
  docs/internal/garden/seeds/2026-08-28-pireus-apple-cpu-dependency-latency-request.md
  stdlib/hardware/pireus/target_cost_observation.sio
  docs/research/pireus_target_cost_observation_semantics.md
  docs/research/receipts/pireus_target_cost_observation_20260827.md
  docs/research/evidence/pireus_target_cost_observation_20260827.txt
  stdlib/hardware/pireus/material_engine_admission.sio
  docs/research/pireus_material_engine_admission_semantics.md
  docs/research/receipts/pireus_material_engine_admission_20260828.md
  docs/research/evidence/pireus_material_engine_admission_20260828.txt
  docs/internal/garden/seeds/2026-08-27-pireus-target-cost-observation.md
  stdlib/hardware/pireus/xor_selector_material_admission.sio
  docs/research/pireus_xor_selector_material_admission_semantics.md
  docs/research/receipts/pireus_xor_selector_material_admission_20260827.md
  docs/research/evidence/pireus_xor_selector_material_admission_20260827.txt
  stdlib/hardware/pireus/execution_engine.sio
  docs/research/pireus_execution_engine_semantics.md
  docs/research/receipts/pireus_execution_engine_20260827.md
  stdlib/hardware/pireus/xor_convolution_operation.sio
  docs/research/pireus_xor_convolution_operation_semantics.md
  docs/research/receipts/pireus_xor_convolution_operation_20260827.md
  docs/internal/garden/seeds/2026-08-28-pireus-material-engine-admission.md
  docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md
  docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt
  docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md
  docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt
)

# Slots 4..7 feed target-cost evaluation; 29..32 feed material admission.
# The repeated Apple/DGX paths therefore have distinct positional roles.

fail() {
  printf 'pireus-apple-cpu-dependency-latency-request: FAIL: %s\n' "$*" >&2
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

stage_three_frame() {
  local action="$1" language="$2" role="$3" command_sha="$4" result_sha="$5"
  printf '9020 3 %s %s %s 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "${action}" "${language}" "${role}" \
    "$(sha_limbs "${SOURCE_SHA256}")" \
    "$(sha_limbs "${SEMANTICS_SHA256}")" \
    "$(sha_limbs "${PARENT_MANIFEST_SHA256}")" \
    "$(sha_limbs "${TOOLCHAIN_SHA256}")" \
    "$(sha_limbs "${HARDWARE_SHA256}")" \
    "${command_sha}" "${result_sha}" "${ZERO}"
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
  printf '%s' "SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_apple_cpu_dependency_latency_request.sio ${AUTHORITY_ARGS[*]}"
}

test_command_record() {
  printf '%s' "SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_request.sio ${AUTHORITY_ARGS[*]}"
}

tamper_command_record() {
  local args=("${AUTHORITY_ARGS[@]}")
  args[16]="${TAMPER_PARENT}"
  printf '%s' "set -o pipefail; SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_apple_cpu_dependency_latency_request.sio ${args[*]} | tee ${TAMPER_OUTPUT}"
}

default_check_command_record() {
  printf '%s' './bin/souc check stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio'
}

run_authority() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_apple_cpu_dependency_latency_request.sio \
      "${AUTHORITY_ARGS[@]}"
  )
}

run_test() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_request.sio \
      "${AUTHORITY_ARGS[@]}"
  )
}

run_tamper() {
  local args=("${AUTHORITY_ARGS[@]}")
  [[ ! -e "${TAMPER_DIR}" ]] || fail "tamper directory already exists: ${TAMPER_DIR}"
  mkdir -m 0700 "${TAMPER_DIR}" || fail 'cannot create private tamper directory'
  cp "${MATERIAL_EVIDENCE}" "${TAMPER_PARENT}"
  printf '\nPIREUS_APPLE_CPU_LATENCY_TAMPER\n' >> "${TAMPER_PARENT}"
  args[16]="${TAMPER_PARENT}"
  (
    set -o pipefail
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_apple_cpu_dependency_latency_request.sio \
      "${args[@]}" | tee "${TAMPER_OUTPUT}"
  )
}

run_default_check() {
  (
    cd "${ROOT}"
    ./bin/souc check stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio
  )
}

[[ -x "${GUARDIAN}" ]] || fail 'native Sounio Loom guardian unavailable'
[[ "${#AUTHORITY_ARGS[@]}" -eq 33 ]] || fail 'authority argument count drift'
[[ "${AUTHORITY_ARGS[4]}" == 'docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md' ]] || fail 'target Apple slot drift'
[[ "${AUTHORITY_ARGS[6]}" == 'docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md' ]] || fail 'target DGX slot drift'
[[ "${AUTHORITY_ARGS[29]}" == "${AUTHORITY_ARGS[4]}" ]] || fail 'material Apple receipt slot drift'
[[ "${AUTHORITY_ARGS[30]}" == "${AUTHORITY_ARGS[5]}" ]] || fail 'material Apple evidence slot drift'
[[ "${AUTHORITY_ARGS[31]}" == "${AUTHORITY_ARGS[6]}" ]] || fail 'material DGX receipt slot drift'
[[ "${AUTHORITY_ARGS[32]}" == "${AUTHORITY_ARGS[7]}" ]] || fail 'material DGX evidence slot drift'

require_hash "${GUARDIAN}" 208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
require_hash "${NODE}" 81925c0995b5c1427b5d538e6a90ca2fdc4daffb786b09af749beaf7369d4e90
[[ "$(node --version)" == 'v22.22.2' ]] || fail 'Node version drift'
require_hash "${ROOT}/scripts/docs/check_docs_registry.mjs" 535a4b87714b1a5a6824d8fa2f88158e734fe1ff420dd9d4c39852860ea4238d
require_hash "${ROOT}/scripts/dev/check_docs_registry.sh" d53930721b2371dcfaa9b5ae4add81dfae8d0a78eb4699dea93fd586042afdf0
require_hash "${ROOT}/scripts/dev/check_docs_consistency.sh" 21030af5d04e85c94f9ef5653def07b658777442b9e3de04476a716ae7e39c65
require_hash "${PDF}" 939c9543ff98eefb80f5c5a517bf6f08e864497ea8e032334849f3e39a7b3b07
require_hash "${XED}" e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038
require_hash "${ROOT}/bin/souc" ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
require_hash "${ROOT}/bin/souc-lean-single-x86_64" 6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
require_hash "${GARDEN}" "${GARDEN_SHA256}"
require_hash "${MODULE}" "${SOURCE_SHA256}"
require_hash "${EXAMPLE}" "${EXAMPLE_SHA256}"
require_hash "${TEST}" "${TEST_SHA256}"
require_hash "${CONCEPT}" "${CONCEPT_SHA256}"
require_hash "${SEMANTICS}" "${SEMANTICS_SHA256}"
require_hash "${RECEIPT}" "${RECEIPT_SHA256}"
require_hash "${EVIDENCE}" "${EVIDENCE_SHA256}"
require_hash "${REGISTRY}" "${REGISTRY_SHA256}"
require_hash "${TOPIC_REGISTRY}" "${TOPIC_REGISTRY_SHA256}"
require_hash "${AUTHORITY_MATRIX}" "${AUTHORITY_MATRIX_SHA256}"
require_hash "${ACCEPTANCE_REPORT}" "${ACCEPTANCE_REPORT_SHA256}"

require_hash "${TARGET_SOURCE}" 7ea2815c112b85476fc6ac4d8bb9388ee032062822c6905485c2084ee416d6bc
require_hash "${TARGET_SEMANTICS}" 0a899be7cd25375c8c444b9e1f0a71dd102ca8958072a4290073ae21c926a199
require_hash "${TARGET_RECEIPT}" b7577c782a82431eb54312137a52c1811f14316d5e5a5fb9e7aa9581f1c304ca
require_hash "${TARGET_EVIDENCE}" 06f21108ddc89c8c468097b7cefec6a766bda065918fd5797be194878371577b
require_hash "${MATERIAL_SOURCE}" b98d799fec6452f9afaecc7e418578cc0ca72a1d3e7c0a1f84d474e0aa2730ec
require_hash "${MATERIAL_SEMANTICS}" bbbd7bc9c99e1de46a5317d99a7893adebfdf7fb46454cbd7002b3098281b6ee
require_hash "${MATERIAL_RECEIPT}" 352ad87a4d05ef32333acb2f31a48f49e7d520ab99c3e99a8b4a3e6bf3f55aa4
require_hash "${MATERIAL_EVIDENCE}" 08a6e0d4b2e5a0bb4e981cfcc8df5192d9b9b77f53eb3862c86f29d9409e425b

git -C "${ROOT}" cat-file -e "${GARDEN_COMMIT}^{commit}" || fail 'Garden commit missing'
git -C "${ROOT}" cat-file -e "${EXECUTABLE_COMMIT}^{commit}" || fail 'first executable commit missing'
git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" "${EXECUTABLE_COMMIT}" ||
  fail 'Garden is not an ancestor of first executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" HEAD ||
  fail 'first executable is not an ancestor of HEAD'
if git -C "${ROOT}" show \
  "${EXECUTABLE_COMMIT}:stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio" |
  grep -Fq 'pireus_apple_cpu_dependency_latency_request_matches_frozen_semantics'; then
  fail 'exact matcher exists in first executable commit'
fi
grep -Fq 'pireus_apple_cpu_dependency_latency_request_matches_frozen_semantics' "${MODULE}" ||
  fail 'exact matcher missing from frozen module'

PARENT_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    docs/internal/garden/seeds/2026-08-28-pireus-apple-cpu-dependency-latency-request.md \
    stdlib/hardware/pireus/target_cost_observation.sio \
    docs/research/pireus_target_cost_observation_semantics.md \
    docs/research/receipts/pireus_target_cost_observation_20260827.md \
    docs/research/evidence/pireus_target_cost_observation_20260827.txt \
    stdlib/hardware/pireus/material_engine_admission.sio \
    docs/research/pireus_material_engine_admission_semantics.md \
    docs/research/receipts/pireus_material_engine_admission_20260828.md \
    docs/research/evidence/pireus_material_engine_admission_20260828.txt
})"
[[ "$(sha_text "${PARENT_MANIFEST}")" == "${PARENT_MANIFEST_SHA256}" ]] ||
  fail 'parent manifest drift'

SOURCE_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio \
    examples/pireus_apple_cpu_dependency_latency_request.sio \
    tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_request.sio
})"
[[ "$(sha_text "${SOURCE_MANIFEST}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'

EXECUTION_INPUT_MANIFEST="$({
  cd "${ROOT}"
  sha256sum "${AUTHORITY_ARGS[@]}"
})"
[[ "$(sha_text "${EXECUTION_INPUT_MANIFEST}")" == "${EXECUTION_INPUT_MANIFEST_SHA256}" ]] ||
  fail 'execution input manifest drift'

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
[[ "$(sha_text_exact "$(default_check_command_record)")" == "${DEFAULT_CHECK_COMMAND_SHA256}" ]] ||
  fail 'default check command record drift'

ALLOW_FROZEN='SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
ALLOW_EXECUTABLE='SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize 16b33447745b2d1f5af45f44327259cf3c72dadab5c7245325f5a13faa95c531 \
  "${ALLOW_FROZEN}" "$(freeze_frame)"
authorize 644ba5f2683edd31e1c5bce3822426742ef3227c2b3bad256a874d391fb7d2f3 \
  "${ALLOW_FROZEN}" \
  "$(stage_three_frame 8 1 1 "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" "$(sha_limbs "${RESULT_SHA256}")")"
authorize e9690659f0cc1c01f94378a9c2ec18810bd0e503436fa4c47becfa85b1f47d7f \
  "${ALLOW_FROZEN}" \
  "$(stage_three_frame 8 1 1 "$(sha_limbs "${TEST_COMMAND_SHA256}")" "$(sha_limbs "${TEST_RESULT_SHA256}")")"
authorize b0939964446b25c6a50f840640fcc2dd09fff0c8280257bed076773a8f2ca6e9 \
  "${ALLOW_FROZEN}" \
  "$(stage_three_frame 9 1 1 "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" "$(sha_limbs "${RESULT_SHA256}")")"
authorize 9daeda4872ec777d1e0dad17ff92bac49c4505a8e7d83dbc9759868d886f78d4 \
  "${ALLOW_FROZEN}" \
  "$(stage_three_frame 10 1 1 "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" "$(sha_limbs "${RESULT_SHA256}")")"
authorize b92772b26b11e1f033a1c8c023091d1928765b7ad643ec4d9d4522f6f5389746 \
  "${ALLOW_FROZEN}" \
  "$(stage_three_frame 5 6 6 "${ZERO}" "${ZERO}")"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pireus-apple-cpu-latency.XXXXXX")"
trap 'rm -rf -- "${WORK_DIR}" "${TAMPER_DIR}"' EXIT
for run in 1 2 3; do
  authorize 9b23f721f5b638959bd33de75edf43a1eef7b2cb8222ecbb90cb2d7df192bc68 \
    "${ALLOW_EXECUTABLE}" "$(preaction_frame "${AUTHORITY_COMMAND_SHA256}")" >/dev/null
  run_authority >"${WORK_DIR}/authority-${run}.txt"
  require_hash "${WORK_DIR}/authority-${run}.txt" "${RESULT_SHA256}"
done
cmp -s "${WORK_DIR}/authority-1.txt" "${WORK_DIR}/authority-2.txt" ||
  fail 'authority runs 1 and 2 differ'
cmp -s "${WORK_DIR}/authority-2.txt" "${WORK_DIR}/authority-3.txt" ||
  fail 'authority runs 2 and 3 differ'
[[ "$(wc -l <"${WORK_DIR}/authority-1.txt")" == 124 ]] || fail 'authority line count drift'
[[ "$(wc -c <"${WORK_DIR}/authority-1.txt")" == 2145 ]] || fail 'authority byte count drift'
require_line "${WORK_DIR}/authority-1.txt" ' matched=9'
require_line "${WORK_DIR}/authority-1.txt" ' target_cost_live=1'
require_line "${WORK_DIR}/authority-1.txt" ' material_live=1'
require_line "${WORK_DIR}/authority-1.txt" ' request_id=4'
require_line "${WORK_DIR}/authority-1.txt" ' machine=707301'
require_line "${WORK_DIR}/authority-1.txt" ' engine=707302'
require_line "${WORK_DIR}/authority-1.txt" ' state=708201'
require_line "${WORK_DIR}/authority-1.txt" ' feasibility=708300'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_LATENCY_BOUNDARY interface_ready=0'
require_line "${WORK_DIR}/authority-1.txt" ' execution_authorized=0'
require_line "${WORK_DIR}/authority-1.txt" ' value_present=0'
require_line "${WORK_DIR}/authority-1.txt" ' parity_open=0'
require_line "${WORK_DIR}/authority-1.txt" ' claim_ready=0'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_LATENCY_ONTOLOGY triples=26'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_LATENCY_NEGATIVES passed=28'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_LATENCY_SUMMARY error=0'
require_line "${WORK_DIR}/authority-1.txt" ' failures=0'

authorize 2ef84722f846e53df024999dc753b076e5a3e057fa7bba3cf5c4aa895101fa46 \
  "${ALLOW_EXECUTABLE}" "$(preaction_frame "${TEST_COMMAND_SHA256}")" >/dev/null
run_test >"${WORK_DIR}/test.txt"
require_hash "${WORK_DIR}/test.txt" "${TEST_RESULT_SHA256}"
require_line "${WORK_DIR}/test.txt" 'pireus Apple CPU dependency latency request test passed'

authorize 1b578a5afbbbea992d1b78433101c812b0c4d97ca1b187cd77f7c17217f47119 \
  "${ALLOW_FROZEN}" \
  "$(stage_three_frame 11 1 1 "$(sha_limbs "${TAMPER_COMMAND_SHA256}")" "${ZERO}")"
set +e
run_tamper >/dev/null
TAMPER_STATUS=$?
set -e
[[ "${TAMPER_STATUS}" -eq 1 ]] || fail "tamper exit ${TAMPER_STATUS}, expected 1"
require_hash "${TAMPER_PARENT}" "${TAMPERED_PARENT_SHA256}"
require_hash "${TAMPER_OUTPUT}" "${TAMPER_RESULT_SHA256}"
[[ "$(wc -l <"${TAMPER_OUTPUT}")" == 124 ]] || fail 'tamper line count drift'
[[ "$(wc -c <"${TAMPER_OUTPUT}")" == 2144 ]] || fail 'tamper byte count drift'
require_line "${TAMPER_OUTPUT}" ' matched=8'
require_line "${TAMPER_OUTPUT}" 'PIREUS_APPLE_LATENCY_SUMMARY error=1'
require_line "${TAMPER_OUTPUT}" ' failures=3'

PYTHON_FRAME="$(stage_three_frame 4 7 7 "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" "$(sha_limbs "${RESULT_SHA256}")")"
[[ "$(sha_text "${PYTHON_FRAME}")" == \
  b5c461c0b93fedc804935193756a48bb4a0b3fed8139871d5ad4a50e0aacd318 ]] ||
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

authorize 22745f4b71047f191a8c9eda3f3702e494bb1acf879dbefd7e17e6dfc92d5b91 \
  "${ALLOW_EXECUTABLE}" "$(preaction_frame "${DEFAULT_CHECK_COMMAND_SHA256}")" >/dev/null
set +e
run_default_check >"${WORK_DIR}/default-check.txt" 2>&1
DEFAULT_STATUS=$?
set -e
[[ "${DEFAULT_STATUS}" -eq 1 ]] || fail "default check exit ${DEFAULT_STATUS}, expected 1"
require_hash "${WORK_DIR}/default-check.txt" "${DEFAULT_CHECK_RESULT_SHA256}"
[[ "$(wc -l <"${WORK_DIR}/default-check.txt")" == 10 ]] || fail 'default check line count drift'
[[ "$(wc -c <"${WORK_DIR}/default-check.txt")" == 602 ]] || fail 'default check byte count drift'
require_line "${WORK_DIR}/default-check.txt" 'science-boundary: mode=advisory verdict=UNKNOWN'
require_line "${WORK_DIR}/default-check.txt" 'run_check_mode: AST closure incomplete nodes=0'
require_line "${WORK_DIR}/default-check.txt" ' unresolved=0'
require_line "${WORK_DIR}/default-check.txt" ' saturated=false'

require_line "${EVIDENCE}" 'apple_login_locator=demetrios@sounio-language-macbook'
require_line "${EVIDENCE}" 'apple_locator_in_binding=false'
require_line "${EVIDENCE}" 'measurement_feasibility=UNKNOWN'
require_line "${EVIDENCE}" 'apple_remote_execution=false'
require_line "${EVIDENCE}" 'authority_same_host_byte_identical_runs=3'
require_line "${EVIDENCE}" 'python_exit_code=110'
require_line "${EVIDENCE}" 'python_execution_request_denied=true'
require_line "${EVIDENCE}" 'python_process_invocation_by_gate=false'
require_line "${EVIDENCE}" 'external_llm_confirmed_result=false'
require_line "${EVIDENCE}" 'claim_ready=false'
grep -Fq 'The login locator is routing metadata only.' "${RECEIPT}" ||
  fail 'locator boundary missing from receipt'
grep -Fq 'No connection to the Mac was' "${RECEIPT}" ||
  fail 'no-remote boundary missing from receipt'
grep -Fq 'review_promoted=false' "${RECEIPT}" ||
  fail 'review-only boundary missing from receipt'

(
  cd "${ROOT}"
  node scripts/docs/check_docs_registry.mjs >/dev/null
  bash scripts/dev/check_docs_registry.sh >/dev/null
  bash scripts/dev/check_docs_consistency.sh >/dev/null
)

printf 'PIREUS_APPLE_CPU_DEPENDENCY_LATENCY_REQUEST_GATE_PASS parents=9 template=4 machine=707301 engine=707302 ontology=26 negatives=28 deterministic_replays=3 tamper=PASS python_e110=PASS remote_execution=false interface_ready=false feasibility=UNKNOWN parity_open=false claim_ready=false\n'
