#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_COMMON_DIR="$(git -C "${ROOT}" rev-parse --path-format=absolute --git-common-dir)"
GUARDIAN="${SOUNIO_LOOM_LANGUAGE_AUTHORITY_RUNTIME:-${GIT_COMMON_DIR}/sounio-coord-runtime/current/bin/sounio-loom-language-authority-runtime}"
NODE="$(command -v node)"

PDF="/tmp/intel-sdm-vol-2c-326018-092.pdf"
XED="/tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt"
GARDEN="${ROOT}/docs/internal/garden/seeds/2026-08-28-pireus-apple-cpu-dependency-latency-interface-feasibility.md"
MODULE="${ROOT}/stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio"
EXAMPLE="${ROOT}/examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio"
TEST="${ROOT}/tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_interface_feasibility.sio"
CONCEPT="${ROOT}/docs/internal/concepts/pireus-apple-cpu-dependency-latency-interface-feasibility.md"
SEMANTICS="${ROOT}/docs/research/pireus_apple_cpu_dependency_latency_interface_feasibility_semantics.md"
RECEIPT="${ROOT}/docs/research/receipts/pireus_apple_cpu_dependency_latency_interface_feasibility_20260828.md"
EVIDENCE="${ROOT}/docs/research/evidence/pireus_apple_cpu_dependency_latency_interface_feasibility_20260828.txt"
REGISTRY="${ROOT}/docs/internal/concepts/registry.tsv"
TOPIC_REGISTRY="${ROOT}/docs/governance/topic-registry.v1.json"
AUTHORITY_MATRIX="${ROOT}/docs/governance/DOCS_AUTHORITY_MATRIX.md"
ACCEPTANCE_REPORT="${ROOT}/docs/governance/DOCS_ACCEPTANCE_REPORT.md"

REQUEST_SOURCE="${ROOT}/stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio"
REQUEST_SEMANTICS="${ROOT}/docs/research/pireus_apple_cpu_dependency_latency_request_semantics.md"
REQUEST_RECEIPT="${ROOT}/docs/research/receipts/pireus_apple_cpu_dependency_latency_request_20260828.md"
REQUEST_EVIDENCE="${ROOT}/docs/research/evidence/pireus_apple_cpu_dependency_latency_request_20260828.txt"

GARDEN_COMMIT='30237723bc53bbee48a93893be4da5b5f2118053'
EXECUTABLE_COMMIT='c924d0014c88af8873eeaa3ca5d2c11cf468a167'
GARDEN_SHA256='19482cbceb1bf7f3f7236446ebeff8b7d46c7b99249ba5910ece145fad641dd7'
FIRST_EXECUTABLE_SOURCE_SHA256='0893a32298d30cd1978039fa5b69c637e446aa5da112812fcf776cd52fbc4767'
SOURCE_SHA256='d8c7e6f9410c36f6858fb2379efa010a5adbaa32c615d89edc3e764a0606a6be'
EXAMPLE_SHA256='b7e0f89c3684025407094d5abcdfa5508f7de56e5b415d14882d54dab6b41873'
TEST_SHA256='dc9345b15444a53fa46c14dc68e090be694757e0fbd12ad6ed0b37944668435c'
CONCEPT_SHA256='ceb4b664101c6a43029cb35196856b0428c8ff6e4381b41edbb4b7108628905e'
SEMANTICS_SHA256='6819916ac4240923a149dd95ee9dcbeaba8d3826b7452dd819e177ff62ce8c7f'
RECEIPT_SHA256='d9b0684f62d44e3641bd28db4f0afc3e65f89ef32472667a6eaae7c3c8005592'
EVIDENCE_SHA256='16af971f93318f791bfdeb6e641f6517da15c4c0ba9240e7c638ecd50b28a391'
REGISTRY_SHA256='f94d2bf850ef8da17904796172187fb0b95c0fe813266d1e09befd133a55e9a2'
TOPIC_REGISTRY_SHA256='3cd791ab188de4b9dd31c7f004e17d15749b29eaf11837d639f7c53c2560c74a'
AUTHORITY_MATRIX_SHA256='56f8de3e9e8cc5adf460a22c79b6977ed6d5c6b200ef55a08a9483a25806a5ba'
ACCEPTANCE_REPORT_SHA256='4c25692bacfbffbe8c864185ae63e4ac37750e47cc9382e6af154a0d9b806e70'
SOURCE_MANIFEST_SHA256='4782063cf31b6a1a291e5a4832bb4567c926c2923073e68783d7518fbc6aa888'
PARENT_MANIFEST_SHA256='bb0c19a4f03dea06ed496b3a9f7d8f29b3122962a8d08f0cc03f848cb0b91607'
EXECUTION_INPUT_MANIFEST_SHA256='8e62ffa5ffe56d618ea7f72d157714029c6151c9adf2b78196cbb549b55b87c1'
TOOLCHAIN_SHA256='850c094e02d85fee153297ccf8babbe171e3ec47def68ac2976c3473092b36ac'
HARDWARE_SHA256='464f1a4530cb0829854ddbafc0786d12cc9fc98cef1afced51f40679ba27517c'
AUTHORITY_COMMAND_SHA256='b1cbc4eabcd823c20c612eac0dc023f3b7876445026db62ca268c49aa49070f1'
TEST_COMMAND_SHA256='17e2dbca7b1b32c7ae8e3b75a88e4d629b410ad5f81bd641390ee900e46dd692'
PREMATCHER_COMMAND_SHA256='dcc2c910e7684746ce7de95cf76ae126a2d482234217a6a0235b39676a33fdb8'
RESULT_SHA256='488b92632a0fdaa985618a67d03f84b81f69f0d7b33e2af243360f84215e81f5'
TEST_RESULT_SHA256='6d5f78969bcbf3667fb0f0020cc0b28b42ed4e815ab0a20a84124dea6d93a57b'
PREMATCHER_RESULT_SHA256='8d1fd281f079f0287e4ceddfea31a3c51594e8cfb4b0196e2c9fa1b68b236c06'
ZERO='0 0 0 0 0 0 0 0'

DIRECT_PARENTS=(
  docs/internal/garden/seeds/2026-08-28-pireus-apple-cpu-dependency-latency-interface-feasibility.md
  stdlib/hardware/pireus/apple_cpu_dependency_latency_request.sio
  docs/research/pireus_apple_cpu_dependency_latency_request_semantics.md
  docs/research/receipts/pireus_apple_cpu_dependency_latency_request_20260828.md
  docs/research/evidence/pireus_apple_cpu_dependency_latency_request_20260828.txt
)

REQUEST_ARGS=(
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

# Apple/DGX entries recur because the inherited request evaluator consumes
# them once in target-cost slots and again in material-admission slots. Their
# positional roles are distinct even though the bytes are identical.

AUTHORITY_ARGS=("${DIRECT_PARENTS[@]}" "${REQUEST_ARGS[@]}")

fail() {
  printf 'pireus-apple-cpu-dependency-latency-interface-feasibility: FAIL: %s\n' "$*" >&2
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
  local command_sha="$1" source_sha="${2:-${SOURCE_SHA256}}"
  printf '9020 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 %s %s %s %s %s %s %s %s' \
    "$(sha_limbs "${source_sha}")" "${ZERO}" "${ZERO}" \
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
  local expected_decision="$1" frame="$2" decision frame_sha
  frame_sha="$(sha_text "${frame}")"
  decision="$(printf '%s\n' "${frame}" | "${GUARDIAN}")"
  [[ "${decision}" == "${expected_decision}" ]] ||
    fail "Loom decision mismatch: ${decision}"
  printf 'loom_decision=%s frame_sha256=%s\n' "${decision}" "${frame_sha}"
}

authority_command_record() {
  printf '%s' "SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio ${AUTHORITY_ARGS[*]}"
}

test_command_record() {
  printf '%s' "SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_interface_feasibility.sio ${AUTHORITY_ARGS[*]}"
}

prematcher_command_record() {
  printf '%s' "git archive ${EXECUTABLE_COMMIT}; SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio ${AUTHORITY_ARGS[*]}"
}

tamper_command_record() {
  local parent_index="$1" tampered_parent="$2"
  local args=("${AUTHORITY_ARGS[@]}")
  args["${parent_index}"]="${tampered_parent}"
  printf '%s' "SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio ${args[*]}"
}

run_authority() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio \
      "${AUTHORITY_ARGS[@]}"
  )
}

run_test() {
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_interface_feasibility.sio \
      "${AUTHORITY_ARGS[@]}"
  )
}

run_prematcher() {
  local tree="$1"
  mkdir -m 0700 "${tree}"
  (
    set -o pipefail
    git -C "${ROOT}" archive "${EXECUTABLE_COMMIT}" | tar -x -C "${tree}"
  )
  require_hash "${tree}/bin/souc" ad3ee58b3835cccfbf9382fba01498bc61bdcb8402c8ef47c1c3abf26099c008
  require_hash "${tree}/bin/souc-lean-single-x86_64" 6bb6278dd6244faf7fe6c54eae248d503737d03ca1c000dba88e83fea70b26f2
  (
    cd "${tree}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio \
      "${AUTHORITY_ARGS[@]}"
  )
}

run_tamper() {
  local parent_index="$1" tampered_parent="$2"
  local args=("${AUTHORITY_ARGS[@]}")
  args["${parent_index}"]="${tampered_parent}"
  (
    cd "${ROOT}"
    SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run \
      examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio \
      "${args[@]}"
  )
}

[[ -x "${GUARDIAN}" ]] || fail 'native Sounio Loom guardian unavailable'
[[ "${#DIRECT_PARENTS[@]}" -eq 5 ]] || fail 'direct-parent count drift'
[[ "${#REQUEST_ARGS[@]}" -eq 33 ]] || fail 'request argument count drift'
[[ "${#AUTHORITY_ARGS[@]}" -eq 38 ]] || fail 'authority argument count drift'

require_hash "${GUARDIAN}" 208cbb5ded5a8f0ae56e8a1ba3a5f578f06a622d56c16ba211e078818e6e3a60
require_hash "${NODE}" 81925c0995b5c1427b5d538e6a90ca2fdc4daffb786b09af749beaf7369d4e90
[[ "$(node --version)" == 'v22.22.2' ]] || fail 'Node version drift'
require_hash "${ROOT}/scripts/docs/check_docs_registry.mjs" 535a4b87714b1a5a6824d8fa2f88158e734fe1ff420dd9d4c39852860ea4238d
require_hash "${ROOT}/scripts/docs/governance_registry.mjs" e9ef8072ff1f034c1d0ec62ca267d8c6a9aeefc2560bc534d5166a08d0b7b062
require_hash "${ROOT}/scripts/docs/sync_governance_metadata.mjs" 23f5c2c803270cc5c82ca8dc01babfef093e0bf2847a03a7125c748a551afcd1
require_hash "${ROOT}/scripts/dev/check_docs_registry.sh" d53930721b2371dcfaa9b5ae4add81dfae8d0a78eb4699dea93fd586042afdf0
require_hash "${ROOT}/scripts/dev/check_docs_consistency.sh" 21030af5d04e85c94f9ef5653def07b658777442b9e3de04476a716ae7e39c65
require_hash "${ROOT}/scripts/dev/check_offload_policy.sh" f7b6116b537836b7ea1b8d9d8058da42c4af4130ff2e6f8468852d96b100db71
require_hash "${ROOT}/tools/loom/LANGUAGE_AUTHORITY_V1.md" 6e15171863331b0b86e7b4948ce11d2194f31f1ed24b2bf93ffacaa4928eeac6
require_hash "${ROOT}/stdlib/coordination/loom_language_authority.sio" 64bb0118793fe46dcb392abc1a9212eb15bd55047461576a3ef1a6cefa3f17da
require_hash "${ROOT}/tools/loom/language_authority.freeze.v1" 5fe5e5c9cdcb83935770f58df52f2d614d11f8abde519c4a2505ca20998fae2e
require_hash "${ROOT}/scripts/ci/sounio_loom_language_authority_freeze_selftest.sh" e7a9074cc1f9b7852bc9e8174f05d05953eaa8e1c5bde7d5fa02ec210082eb58
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
require_hash "${REQUEST_SOURCE}" 3a92553aae0c4a9606f4964e1613a31452b7fb4d197b128dbb73cc24b87b550e
require_hash "${REQUEST_SEMANTICS}" 9bd767db814e47bfc087e07c0f9ff33b65faea5b885ae0f8ed3a6e646c015e6d
require_hash "${REQUEST_RECEIPT}" 0ee12f3502efb26056bdbcf850360c0a5df727627a3c67499d363744f7c73272
require_hash "${REQUEST_EVIDENCE}" cf4455690426038cc7477b673bcf763e9755e8147f1ff55e882086826626482b

git -C "${ROOT}" cat-file -e "${GARDEN_COMMIT}^{commit}" || fail 'Garden commit missing'
git -C "${ROOT}" cat-file -e "${EXECUTABLE_COMMIT}^{commit}" || fail 'first executable commit missing'
git -C "${ROOT}" merge-base --is-ancestor "${GARDEN_COMMIT}" "${EXECUTABLE_COMMIT}" ||
  fail 'Garden is not an ancestor of first executable'
git -C "${ROOT}" merge-base --is-ancestor "${EXECUTABLE_COMMIT}" HEAD ||
  fail 'first executable is not an ancestor of HEAD'
[[ "$(git -C "${ROOT}" show "${EXECUTABLE_COMMIT}:stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio" | sha256sum | cut -d' ' -f1)" == "${FIRST_EXECUTABLE_SOURCE_SHA256}" ]] ||
  fail 'first executable source hash drift'
if git -C "${ROOT}" show \
  "${EXECUTABLE_COMMIT}:stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio" |
  grep -Fq 'pireus_apple_cpu_dependency_latency_interface_feasibility_matches_frozen_semantics'; then
  fail 'exact matcher exists in first executable commit'
fi
grep -Fq 'pireus_apple_cpu_dependency_latency_interface_feasibility_matches_frozen_semantics' "${MODULE}" ||
  fail 'exact matcher missing from frozen module'

PARENT_MANIFEST="$({
  cd "${ROOT}"
  sha256sum "${DIRECT_PARENTS[@]}"
})"
[[ "$(sha_text "${PARENT_MANIFEST}")" == "${PARENT_MANIFEST_SHA256}" ]] ||
  fail 'direct-parent manifest drift'

SOURCE_MANIFEST="$({
  cd "${ROOT}"
  sha256sum \
    stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio \
    examples/pireus_apple_cpu_dependency_latency_interface_feasibility.sio \
    tests/stdlib/hardware/test_pireus_apple_cpu_dependency_latency_interface_feasibility.sio
})"
[[ "$(sha_text "${SOURCE_MANIFEST}")" == "${SOURCE_MANIFEST_SHA256}" ]] ||
  fail 'source manifest drift'

EXECUTION_INPUT_MANIFEST="$({
  cd "${ROOT}"
  sha256sum "${AUTHORITY_ARGS[@]}"
})"
[[ "$(sha_text "${EXECUTION_INPUT_MANIFEST}")" == "${EXECUTION_INPUT_MANIFEST_SHA256}" ]] ||
  fail 'execution-input manifest drift'

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
[[ "$(sha_text_exact "$(prematcher_command_record)")" == "${PREMATCHER_COMMAND_SHA256}" ]] ||
  fail 'pre-matcher command record drift'

ALLOW_FROZEN='SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SEMANTICS_FROZEN'
ALLOW_EXECUTABLE='SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=SOUNIO_EXECUTABLE'
authorize "${ALLOW_FROZEN}" "$(freeze_frame)"
authorize "${ALLOW_FROZEN}" \
  "$(stage_three_frame 8 1 1 "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" "$(sha_limbs "${RESULT_SHA256}")")"
authorize "${ALLOW_FROZEN}" \
  "$(stage_three_frame 8 1 1 "$(sha_limbs "${TEST_COMMAND_SHA256}")" "$(sha_limbs "${TEST_RESULT_SHA256}")")"
authorize "${ALLOW_FROZEN}" \
  "$(stage_three_frame 9 1 1 "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" "$(sha_limbs "${RESULT_SHA256}")")"
authorize "${ALLOW_FROZEN}" \
  "$(stage_three_frame 10 1 1 "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" "$(sha_limbs "${RESULT_SHA256}")")"
authorize "${ALLOW_FROZEN}" "$(stage_three_frame 5 6 6 "${ZERO}" "${ZERO}")"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pireus-apple-interface-freeze.XXXXXX")"
[[ -d "${WORK_DIR}" ]] || fail 'private work directory creation failed'
trap 'rm -rf -- "${WORK_DIR}"' EXIT

authorize "${ALLOW_EXECUTABLE}" \
  "$(preaction_frame "${PREMATCHER_COMMAND_SHA256}" "${FIRST_EXECUTABLE_SOURCE_SHA256}")" >/dev/null
run_prematcher "${WORK_DIR}/pre-matcher-tree" >"${WORK_DIR}/pre-matcher.txt"
require_hash "${WORK_DIR}/pre-matcher.txt" "${PREMATCHER_RESULT_SHA256}"
[[ "$(wc -l <"${WORK_DIR}/pre-matcher.txt")" == 141 ]] || fail 'pre-matcher line count drift'
[[ "$(wc -c <"${WORK_DIR}/pre-matcher.txt")" == 2499 ]] || fail 'pre-matcher byte count drift'
require_line "${WORK_DIR}/pre-matcher.txt" ' matched=5'
require_line "${WORK_DIR}/pre-matcher.txt" 'PIREUS_APPLE_INTERFACE_ONTOLOGY triples=25'
require_line "${WORK_DIR}/pre-matcher.txt" 'PIREUS_APPLE_INTERFACE_NEGATIVES passed=32'
require_line "${WORK_DIR}/pre-matcher.txt" 'PIREUS_APPLE_INTERFACE_SUMMARY error=0'
require_line "${WORK_DIR}/pre-matcher.txt" ' failures=0'

for run in 1 2 3; do
  authorize "${ALLOW_EXECUTABLE}" "$(preaction_frame "${AUTHORITY_COMMAND_SHA256}")" >/dev/null
  run_authority >"${WORK_DIR}/authority-${run}.txt"
  require_hash "${WORK_DIR}/authority-${run}.txt" "${RESULT_SHA256}"
done
cmp -s "${WORK_DIR}/authority-1.txt" "${WORK_DIR}/authority-2.txt" ||
  fail 'authority runs 1 and 2 differ'
cmp -s "${WORK_DIR}/authority-2.txt" "${WORK_DIR}/authority-3.txt" ||
  fail 'authority runs 2 and 3 differ'
[[ "$(wc -l <"${WORK_DIR}/authority-1.txt")" == 143 ]] || fail 'authority line count drift'
[[ "$(wc -c <"${WORK_DIR}/authority-1.txt")" == 2537 ]] || fail 'authority byte count drift'
require_line "${WORK_DIR}/authority-1.txt" 'SOUNIO_SEMANTIC_AUTHORITY schema=pireus.apple-cpu-interface-feasibility.v0 stage=SEMANTICS_FROZEN'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_INTERFACE_FROZEN match=1'
require_line "${WORK_DIR}/authority-1.txt" ' matched=5'
require_line "${WORK_DIR}/authority-1.txt" ' request_live=1'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_INTERFACE_MANIFEST families=6'
require_line "${WORK_DIR}/authority-1.txt" ' material_candidates=0'
require_line "${WORK_DIR}/authority-1.txt" ' verdict=709510'
require_line "${WORK_DIR}/authority-1.txt" ' remote_execution=0'
require_line "${WORK_DIR}/authority-1.txt" ' parity_open=0'
require_line "${WORK_DIR}/authority-1.txt" ' claim_ready=0'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_INTERFACE_ONTOLOGY triples=25'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_INTERFACE_NEGATIVES passed=32'
require_line "${WORK_DIR}/authority-1.txt" 'PIREUS_APPLE_INTERFACE_SUMMARY error=0'
require_line "${WORK_DIR}/authority-1.txt" ' failures=0'

authorize "${ALLOW_EXECUTABLE}" "$(preaction_frame "${TEST_COMMAND_SHA256}")" >/dev/null
run_test >"${WORK_DIR}/test.txt"
require_hash "${WORK_DIR}/test.txt" "${TEST_RESULT_SHA256}"
require_line "${WORK_DIR}/test.txt" 'pireus Apple CPU interface feasibility test passed'

for parent_index in 0 1 2 3 4; do
  TAMPER_PARENT="${WORK_DIR}/direct-parent-${parent_index}.txt"
  TAMPER_OUTPUT="${WORK_DIR}/tamper-${parent_index}.txt"
  cp "${ROOT}/${DIRECT_PARENTS[${parent_index}]}" "${TAMPER_PARENT}"
  printf '\nPIREUS_APPLE_INTERFACE_PARENT_TAMPER index=%s\n' "${parent_index}" >>"${TAMPER_PARENT}"
  TAMPER_COMMAND_SHA256="$(sha_text_exact "$(tamper_command_record "${parent_index}" "${TAMPER_PARENT}")")"
  authorize "${ALLOW_FROZEN}" \
    "$(stage_three_frame 11 1 1 "$(sha_limbs "${TAMPER_COMMAND_SHA256}")" "${ZERO}")" >/dev/null
  set +e
  run_tamper "${parent_index}" "${TAMPER_PARENT}" >"${TAMPER_OUTPUT}"
  TAMPER_STATUS=$?
  set -e
  [[ "${TAMPER_STATUS}" -eq 1 ]] ||
    fail "direct-parent ${parent_index} tamper exit ${TAMPER_STATUS}, expected 1"
  require_line "${TAMPER_OUTPUT}" ' matched=4'
  require_line "${TAMPER_OUTPUT}" "PIREUS_APPLE_INTERFACE_PARENT_FILE index=${parent_index}"
  require_line "${TAMPER_OUTPUT}" ' match=0'
  require_line "${TAMPER_OUTPUT}" 'PIREUS_APPLE_INTERFACE_FROZEN match=0'
  require_line "${TAMPER_OUTPUT}" 'PIREUS_APPLE_INTERFACE_SUMMARY error=1'
  require_line "${TAMPER_OUTPUT}" ' failures=3'
done

# Raw mutation is exercised at this child's five direct causal boundaries.
# All 38 direct plus inherited execution inputs are independently byte-pinned
# by EXECUTION_INPUT_MANIFEST_SHA256; the frozen request parent owns its own
# raw-parent sabotage closure.

PYTHON_FRAME="$(stage_three_frame 4 7 7 "$(sha_limbs "${AUTHORITY_COMMAND_SHA256}")" "$(sha_limbs "${RESULT_SHA256}")")"
set +e
PYTHON_DECISION="$(printf '%s\n' "${PYTHON_FRAME}" | "${GUARDIAN}")"
PYTHON_STATUS=$?
set -e
[[ "${PYTHON_STATUS}" -eq 110 ]] || fail "Python refusal exit ${PYTHON_STATUS}"
[[ "${PYTHON_DECISION}" == \
  'SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=SEMANTICS_FROZEN' ]] ||
  fail "Python refusal decision drift: ${PYTHON_DECISION}"

require_line "${EVIDENCE}" 'stage=SEMANTICS_FROZEN'
require_line "${EVIDENCE}" 'parent_file_count=5'
require_line "${EVIDENCE}" 'parent_file_match_count=5'
require_line "${EVIDENCE}" 'request_parent_live=true'
require_line "${EVIDENCE}" 'material_candidate_count=0'
require_line "${EVIDENCE}" 'verdict_id=709510'
require_line "${EVIDENCE}" 'verdict=UNASSESSED'
require_line "${EVIDENCE}" 'negative_passed=32'
require_line "${EVIDENCE}" 'negative_total=32'
require_line "${EVIDENCE}" 'target_transport_contacted=false'
require_line "${EVIDENCE}" 'material_probe_executed=false'
require_line "${EVIDENCE}" 'external_llm_confirmed_result=false'
require_line "${EVIDENCE}" 'python_oracle_executed=false'
require_line "${EVIDENCE}" 'rust_oracle_executed=false'
require_line "${EVIDENCE}" 'waiver_requested=false'
require_line "${EVIDENCE}" 'next_stage=PARITY_OPEN'
require_line "${REQUEST_EVIDENCE}" 'apple_login_locator=demetrios@sounio-language-macbook'
require_line "${REQUEST_EVIDENCE}" 'apple_locator_in_binding=false'
grep -Fq 'No C++ producer was executed.' "${RECEIPT}" ||
  fail 'no-C++ boundary missing from receipt'
grep -Fq 'No SSH or tailnet connection was opened.' "${RECEIPT}" ||
  fail 'no-remote boundary missing from receipt'
grep -Fq 'cannot be promoted to semantic' "${RECEIPT}" ||
  fail 'review-only boundary missing from receipt'

(
  cd "${ROOT}"
  node scripts/docs/check_docs_registry.mjs >/dev/null
  bash scripts/dev/check_docs_registry.sh >/dev/null
  bash scripts/dev/check_docs_consistency.sh >/dev/null
  bash scripts/dev/check_offload_policy.sh --files \
    stdlib/hardware/pireus/apple_cpu_dependency_latency_interface_feasibility.sio \
    docs/research/pireus_apple_cpu_dependency_latency_interface_feasibility_semantics.md \
    docs/research/receipts/pireus_apple_cpu_dependency_latency_interface_feasibility_20260828.md \
    docs/research/evidence/pireus_apple_cpu_dependency_latency_interface_feasibility_20260828.txt \
    scripts/ci/pireus_apple_cpu_dependency_latency_interface_feasibility.sh >/dev/null
)

printf 'PIREUS_APPLE_CPU_DEPENDENCY_LATENCY_INTERFACE_FEASIBILITY_GATE_PASS schema_freeze=true empirical_verdict=UNASSESSED parents=5 execution_inputs_pinned=38 request_live=true candidates=0 ontology=25 negatives=32 deterministic_replays=3 raw_direct_parent_tampers=5 python_e110=PASS remote_execution=false material_probe=false next_stage=PARITY_OPEN claim_ready=false\n'
